# -*- encoding: utf-8 -*-
# Copyright (c) 2016 Servionica LLC
#
# Authors: Alexander Chadin <a.chadin@servionica.ru>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
from copy import deepcopy

import oslo_cache
from oslo_config import cfg
from oslo_log import log

from watcher._i18n import _LE
from watcher.common import exception
from watcher.decision_engine.model.vm_state import VMState
from watcher.decision_engine.strategy.strategies.base import BaseStrategy
from watcher.metrics_engine.cluster_history.ceilometer import \
    CeilometerClusterHistory

LOG = log.getLogger(__name__)


sd_opts = [
    cfg.FloatOpt('cpu_threshold',
                 default=0.2,
                 help='Max threshold standard deviation values for cpu.'),
    cfg.FloatOpt('ram_threshold',
                 default=0.2,
                 help='Max threshold standard deviation values for ram.'),
    cfg.FloatOpt('cpu_weight',
                 default=1.0,
                 help=''),
    cfg.FloatOpt('ram_weight',
                 default=1.0,
                 help=''),
]

CONF = cfg.CONF

CONF.register_opts(sd_opts, 'watcher_standard_deviation')


def _set_memoize(conf):
    oslo_cache.configure(conf)
    region = oslo_cache.create_region()
    configured_region = oslo_cache.configure_cache_region(conf, region)
    return oslo_cache.core.get_memoization_decorator(conf,
                                                     configured_region,
                                                     'cache')


class StandardDeviation(BaseStrategy):
    MIGRATION = "migrate"
    DEFAULT_NAME = "sd"
    DEFAULT_DESCRIPTION = "Standard Deviation Algorithm"
    MEMOIZE = _set_memoize(CONF)

    def __init__(self, name=DEFAULT_NAME, description=DEFAULT_DESCRIPTION,
                 osc=None):
        super(StandardDeviation, self).__init__(name, description, osc)
        self._ceilometer = None
        self._nova = None
        self.cpu_threshold = CONF.watcher_standard_deviation.cpu_threshold
        self.ram_threshold = CONF.watcher_standard_deviation.ram_threshold
        self.cpu_weight = CONF.watcher_standard_deviation.cpu_weight
        self.ram_weight = CONF.watcher_standard_deviation.ram_weight

    @property
    def ceilometer(self):
        if self._ceilometer is None:
            self._ceilometer = CeilometerClusterHistory(osc=self.osc)
        return self._ceilometer

    @property
    def nova(self):
        if self._nova is None:
            self._nova = self.osc.nova()
        return self._nova

    @nova.setter
    def nova(self, n):
        self._nova = n

    @ceilometer.setter
    def ceilometer(self, c):
        self._ceilometer = c

    def transform_vm_cpu_to_host_cpu(self, vm_load, host_vcpus):
        """
        This method transforms vm cpu utilization
        to overall host cpu utilization.

        :param vm_load: dict that contains vm uuid and utilization info.
        :param host_vcpus: int
        """

        return vm_load['cpu_util'] * (vm_load['vcpus']/float(host_vcpus))

    @MEMOIZE
    def get_vm_load(self, vm_uuid):
        LOG.warning('get_vm_load started')
        flavor_id = self.nova.servers.get(vm_uuid).flavor['id']
        vm_vcpus = self.nova.flavors.get(flavor_id).vcpus
        vm_load = {'uuid': vm_uuid, 'vcpus': vm_vcpus}
        for meter in ['cpu_util', 'memory.resident']:
            avg_meter = self.ceilometer.statistic_aggregation(
                            resource_id=vm_uuid,
                            meter_name=meter,
                            period="120",
                            aggregate='avg'
                            )
            if avg_meter is None:
                LOG.error(
                    _LE("No values returned by %(resource_id)s "
                        "for %(metric_name)s"),
                    resource_id=vm_uuid,
                    metric_name=avg_meter,
                )
            vm_load[meter] = avg_meter if avg_meter else 0
        return vm_load

    def normalize_hosts_load(self, hosts):
        normalized_hosts = deepcopy(hosts)
        for host in normalized_hosts:
            if 'cpu_util' in normalized_hosts[host]:
                normalized_hosts[host]['cpu_util'] /= float(100)

            if 'memory.resident' in normalized_hosts[host]:
                h_id = self.nova.hypervisors.search(host)[0].id
                h_memory = self.nova.hypervisors.get(h_id).memory_mb
                normalized_hosts[host]['memory.resident'] /= float(h_memory)

        return normalized_hosts

    def get_hosts_load(self, current_model):
        hosts_load = {}
        for hypervisor_id in current_model.get_all_hypervisors():
            hosts_load[hypervisor_id] = {}
            h_id = self.nova.hypervisors.search(hypervisor_id)[0].id
            host_vcpus = self.nova.hypervisors.get(h_id).vcpus
            hosts_load[hypervisor_id]['vcpus'] = host_vcpus

            for metric in ['cpu_util', 'memory.resident']:
                hosts_load[hypervisor_id][metric] = 0

            vms_id = current_model.get_mapping(). \
                get_node_vms_from_id(hypervisor_id)
            for vm_id in vms_id:
                vm = current_model.get_vm_from_id(vm_id)

                if vm.state != VMState.ACTIVE.value:
                    continue

                vm_load = self.get_vm_load(vm_id)
                if 'cpu_util' in vm_load:
                    vm_load['cpu_util'] = \
                        self.transform_vm_cpu_to_host_cpu(vm_load, host_vcpus)

                LOG.warning("{0}: {1}, {2}".format(vm_load['uuid'],
                                                   vm_load['cpu_util'],
                                                   vm_load['memory.resident']))

                for metric in ['cpu_util', 'memory.resident']:
                    hosts_load[hypervisor_id][metric] += vm_load[metric]
        LOG.warning(hosts_load)
        return hosts_load

    def get_sd(self, hosts, meter_name):
        mean = reduce(lambda res, x: res + hosts[x][meter_name],
                      hosts, 0) / len(hosts)
        variaton = float(reduce(
            lambda res, x: res + (hosts[x][meter_name] - mean) ** 2,
            hosts, 0)) / len(hosts)
        sd = math.sqrt(variaton)
        return sd

    def calculate_migration_case(self, hosts, vm_id, src_hp_id, dst_hp_id):
        migration_case = []
        new_hosts = deepcopy(hosts)
        vm_load = self.get_vm_load(vm_id)
        d_host_vcpus = new_hosts[dst_hp_id]['vcpus']
        s_host_vcpus = new_hosts[src_hp_id]['vcpus']
        transform_method = self.transform_vm_cpu_to_host_cpu
        for metric in ['cpu_util', 'memory.resident']:
            if metric is 'cpu_util':
                new_hosts[src_hp_id][metric] -= transform_method(vm_load,
                                                                 s_host_vcpus)
                new_hosts[dst_hp_id][metric] += transform_method(vm_load,
                                                                 d_host_vcpus)
            else:
                new_hosts[src_hp_id][metric] -= vm_load[metric]
                new_hosts[dst_hp_id][metric] += vm_load[metric]
        LOG.warning(new_hosts)
        normalized_hosts = self.normalize_hosts_load(new_hosts)
        for metric in ['cpu_util', 'memory.resident']:
            migration_case.append(self.get_sd(normalized_hosts, metric))
        LOG.warning('calculate_migration_case {0} {1}'.format(
            migration_case[0], migration_case[1]))
        migration_case.append(new_hosts)
        return migration_case

    def simulate_migrations(self, current_model, hosts):
        vm_host_map = []
        for source_hp_id in current_model.get_all_hypervisors():
            vms_id = current_model.get_mapping(). \
                get_node_vms_from_id(source_hp_id)
            for vm_id in vms_id:
                min_sd_case = {'value': 1}
                vm = current_model.get_vm_from_id(vm_id)
                if vm.state != VMState.ACTIVE.value:
                    continue
                for dst_hp_id in current_model.get_all_hypervisors():
                    if source_hp_id == dst_hp_id:
                        continue
                    sd_case = self.calculate_migration_case(hosts, vm_id,
                                                            source_hp_id,
                                                            dst_hp_id)
                    common_sd = self.cpu_weight * sd_case[0] + \
                        self.ram_weight * sd_case[1]
                    if common_sd < min_sd_case['value']:
                        min_sd_case = {'host': dst_hp_id, 'value': common_sd,
                                       's_host': source_hp_id, 'vm': vm_id}
                    LOG.warning(common_sd)
                    LOG.warning(min_sd_case)
                vm_host_map.append(min_sd_case)
        return sorted(vm_host_map, key=lambda x: x['value'])

    def check_threshold(self, current_model):
        hosts_load = self.get_hosts_load(current_model)
        normalized_load = self.normalize_hosts_load(hosts_load)
        cpu_sd = self.get_sd(normalized_load, 'cpu_util')
        ram_sd = self.get_sd(normalized_load, 'memory.resident')
        LOG.warning("cpu_sd: {0}, mem_sd: {1}".format(cpu_sd, ram_sd))
        if cpu_sd > self.cpu_threshold or ram_sd > self.ram_threshold:
            return self.simulate_migrations(current_model, hosts_load)

    def add_migration(self,
                      resource_id,
                      migration_type,
                      src_hypervisor,
                      dst_hypervisor):
        parameters = {'migration_type': migration_type,
                      'src_hypervisor': src_hypervisor,
                      'dst_hypervisor': dst_hypervisor}
        self.solution.add_action(action_type=self.MIGRATION,
                                 resource_id=resource_id,
                                 input_parameters=parameters)

    def create_migration_vm(self, current_model, mig_vm, mig_src_hypervisor,
                            mig_dst_hypervisor):
        """Create migration VM """
        if current_model.get_mapping().migrate_vm(
                mig_vm, mig_src_hypervisor, mig_dst_hypervisor):
            self.add_migration(mig_vm.uuid, 'live',
                               mig_src_hypervisor.uuid,
                               mig_dst_hypervisor.uuid)

    def migrate(self, current_model, vm_uuid, src_host, dst_host):
        mig_vm = current_model.get_vm_from_id(vm_uuid)
        mig_src_hypervisor = current_model.get_hypervisor_from_id(src_host)
        mig_dst_hypervisor = current_model.get_hypervisor_from_id(dst_host)
        self.create_migration_vm(current_model, mig_vm, mig_src_hypervisor,
                                 mig_dst_hypervisor)

    def execute(self, orign_model):
        current_model = orign_model

        if orign_model is None:
            raise exception.ClusterStateNotDefined()

        migration = self.check_threshold(current_model)
        LOG.warning(migration)
        if migration:
            hosts_load = self.get_hosts_load(current_model)
            min_cpu = 1
            min_ram = 1
            for vm_host in migration:
                vm_load = self.calculate_migration_case(hosts_load,
                                                        vm_host['vm'],
                                                        vm_host['s_host'],
                                                        vm_host['host'])
                if vm_load[0] < min_cpu or vm_load[1] < min_ram:
                    min_cpu = vm_load[0] if vm_load[0] < min_cpu else min_cpu
                    min_ram = vm_load[1] if vm_load[1] < min_ram else min_ram
                    hosts_load = vm_load[-1]
                    self.migrate(current_model, vm_host['vm'],
                                 vm_host['s_host'], vm_host['host'])
                if min_cpu < self.cpu_threshold or \
                   min_ram < self.ram_threshold:
                    break
        self.solution.model = current_model
        self.solution.efficacy = 100
        return self.solution
