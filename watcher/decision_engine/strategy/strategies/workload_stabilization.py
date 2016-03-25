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
import oslo_cache

from copy import deepcopy
from oslo_config import cfg
from oslo_log import log
from watcher._i18n import _LE, _LI, _
from watcher.common import exception
from watcher.decision_engine.model import vm_state
from watcher.decision_engine.strategy.strategies import base
from watcher.metrics_engine.cluster_history import ceilometer as \
    ceilometer_cluster_history

LOG = log.getLogger(__name__)

metrics = ['cpu_util', 'memory.resident']
thresholds_dict = {'cpu_util': 0.2, 'memory.resident': 0.2}
weights_dict = {'cpu_util_weight': 1.0, 'memory.resident_weight': 1.0}

ws_opts = [
    cfg.ListOpt('metrics',
                default=metrics,
                required=True,
                help='Metrics used as rates of cluster loads.'),
    cfg.DictOpt('thresholds',
                default=thresholds_dict,
                help=''),
    cfg.DictOpt('weights',
                default=weights_dict,
                help='These weights used to calculate '
                     'common standard deviation. Name of weight '
                     'contains meter name and _weight suffix.'),
]

CONF = cfg.CONF

CONF.register_opts(ws_opts, 'watcher_workload_stabilization')


def _set_memoize(conf):
    oslo_cache.configure(conf)
    region = oslo_cache.create_region()
    configured_region = oslo_cache.configure_cache_region(conf, region)
    return oslo_cache.core.get_memoization_decorator(conf,
                                                     configured_region,
                                                     'cache')


class WorkloadStabilization(base.BaseStrategy):
    """Workload Stabilization control using live migration

    *Description*

    This is workload stabilization strategy based on standard deviation
    algorithm. The goal is to determine if there is an overload in a cluster
    and respond to it by migrating VMs to stabilize the cluster.

    *Requirements*

    * Software: Ceilometer component ceilometer-compute running
      in each compute host, and Ceilometer API can report such telemetries
      ``memory.resident`` and ``cpu_util`` successfully.
    * You must have at least 2 physical compute nodes to run this strategy.

    *Limitations*

    - It assume that live migrations are possible

    *Spec URL*

    https://review.openstack.org/#/c/286153/
    """

    MIGRATION = "migrate"
    DEFAULT_NAME = "workload_stabilization"
    DEFAULT_DESCRIPTION = "Workload Stabilization Algorithm"
    MEMOIZE = _set_memoize(CONF)

    def __init__(self, name=DEFAULT_NAME, description=DEFAULT_DESCRIPTION,
                 osc=None):
        super(WorkloadStabilization, self).__init__(name, description, osc)
        self._ceilometer = None
        self._nova = None
        self.weights = CONF.watcher_workload_stabilization.weights
        self.metrics = CONF.watcher_workload_stabilization.metrics
        self.thresholds = CONF.watcher_workload_stabilization.thresholds

    @property
    def ceilometer(self):
        if self._ceilometer is None:
            self._ceilometer = (ceilometer_cluster_history.
                                CeilometerClusterHistory(osc=self.osc))
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

    def transform_vm_cpu(self, vm_load, host_vcpus):
        """This method transforms vm cpu utilization to overall host cpu utilization.

        :param vm_load: dict that contains vm uuid and utilization info.
        :param host_vcpus: int
        :return: float value
        """
        return vm_load['cpu_util'] * (vm_load['vcpus']/float(host_vcpus))

    @MEMOIZE
    def get_vm_load(self, vm_uuid):
        """Gathering vm load through ceilometer statistic.

        :param vm_uuid: vm for which statistic is gathered.
        :return: dict
        """
        LOG.info(_LI('get_vm_load started'))
        flavor_id = self.nova.servers.get(vm_uuid).flavor['id']
        vm_vcpus = self.nova.flavors.get(flavor_id).vcpus
        vm_load = {'uuid': vm_uuid, 'vcpus': vm_vcpus}
        for meter in self.metrics:
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
            vm_load[meter] = avg_meter or 0
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
        """Get load of every host by gathering vms load"""
        hosts_load = {}
        for hypervisor_id in current_model.get_all_hypervisors():
            hosts_load[hypervisor_id] = {}
            h_id = self.nova.hypervisors.search(hypervisor_id)[0].id
            host_vcpus = self.nova.hypervisors.get(h_id).vcpus
            hosts_load[hypervisor_id]['vcpus'] = host_vcpus

            for metric in self.metrics:
                hosts_load[hypervisor_id][metric] = 0

            vm_ids = current_model.get_mapping(). \
                get_node_vms_from_id(hypervisor_id)
            for vm_id in vm_ids:
                vm = current_model.get_vm_from_id(vm_id)

                if vm.state != vm_state.VMState.ACTIVE.value:
                    continue

                vm_load = self.get_vm_load(vm_id)
                if 'cpu_util' in vm_load:
                    vm_load['cpu_util'] = \
                        self.transform_vm_cpu(vm_load, host_vcpus)

                for metric in self.metrics:
                    hosts_load[hypervisor_id][metric] += vm_load[metric]
        return hosts_load

    def get_sd(self, hosts, meter_name):
        """Get standard deviation among hosts by specified meter"""
        mean = 0
        variaton = 0
        for x in hosts:
            mean += hosts[x][meter_name]
        mean /= len(hosts)
        for x in hosts:
            variaton += (hosts[x][meter_name] - mean) ** 2
        variaton /= len(hosts)
        sd = math.sqrt(variaton)
        return sd

    def calculate_weighted_sd(self, sd_case):
        """Calculate common standard deviation among meters on host"""
        weighted_sd = 0
        for metric, value in zip(self.metrics, sd_case):
            try:
                weighted_sd += value * float(self.weights[metric+'_weight'])
            except KeyError as exc:
                LOG.exception(exc)
                raise exception.WatcherException(
                    _("Incorrect mapping: could not find associated weight"
                      " for %s in weight dict.") % metric)
        return weighted_sd

    def calculate_migration_case(self, hosts, vm_id, src_hp_id, dst_hp_id):
        """Calculate migration case

        Return list of standard deviation values, that appearing in case of
        migration of vm from source host to destination host
        :param hosts: hosts with their workload
        :param vm_id: the virtual machine
        :param src_hp_id: the source hypervisor id
        :param dst_hp_id: the destination hypervisor id
        :return: list of standard deviation values
        """
        migration_case = []
        new_hosts = deepcopy(hosts)
        vm_load = self.get_vm_load(vm_id)
        d_host_vcpus = new_hosts[dst_hp_id]['vcpus']
        s_host_vcpus = new_hosts[src_hp_id]['vcpus']
        for metric in self.metrics:
            if metric is 'cpu_util':
                new_hosts[src_hp_id][metric] -= self.transform_vm_cpu(
                    vm_load,
                    s_host_vcpus)
                new_hosts[dst_hp_id][metric] += self.transform_vm_cpu(
                    vm_load,
                    d_host_vcpus)
            else:
                new_hosts[src_hp_id][metric] -= vm_load[metric]
                new_hosts[dst_hp_id][metric] += vm_load[metric]
        normalized_hosts = self.normalize_hosts_load(new_hosts)
        for metric in self.metrics:
            migration_case.append(self.get_sd(normalized_hosts, metric))
        migration_case.append(new_hosts)
        return migration_case

    def simulate_migrations(self, current_model, hosts):
        """Make sorted list of pairs vm:dst_host"""
        vm_host_map = []
        for source_hp_id in current_model.get_all_hypervisors():
            vms_id = current_model.get_mapping(). \
                get_node_vms_from_id(source_hp_id)
            for vm_id in vms_id:
                min_sd_case = {'value': len(self.metrics)}
                vm = current_model.get_vm_from_id(vm_id)
                if vm.state != vm_state.VMState.ACTIVE.value:
                    continue
                for dst_hp_id in current_model.get_all_hypervisors():
                    if source_hp_id == dst_hp_id:
                        continue
                    sd_case = self.calculate_migration_case(hosts, vm_id,
                                                            source_hp_id,
                                                            dst_hp_id)

                    weighted_sd = self.calculate_weighted_sd(sd_case[:-1])

                    if weighted_sd < min_sd_case['value']:
                        min_sd_case = {'host': dst_hp_id, 'value': weighted_sd,
                                       's_host': source_hp_id, 'vm': vm_id}
                    LOG.debug('Weighted SD: {0}'.format(weighted_sd))
                vm_host_map.append(min_sd_case)
        return sorted(vm_host_map, key=lambda x: x['value'])

    def check_threshold(self, current_model):
        """Check if cluster is needed in balancing"""
        hosts_load = self.get_hosts_load(current_model)
        normalized_load = self.normalize_hosts_load(hosts_load)
        for metric in self.metrics:
            metric_sd = self.get_sd(normalized_load, metric)
            if metric_sd > float(self.thresholds[metric]):
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

    def fill_solution(self, current_model):
        self.solution.model = current_model
        self.solution.efficacy = 100
        return self.solution

    def execute(self, orign_model):
        current_model = orign_model

        if orign_model is None:
            raise exception.ClusterStateNotDefined()

        migration = self.check_threshold(current_model)
        if migration:
            hosts_load = self.get_hosts_load(current_model)
            min_sd = 1
            balanced = False
            for vm_host in migration:
                vm_load = self.calculate_migration_case(hosts_load,
                                                        vm_host['vm'],
                                                        vm_host['s_host'],
                                                        vm_host['host'])
                weighted_sd = self.calculate_weighted_sd(vm_load[:-1])
                if weighted_sd < min_sd:
                    min_sd = weighted_sd
                    hosts_load = vm_load[-1]
                    self.migrate(current_model, vm_host['vm'],
                                 vm_host['s_host'], vm_host['host'])

                for metric, value in zip(self.metrics, vm_load[:-1]):
                    if value < float(self.thresholds[metric]):
                        balanced = True
                        break
                if balanced:
                    return self.fill_solution(current_model)
        return self.fill_solution(current_model)
