# -*- encoding: utf-8 -*-
# Copyright (c) 2016 Servionica LLC
#
# Author: Alexander Chadin <a.chadin@servionica.ru>
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

import mock

from watcher.decision_engine.strategy import strategies
from watcher.tests import base
from watcher.tests.decision_engine.strategy.strategies \
    import faker_cluster_state
from watcher.tests.decision_engine.strategy.strategies \
    import faker_metrics_collector


class TestStandardDeviation(base.BaseTestCase):
    # fake metrics
    fake_metrics = faker_metrics_collector.FakerMetricsCollector()

    # fake cluster
    fake_cluster = faker_cluster_state.FakerModelCollector()

    hosts_load_assert = {'Node_0':
                         {'cpu_util': 7.0, 'memory.resident': 7, 'vcpus': 2},
                         'Node_1':
                         {'cpu_util': 5.0, 'memory.resident': 5, 'vcpus': 2},
                         'Node_2':
                         {'cpu_util': 10.0, 'memory.resident': 29, 'vcpus': 2},
                         'Node_3':
                         {'cpu_util': 4.0, 'memory.resident': 8, 'vcpus': 2},
                         'Node_4':
                         {'cpu_util': 2.0, 'memory.resident': 4, 'vcpus': 2}}

    def test_standard_deviation_get_vm_load(self):
        sd = strategies.StandardDeviation()
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        vm_0_dict = {'uuid': 'VM_0', 'vcpus': 1,
                     'cpu_util': 7, 'memory.resident': 2}
        self.assertEqual(sd.get_vm_load("VM_0"), vm_0_dict)

    def test_standard_deviation_normalize_hosts_load(self):
        sd = strategies.StandardDeviation()
        fake_hosts = {'compute1': {'cpu_util': 86, 'memory.resident': 1200},
                      'compute2': {'cpu_util': 50, 'memory.resident': 1600}}
        normalized_hosts = {'compute1':
                            {'cpu_util': 0.86, 'memory.resident': 0.60},
                            'compute2':
                            {'cpu_util': 0.5, 'memory.resident': 0.80}}
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        self.assertEqual(sd.normalize_hosts_load(fake_hosts), normalized_hosts)

    def test_standard_deviation_get_hosts_load(self):
        sd = strategies.StandardDeviation()
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().vcpus = 2
        self.assertEqual(
            sd.get_hosts_load(self.fake_cluster.generate_scenario_1()),
            self.hosts_load_assert)

    def test_standard_deviation_get_hosts_load_without_vms(self):
        sd = strategies.StandardDeviation()
        model = self.fake_cluster.generate_scenario_4_with_1_hypervisor_no_vm()
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().vcpus = 2
        self.assertEqual(sd.get_hosts_load(model),
                         {'Node_0': {'cpu_util': 0, 'vcpus': 2,
                                     'memory.resident': 0}})

    def test_standard_deviation_get_sd(self):
        sd = strategies.StandardDeviation()
        test_cpu_sd = 2.73
        test_ram_sd = 9.33
        self.assertEqual(
            round(sd.get_sd(self.hosts_load_assert, 'cpu_util'), 2),
            test_cpu_sd)
        self.assertEqual(
            round(sd.get_sd(self.hosts_load_assert, 'memory.resident'), 2),
            test_ram_sd)

    def test_standard_deviation_calculate_migration_case(self):
        sd = strategies.StandardDeviation()
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        self.assertEqual(sd.calculate_migration_case(
            self.hosts_load_assert, "VM_5", "Node_2", "Node_1")[2]["Node_1"],
            {'cpu_util': 10.0, 'memory.resident': 21, 'vcpus': 2})

    def test_standard_deviation_simulate_migrations(self):
        sd = strategies.StandardDeviation()
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        self.assertEqual(
            sd.simulate_migrations(self.fake_cluster.generate_scenario_1(),
                                   self.hosts_load_assert)[0]['host'],
            'Node_4')

    def test_standard_deviation_check_threshold(self):
        sd = strategies.StandardDeviation()
        sd.cpu_threshold = 0.001
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().vcpus = 2
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        sd.simulate_migrations = mock.Mock(return_value=True)
        self.assertTrue(
            sd.check_threshold(self.fake_cluster.generate_scenario_1()))

    def test_standard_deviation_execute_one_migration(self):
        sd = strategies.StandardDeviation()
        model = self.fake_cluster.generate_scenario_1()
        sd.cpu_threshold = 0.001
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().vcpus = 2
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        sd.simulate_migrations = mock.Mock(return_value=[{'vm': 'VM_4',
                                                          's_host': 'Node_2',
                                                          'host': 'Node_1'}])
        with mock.patch.object(sd, 'migrate') as mock_migration:
            sd.execute(model)
            mock_migration.assert_called_once_with(model, 'VM_4', 'Node_2',
                                                   'Node_1')

    def test_standard_deviation_execute_multiply_migrations(self):
        sd = strategies.StandardDeviation()
        model = self.fake_cluster.generate_scenario_1()
        sd.cpu_threshold = 0.042
        sd.ram_threshold = 0.0001
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().vcpus = 2
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        sd.simulate_migrations = mock.Mock(return_value=[{'vm': 'VM_4',
                                                          's_host': 'Node_2',
                                                          'host': 'Node_1'},
                                                         {'vm': 'VM_3',
                                                          's_host': 'Node_2',
                                                          'host': 'Node_3'}])
        with mock.patch.object(sd, 'migrate') as mock_migrate:
            sd.execute(model)
            self.assertEqual(mock_migrate.call_count, 2)

    def test_standard_deviation_execute_nothing_to_migrate(self):
        sd = strategies.StandardDeviation()
        model = self.fake_cluster.generate_scenario_1()
        sd.cpu_threshold = 0.042
        sd.ram_threshold = 0.0001
        sd.ceilometer = mock.MagicMock(
            statistic_aggregation=self.fake_metrics.mock_get_statistics)
        sd.nova.servers.get = mock.MagicMock()
        sd.nova.flavors.get = mock.MagicMock()
        sd.nova.flavors.get().vcpus = 1
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().vcpus = 2
        sd.nova.hypervisors.search = mock.MagicMock()
        sd.nova.hypervisors.get = mock.MagicMock()
        sd.nova.hypervisors.get().memory_mb = 2000
        sd.simulate_migrations = mock.Mock(return_value=False)
        with mock.patch.object(sd, 'migrate') as mock_migrate:
            sd.execute(model)
            mock_migrate.assert_not_called()
