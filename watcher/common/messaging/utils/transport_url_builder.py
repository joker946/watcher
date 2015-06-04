# -*- encoding: utf-8 -*-
# Copyright (c) 2015 b<>com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from oslo_config import cfg
from watcher.openstack.common import log

LOG = log.getLogger(__name__)
CONF = cfg.CONF


class TransportUrlBuilder(object):

    @property
    def url(self):
        return "%s://%s:%s@%s:%s/%s" % (
            CONF.watcher_messaging.protocol,
            CONF.watcher_messaging.user,
            CONF.watcher_messaging.password,
            CONF.watcher_messaging.host,
            CONF.watcher_messaging.port,
            CONF.watcher_messaging.virtual_host
        )
