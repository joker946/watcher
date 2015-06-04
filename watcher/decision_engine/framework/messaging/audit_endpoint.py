# -*- encoding: utf-8 -*-
# Copyright (c) 2015 b<>com
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

from watcher.decision_engine.framework.command.trigger_audit_command import \
    TriggerAuditCommand
from watcher.decision_engine.framework.ressourcedb_collector import RessourceDB
from watcher.decision_engine.framework.statedb_collector import NovaCollector
from watcher.openstack.common import log


LOG = log.getLogger(__name__)


class AuditEndpoint(object):
    def __init__(self, de):
        self.de = de

    def do_trigger_audit(self, context, audit_uuid):
        statedb = NovaCollector()
        ressourcedb = RessourceDB()
        audit = TriggerAuditCommand(self.de, statedb,
                                    ressourcedb)
        audit.execute(audit_uuid, context)

    def trigger_audit(self, context, audit_uuid):
        LOG.debug("Trigger audit %s" % audit_uuid)
        self.de.executor.submit(self.do_trigger_audit,
                                context,
                                audit_uuid)
        return audit_uuid
