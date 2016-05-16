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

from apscheduler.schedulers import background
from concurrent import futures
import time

from oslo_config import cfg
from oslo_log import log

from watcher.common import context
from watcher.common.messaging.events import event as watcher_event
from watcher.decision_engine.audit import base
from watcher.decision_engine.messaging import events as de_events
from watcher.decision_engine.planner import manager as planner_manager
from watcher.decision_engine.strategy.context import default as default_context
from watcher.objects import action_plan as action_objects
from watcher.objects import audit as audit_objects


LOG = log.getLogger(__name__)
CONF = cfg.CONF


class DefaultAuditHandler(base.BaseAuditHandler):
    def __init__(self, messaging):
        super(DefaultAuditHandler, self).__init__()
        self._messaging = messaging
        self._strategy_context = default_context.DefaultStrategyContext()
        self._planner_manager = planner_manager.PlannerManager()
        self._planner = None

    @property
    def planner(self):
        if self._planner is None:
            self._planner = self._planner_manager.load()
        return self._planner

    @property
    def messaging(self):
        return self._messaging

    @property
    def strategy_context(self):
        return self._strategy_context

    def notify(self, audit_uuid, event_type, status):
        event = watcher_event.Event()
        event.type = event_type
        event.data = {}
        payload = {'audit_uuid': audit_uuid,
                   'audit_status': status}
        self.messaging.status_topic_handler.publish_event(
            event.type.name, payload)

    def update_audit_state(self, request_context, audit_uuid, state):
        LOG.debug("Update audit state: %s", state)
        audit = audit_objects.Audit.get_by_uuid(request_context, audit_uuid)
        audit.state = state
        audit.save()
        self.notify(audit_uuid, de_events.Events.TRIGGER_AUDIT, state)
        return audit

    def execute(self, audit_uuid, request_context):
        try:
            LOG.debug("Trigger audit %s", audit_uuid)
            # change state of the audit to ONGOING
            audit = self.update_audit_state(request_context, audit_uuid,
                                            audit_objects.State.ONGOING)

            # execute the strategy
            solution = self.strategy_context.execute_strategy(audit_uuid,
                                                              request_context)

            if audit.type == audit_objects.AuditType.CONTINUOUS.value:
                a_plan_filters = {'audit_uuid': audit.uuid,
                                  'state': action_objects.State.RECOMMENDED}
                action_plans = action_objects.ActionPlan.list(
                    request_context,
                    filters=a_plan_filters)
                for plan in action_plans:
                    plan.state = action_objects.State.CANCELLED
                    plan.save()

            self.planner.schedule(request_context, audit.id, solution)

            # change state of the audit to SUCCEEDED
            self.update_audit_state(request_context, audit_uuid,
                                    audit_objects.State.SUCCEEDED)
        except Exception as e:
            LOG.exception(e)
            self.update_audit_state(request_context, audit_uuid,
                                    audit_objects.State.FAILED)


class PeriodicAuditHandler(DefaultAuditHandler):

    def __init__(self, messaging):
        super(PeriodicAuditHandler, self).__init__(messaging)
        self._executor = futures.ThreadPoolExecutor(
            CONF.watcher_decision_engine.max_workers)
        self.audits = []
        self._scheduler = None

    @property
    def executor(self):
        return self._executor

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = background.BackgroundScheduler()
        return self._scheduler

    def execute(self, audit, context):
        time.sleep(audit.period)
        if audit.state == audit_objects.State.CANCELLED:
            return
        super(PeriodicAuditHandler, self).execute(audit.uuid, context)
        self.audits.remove(audit.uuid)
        return audit.uuid

    def launch_audits_periodically(self):
        audit_context = context.RequestContext(is_admin=True)
        audit_filters = {
            'type': audit_objects.AuditType.CONTINUOUS.value,
            'state__in': (audit_objects.State.PENDING,
                          audit_objects.State.ONGOING,
                          audit_objects.State.SUCCEEDED)
        }
        audits = audit_objects.Audit.list(audit_context,
                                          filters=audit_filters)
        for audit in audits:
            if audit.uuid not in self.audits:
                self.audits.append(audit.uuid)
                self.executor.submit(self.execute, audit, audit_context)

    def start(self):
        self.scheduler.add_job(self.launch_audits_periodically, 'interval',
                               seconds=1)
        self.scheduler.start()
