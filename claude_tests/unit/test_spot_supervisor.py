"""Surviving spot interruptions: resuming an instance, restarting training, and spot-only --resume."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from jarvislabs.exceptions import ValidationError

from hale_llm.cli.cloud import _train_args
from hale_llm.cloud.jarvislabs.errors import CloudError
from hale_llm.cloud.jarvislabs.instance_manager import InstanceManager
from hale_llm.cloud.jarvislabs.settings import JarvisSettings
from hale_llm.cloud.jarvislabs.spot_supervisor import SpotSupervisor

ARGS = ["--config", "c.yaml", "--device", "cuda", "--resume"]


class FakeManager:
    def __init__(self, statuses=("Paused",), resumed_ids=(11, 12, 13)):
        self.statuses, self.resumed_ids = list(statuses), list(resumed_ids)
        self.resumed_from: list[int] = []

    def get(self, machine_id):
        return SimpleNamespace(status=self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0])

    def resume(self, machine_id):
        self.resumed_from.append(machine_id)
        return self.resumed_ids.pop(0)


class FakeJob:
    """``outcomes``: what each wait_until_finished call does (an int = exit code, an Exception = raise it)."""

    def __init__(self, outcomes):
        self.outcomes, self.events = list(outcomes), []

    def wait_until_finished(self, machine_id, poll_seconds):
        self.events.append(("wait", machine_id))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def deploy(self, machine_id):
        self.events.append(("deploy", machine_id))

    def start(self, machine_id, args):
        self.events.append(("start", machine_id, args))


def _supervisor(job, manager, *, spot=True, max_restarts=10):
    settings = JarvisSettings(spot=spot, max_restarts=max_restarts)
    return SpotSupervisor(manager, job, settings, machine_id=5, train_args=ARGS, poll_seconds=0, sleep=lambda s: None)


def test_no_interruption_means_no_recovery():
    job, manager = FakeJob([0]), FakeManager()
    supervisor = _supervisor(job, manager)
    assert supervisor.run() == 0 and supervisor.machine_id == 5 and manager.resumed_from == []


def test_an_interrupted_run_is_resumed_redeployed_and_restarted_on_the_new_instance():
    job = FakeJob([CloudError("instance is paused"), 0])
    manager = FakeManager(statuses=["Paused"], resumed_ids=[11])
    supervisor = _supervisor(job, manager)
    assert supervisor.run() == 0
    assert manager.resumed_from == [5]
    assert supervisor.machine_id == 11  # resuming gave a new id, and polling continued on it
    assert job.events == [("wait", 5), ("deploy", 11), ("start", 11, ARGS), ("wait", 11)]
    assert supervisor.restarts == 1


def test_it_survives_several_interruptions():
    job = FakeJob([CloudError("x"), CloudError("x"), 0])
    manager = FakeManager(statuses=["Pausing"], resumed_ids=[11, 12])
    supervisor = _supervisor(job, manager)
    assert supervisor.run() == 0 and supervisor.machine_id == 12 and supervisor.restarts == 2


def test_it_gives_up_after_max_restarts():
    job = FakeJob([CloudError("x")] * 5)
    with pytest.raises(CloudError, match="giving up"):
        _supervisor(job, FakeManager(), max_restarts=2).run()


def test_an_on_demand_instance_that_was_paused_is_not_restarted():
    job = FakeJob([CloudError("x")])
    manager = FakeManager()
    with pytest.raises(CloudError, match="not a spot"):
        _supervisor(job, manager, spot=False).run()
    assert manager.resumed_from == []


def test_a_failed_poll_on_a_running_instance_is_retried_then_succeeds():
    job = FakeJob([CloudError("ssh hiccup"), 0])
    manager = FakeManager(statuses=["Running"])
    supervisor = _supervisor(job, manager)
    assert supervisor.run() == 0 and manager.resumed_from == []  # no recovery: it was only a blip


def test_persistent_poll_failures_on_a_running_instance_are_raised():
    job = FakeJob([CloudError("ssh down")] * 10)
    with pytest.raises(CloudError, match="ssh down"):
        _supervisor(job, FakeManager(statuses=["Running"])).run()


# ----- InstanceManager.resume -----
class FakeInstances:
    def __init__(self, failures):
        self.failures, self.resume_calls = list(failures), []

    def resume(self, machine_id, **kwargs):
        self.resume_calls.append((machine_id, kwargs))
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(machine_id=77)

    def get(self, machine_id):
        return SimpleNamespace(status="Running", ssh_command="ssh host")


def _manager(failures, **settings):
    client = SimpleNamespace(instances=FakeInstances(failures))
    return InstanceManager(JarvisSettings(**settings), client=client), client.instances


def test_resume_waits_for_a_free_gpu_then_returns_the_new_id(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    manager, instances = _manager([ValidationError("No free A30 GPUs in IN2 right now"), ValidationError("still Pausing")], spot=True)
    assert manager.resume(5) == 77
    assert len(instances.resume_calls) == 3
    assert all(call[1] == {"is_spot": True} for call in instances.resume_calls)


def test_resume_as_on_demand_when_the_settings_are_not_spot(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    manager, instances = _manager([], spot=False)
    manager.resume(5)
    assert instances.resume_calls == [(5, {"is_spot": False})]


def test_resume_gives_up_after_the_wait_limit(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    manager, _ = _manager([ValidationError("No free A30 GPUs")] * 1000, resume_wait_minutes=0)
    with pytest.raises(CloudError, match="could not resume"):
        manager.resume(5)


@pytest.mark.parametrize("status,pauses", [("Running", 1), ("Paused", 0), ("Pausing", 0)])
def test_pause_if_running_only_pauses_a_running_instance(status, pauses):
    calls = []
    client = SimpleNamespace(instances=SimpleNamespace(get=lambda i: SimpleNamespace(status=status), pause=calls.append))
    InstanceManager(JarvisSettings(), client=client).pause_if_running(3)
    assert len(calls) == pauses


def test_pause_if_running_never_raises():
    def broken(_):
        raise RuntimeError("API down")

    client = SimpleNamespace(instances=SimpleNamespace(get=broken))
    InstanceManager(JarvisSettings(), client=client).pause_if_running(3)  # logs the problem, does not hide the real error


# ----- spot-only --resume -----
def test_a_spot_launch_adds_resume_once_and_an_on_demand_launch_does_not():
    assert _train_args("c.yaml", [], spot=True) == ["--config", "c.yaml", "--device", "cuda", "--resume"]
    assert "--resume" not in _train_args("c.yaml", [], spot=False)
    assert _train_args("c.yaml", ["--resume"], spot=True).count("--resume") == 1


def test_an_explicit_no_resume_wins_even_on_spot():
    args = _train_args("c.yaml", ["--no-resume"], spot=True)
    assert "--no-resume" in args and "--resume" not in args
