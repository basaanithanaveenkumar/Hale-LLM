"""``hale-llm-cloud``: launch and manage training runs on Jarvislabs."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from hale_llm.cloud.jarvislabs.errors import CloudError
from hale_llm.cloud.jarvislabs.instance_manager import InstanceManager
from hale_llm.cloud.jarvislabs.jl_cli import JlCli
from hale_llm.cloud.jarvislabs.remote_training_job import RemoteTrainingJob
from hale_llm.cloud.jarvislabs.settings import JarvisSettings

DEFAULT_SETTINGS = "configs/cloud/jarvislabs.yaml"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hale-llm-cloud", description="Train on Jarvislabs GPUs.")
    p.add_argument("--cloud", default=DEFAULT_SETTINGS, help="Instance settings YAML")
    sub = p.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="rent an instance, ship the code, start training")
    launch.add_argument("--config", required=True, help="training YAML, e.g. configs/autoregressive_moe/nano.yaml")
    launch.add_argument("--auto-pause", action="store_true",
                        help="stay connected; when training ends, download results and pause the instance")
    launch.add_argument("--machine-id", type=int, default=None, help="reuse this instance instead of renting one")

    finish = sub.add_parser("finish", help="wait for training to end, download results, then pause the instance")
    finish.add_argument("machine_id", type=int)
    finish.add_argument("--poll-seconds", type=float, default=60.0)

    for name, text in [("status", "instance state and whether training is running"),
                       ("logs", "tail the training log"),
                       ("fetch", "download checkpoints and TensorBoard logs"),
                       ("pause", "stop compute billing, keep the disk"),
                       ("destroy", "delete the instance and its disk")]:
        s = sub.add_parser(name, help=text)
        s.add_argument("machine_id", type=int)
        if name == "logs":
            s.add_argument("--tail", type=int, default=40)
        if name == "fetch":
            s.add_argument("--pause", action="store_true", help="pause the instance after downloading")
        if name == "destroy":
            s.add_argument("--yes", action="store_true", help="confirm the deletion")
    return p


def _train_args(config: str, extra: list[str]) -> list[str]:
    """Arguments for ``hale-llm-train`` on the instance: the config, the user's flags, and CUDA.

    The instance is a CUDA box but the shared configs say ``device: mps`` (for the Mac); without
    ``--device cuda`` training would quietly fall back to the instance's CPU.
    """
    args = ["--config", config, *extra]
    return args if "--device" in extra else [*args, "--device", "cuda"]


def _finish(manager: InstanceManager, job: RemoteTrainingJob, machine_id: int, poll_seconds: float) -> None:
    """Wait for training to end, download the results, and always pause so billing stops."""
    try:
        code = job.wait_until_finished(machine_id, poll_seconds)
        if code != 0:
            # Show why it failed instead of trying to download results that were never written.
            print(f"training FAILED with exit code {code}. Last log lines:\n{job.logs(machine_id, 40)}")
            raise CloudError(f"training failed (exit code {code}); see the log lines above")
        print("training finished with exit code 0")
        where = job.fetch(machine_id, Path("data/cloud_runs") / str(machine_id))
        print(f"downloaded to {where}" if where else "no results folder to download (was this a dry run?)")
    finally:  # pausing keeps the disk, so it is safe even if training or the download failed
        manager.pause(machine_id)


def main() -> None:
    # Flags this parser does not know (e.g. --steps 2000) are passed on to hale-llm-train.
    args, extra = _build_parser().parse_known_args()
    if args.command != "launch" and extra:
        raise SystemExit(f"unrecognized arguments: {' '.join(extra)}")
    try:
        settings = JarvisSettings.from_yaml(args.cloud)
        manager = InstanceManager(settings)
        job = RemoteTrainingJob(settings, JlCli(), Path.cwd())

        if args.command == "launch":
            # Fail before renting anything (rented GPUs bill by the hour).
            if not Path(args.config).is_file():
                raise CloudError(f"training config not found: {args.config}")
            manager.ensure_ssh_key()  # before renting: no key means no SSH, and the instance would bill for nothing
            machine_id = args.machine_id or manager.create(Path(args.config).stem).machine_id
            try:
                manager.wait_until_ssh_ready(machine_id)
                job.deploy(machine_id)
                job.start(machine_id, _train_args(args.config, extra))
            except CloudError:
                # The instance exists and bills by the hour; say how to reuse or stop it.
                logger.error(
                    "launch failed, but instance {0} is still running and billing. Retry on it with "
                    "`--machine-id {0}`, or stop it with `hale-llm-cloud pause {0}` / `destroy {0} --yes`.",
                    machine_id,
                )
                raise
            print(f"machine_id={machine_id}\nnext: hale-llm-cloud logs {machine_id}")
            if args.auto_pause:
                _finish(manager, job, machine_id, 60.0)
        elif args.command == "finish":
            _finish(manager, job, args.machine_id, args.poll_seconds)
        elif args.command == "status":
            inst = manager.get(args.machine_id)
            running = job.is_running(args.machine_id) if inst.status == "Running" else False
            print(f"{inst.name}: {inst.status}, {inst.gpu_type} x{inst.num_gpus}, cost {inst.cost}, training running: {running}")
        elif args.command == "logs":
            print(job.logs(args.machine_id, args.tail))
        elif args.command == "fetch":
            where = job.fetch(args.machine_id, Path("data/cloud_runs") / str(args.machine_id))
            print(f"downloaded to {where}")
            if args.pause:
                manager.pause(args.machine_id)
        elif args.command == "pause":
            manager.pause(args.machine_id)
        elif args.command == "destroy":
            if not args.yes:
                raise CloudError("destroy deletes the disk permanently; repeat with --yes")
            manager.destroy(args.machine_id)
    except CloudError as err:
        logger.error("{}", err)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
