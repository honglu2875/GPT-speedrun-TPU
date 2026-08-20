"""Orchestrating a slice from a machine that is not part of it.

The failure this guards against is silent: the trainer decides who writes
artifacts by comparing its own hostname against RIG_CONTROLLER_HOSTNAME, so
announcing a machine that is not in the slice makes every worker decline,
and the run trains to completion producing nothing.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rig.config import ConfigError, LocalConfig, load_clusters, load_config, save_config
from rig.harness.cluster import ClusterError, probe_cluster


BOTH = """
[rig]
data_path = "shm"
active_cluster = "v6e-64"

[cluster.v4-32]
tpu_vm_count = 4
tpu_vm_hosts = "local-w-[0-3]"

[cluster.v6e-64]
tpu_vm_count = 16
tpu_vm_hosts = "pod-w-[0-15]"
accelerator = "TPU v6 lite"
remote_controller = true
"""


def _probe(hosts, *, local, remote_controller=False, artifact_host=""):
    """Run probe_cluster against a fake slice, without ssh or pdsh."""

    reported = {host: host for host in hosts}
    with (
        patch("rig.harness.cluster.expand_host_expression", return_value=tuple(hosts)),
        patch("rig.harness.cluster.socket.gethostname", return_value=local),
        patch("rig.harness.cluster.subprocess.run") as run,
    ):
        run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()

        def fake(command, **_kwargs):
            host = command[-2]
            return type(
                "R", (), {"returncode": 0, "stdout": reported[host], "stderr": ""}
            )()

        run.side_effect = fake
        return probe_cluster(
            "expr",
            len(hosts),
            remote_controller=remote_controller,
            artifact_host=artifact_host,
        )


class ConfigTests(unittest.TestCase):
    def test_remote_settings_are_per_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(BOTH, encoding="utf-8")
            remote = load_config(root)
            local = load_config(root, cluster="v4-32")
        self.assertTrue(remote.remote_controller)
        self.assertEqual(remote.accelerator, "TPU v6 lite")
        # The older cluster keeps the ordinary in-slice behaviour.
        self.assertFalse(local.remote_controller)

    def test_booleans_round_trip_as_toml_not_python(self) -> None:
        # bool subclasses int, so an encoder that checks int first writes
        # "True", which tomllib then refuses to read back.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(BOTH, encoding="utf-8")
            save_config(load_config(root), root)
            text = (root / ".rig.toml").read_text(encoding="utf-8")
            self.assertIn("remote_controller = true", text)
            self.assertNotIn("True", text)
            self.assertTrue(load_clusters(root)["v6e-64"].remote_controller)

    def test_remote_controller_needs_a_host_not_a_multi_host_slice(self) -> None:
        # A single remote host is the simplest valid remote setup: this machine
        # holds no accelerator and the work runs on the one host named. What
        # remote mode requires is somewhere to reach, not peers.
        with self.assertRaisesRegex(ConfigError, "tpu_vm_hosts"):
            LocalConfig(remote_controller=True, tpu_vm_count=1).validate()
        single = LocalConfig(
            remote_controller=True,
            tpu_vm_count=1,
            tpu_vm_hosts="10.0.0.1",
            accelerator="TPU v6 lite",
            chips_per_host=8,
        ).validate()
        self.assertEqual(single.tpu_vm_count, 1)

    def test_artifact_host_must_be_a_bare_name(self) -> None:
        with self.assertRaisesRegex(ConfigError, "artifact_host"):
            LocalConfig(
                tpu_vm_count=2, tpu_vm_hosts="a-[0-1]", artifact_host="a 0"
            ).validate()


class ProbeTests(unittest.TestCase):
    HOSTS = ("pod-w-0", "pod-w-1", "pod-w-2")

    def test_remote_controller_owns_no_rank_and_picks_the_first_host(self) -> None:
        inventory = _probe(self.HOSTS, local="orchestrator", remote_controller=True)
        self.assertIsNone(inventory.local_host)
        # Every host is a peer: none of them is this machine.
        self.assertEqual(inventory.remote_hosts, self.HOSTS)
        self.assertEqual(inventory.artifact_host, "pod-w-0")

    def test_explicit_artifact_host_overrides_the_first(self) -> None:
        inventory = _probe(
            self.HOSTS,
            local="orchestrator",
            remote_controller=True,
            artifact_host="pod-w-2",
        )
        self.assertEqual(inventory.artifact_host, "pod-w-2")

    def test_an_artifact_host_outside_the_slice_is_refused(self) -> None:
        with self.assertRaisesRegex(ClusterError, "not one of the configured"):
            _probe(
                self.HOSTS,
                local="orchestrator",
                remote_controller=True,
                artifact_host="pod-w-9",
            )

    def test_remote_controller_may_not_also_be_a_worker(self) -> None:
        # Both in the slice and claiming to be outside it would give the host
        # a JAX rank and an orchestrator role at once.
        with self.assertRaisesRegex(ClusterError, "one of the configured"):
            _probe(self.HOSTS, local="pod-w-1", remote_controller=True)

    def test_ordinary_mode_still_requires_membership(self) -> None:
        with self.assertRaisesRegex(ClusterError, "exactly once"):
            _probe(self.HOSTS, local="orchestrator")

    def test_ordinary_mode_keeps_artifacts_on_this_machine(self) -> None:
        inventory = _probe(self.HOSTS, local="pod-w-1")
        self.assertEqual(inventory.local_host, "pod-w-1")
        self.assertEqual(inventory.artifact_host, "pod-w-1")
        self.assertNotIn("pod-w-1", inventory.remote_hosts)


class ControllerIdentityTests(unittest.TestCase):
    """The env var that decides who writes artifacts."""

    def _environment(self, **overrides):
        from rig.harness.models import RunConfig

        config = RunConfig(
            repo_root=Path("/repo"),
            recipe="reference",
            runs_dir=Path("/repo/runs"),
            records_path=Path("/repo/runs/records.jsonl"),
            plan={},
            tpu_vm_count=4,
            tpu_vm_hosts="pod-w-[0-3]",
            **overrides,
        )
        return config

    def test_remote_run_announces_the_artifact_host_not_this_machine(self) -> None:
        import inspect
        from rig.harness import runner

        source = inspect.getsource(runner.run_recipe)
        # The announcement must prefer the configured artifact host; falling
        # back to gethostname() in remote mode matches no worker at all.
        self.assertIn("config.artifact_hostname or socket.gethostname()", source)

    def test_artifacts_are_pulled_before_the_result_is_read(self) -> None:
        import inspect
        from rig.harness import runner

        source = inspect.getsource(runner.run_recipe)
        pull = source.index("fetch_run_artifacts(")
        read = source.index("parse_result_line(")
        self.assertLess(pull, read, "artifacts must arrive before validation")

    def test_pull_is_skipped_when_this_machine_is_in_the_slice(self) -> None:
        import inspect
        from rig.harness import runner

        source = inspect.getsource(runner.run_recipe)
        guard = source[: source.index("fetch_run_artifacts(")]
        self.assertIn("config.remote_controller and config.artifact_host", guard)


class OpportunisticSalvageTests(unittest.TestCase):
    """Partial artifacts must survive a job that never finishes."""

    def _run(self, script: str, **kwargs):
        # _run_process needs real descriptors: it calls fileno() on the pipes.
        from rig.harness import runner

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            err = Path(directory) / "err"
            with out.open("wb") as stdout, err.open("wb") as stderr:
                return runner._run_process(
                    ["python3", "-c", script],
                    cwd=Path("."),
                    environment={},
                    stdout_handle=stdout,
                    stderr_handle=stderr,
                    timeout_seconds=30.0,
                    **kwargs,
                )

    def test_the_stream_loop_ticks_while_the_child_runs(self) -> None:
        calls: list[int] = []
        code, timed_out = self._run(
            "import time; time.sleep(0.5)",
            tick=lambda: calls.append(1),
            tick_seconds=0.1,
        )
        self.assertEqual(code, 0)
        self.assertFalse(timed_out)
        self.assertGreaterEqual(len(calls), 2)

    def test_a_raising_tick_propagates_so_the_puller_must_swallow(self) -> None:
        # The loop deliberately does not catch tick errors -- that would hide
        # real bugs. Safety therefore lives in the puller, which is why the
        # next test pins that it swallows transfer failures itself.
        def boom() -> None:
            raise RuntimeError("host unreachable")

        with self.assertRaises(RuntimeError):
            self._run("import time; time.sleep(0.3)", tick=boom, tick_seconds=0.05)

    def test_no_puller_when_artifacts_are_already_local(self) -> None:
        from rig.harness import runner
        from rig.harness.models import RunConfig

        local = RunConfig(
            repo_root=Path("/repo"),
            recipe="reference",
            runs_dir=Path("/repo/runs"),
            records_path=Path("/repo/runs/records.jsonl"),
            plan={},
            tpu_vm_count=4,
        )
        self.assertIsNone(runner._artifact_puller(local, Path("/repo/runs/x"), {}))

    def test_puller_swallows_transfer_failures(self) -> None:
        from rig.harness import runner
        from rig.harness.models import RunConfig

        remote = RunConfig(
            repo_root=Path("/repo"),
            recipe="reference",
            runs_dir=Path("/repo/runs"),
            records_path=Path("/repo/runs/records.jsonl"),
            plan={},
            tpu_vm_count=4,
            remote_controller=True,
            artifact_host="pod-w-0",
            artifact_hostname="pod-w-0",
        )
        pull = runner._artifact_puller(remote, Path("/repo/runs/x"), {})
        self.assertIsNotNone(pull)
        with patch(
            "rig.harness.runner.fetch_run_artifacts",
            side_effect=OSError("network down"),
        ):
            pull()  # a lost pull is not a lost run


class SingleRemoteHostTests(unittest.TestCase):
    """One remote host is a valid target, not a degenerate cluster.

    This was rejected in four separate places -- config validation, the CLI
    gates, the launch builder, and orphan cleanup -- each keyed on
    "tpu_vm_count > 1" as a stand-in for "work runs elsewhere". Every one of
    them either refused outright or silently ran the trainer on a machine with
    no accelerator.
    """

    def test_a_remote_launch_may_target_exactly_one_host(self) -> None:
        from rig.harness.cluster import build_distributed_launch_command

        command = build_distributed_launch_command(
            host_expression="10.0.0.1",
            host_count=1,
            cwd=Path("/repo/recipes/reference"),
            command=["python", "train.py"],
            environment={"RIG_OUTPUT_DIR": "/repo/runs/x"},
        )
        self.assertEqual(command[0], "pdsh")
        self.assertIn("10.0.0.1", command)

    def test_a_launch_still_needs_somewhere_to_run(self) -> None:
        from rig.harness.cluster import build_distributed_launch_command

        with self.assertRaisesRegex(ClusterError, "at least one"):
            build_distributed_launch_command(
                host_expression="",
                host_count=0,
                cwd=Path("/repo"),
                command=["python"],
                environment={},
            )

    def test_orphan_cleanup_is_not_skipped_for_one_host(self) -> None:
        # A single remote host can strand processes holding the TPU just as a
        # slice can; skipping cleanup there leaves the chip unusable.
        import inspect
        from rig.harness import cluster

        source = inspect.getsource(cluster.terminate_distributed_workers)
        self.assertIn("if host_count < 1:", source)
        self.assertNotIn("if host_count <= 1:", source)

    def test_single_remote_host_is_launched_remotely(self) -> None:
        import inspect
        from rig.harness import runner

        source = inspect.getsource(runner.run_recipe)
        self.assertIn(
            "launch_remotely = config.tpu_vm_count > 1 or config.remote_controller",
            source,
        )
        # ...but takes no distributed initialization.
        self.assertIn("distributed = config.tpu_vm_count > 1", source)
        self.assertIn("if distributed:", source)


if __name__ == "__main__":
    unittest.main()


class CacheOwnershipTests(unittest.TestCase):
    """The RAM cache must stay writable by the user who prepared it."""

    def test_group_survives_running_as_root(self) -> None:
        # The seal makes the cache root-owned so logout cleanup cannot delete
        # it, and group-writable so the owner keeps access. Resolving the group
        # with a bare `id -g` yields 0 when the fragment itself runs as root,
        # producing root:root and locking the user out of their own cache --
        # observed as "Permission denied" creating a new corpus directory.
        from rig.harness.cluster import (
            prepare_ram_cache,
            seal_ram_cache_command,
            unseal_ram_cache_command,
        )
        import inspect

        for fragment in (seal_ram_cache_command(), unseal_ram_cache_command()):
            self.assertIn("SUDO_GID", fragment)
            self.assertNotIn('group="$(id -g)"', fragment)
        setup = inspect.getsource(prepare_ram_cache)
        self.assertIn("SUDO_GID", setup)
        self.assertNotIn('group="$(id -g)"', setup)

    def test_the_fragments_still_grant_root_ownership(self) -> None:
        # Group is the part that changed; root ownership is the whole point.
        from rig.harness.cluster import seal_ram_cache_command

        self.assertIn("chown -R root:", seal_ram_cache_command())
