from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rig.harness.cluster import (
    RAM_CACHE_ROOT,
    RAM_CACHE_SETUP_GUIDANCE,
    SSH_SETUP_GUIDANCE,
    ClusterAccessError,
    ClusterError,
    ClusterInventory,
    bootstrap_rsync,
    build_distributed_launch_command,
    expand_host_expression,
    infer_host_expression,
    prepare_ram_cache,
    probe_cluster,
    run_pdsh,
    seal_ram_cache_command,
    terminate_distributed_workers,
)
from rig.harness.cluster import _rsync_to_hosts
from rig.config import ConfigError, LocalConfig, load_config, save_config


class ClusterTests(unittest.TestCase):
    def test_cloud_tpu_worker_pattern_is_inferred(self) -> None:
        self.assertEqual(
            infer_host_expression(4, "t1v-n-a09f5679-w-0"),
            "t1v-n-a09f5679-w-[0-3]",
        )
        self.assertEqual(infer_host_expression(1, "anything"), "")
        self.assertEqual(infer_host_expression(4, "controller"), "")

    def test_pdsh_exec_backend_expands_host_expression_without_ssh(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            "slice-w-0: \nslice-w-2: \nslice-w-1: \n",
            "",
        )
        with (
            patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/pdsh"),
            patch("rig.harness.cluster.subprocess.run", return_value=completed) as run,
        ):
            hosts = expand_host_expression("slice-w-[0-2]")
        self.assertEqual(hosts, ("slice-w-0", "slice-w-2", "slice-w-1"))
        self.assertEqual(run.call_args.args[0][-1], "/bin/echo")

    def test_probe_resolves_controller_and_peer_hosts(self) -> None:
        hosts = tuple(f"slice-w-{index}" for index in range(4))
        completed = tuple(
            subprocess.CompletedProcess([], 0, f"{host}\n", "") for host in hosts
        )
        with (
            patch("rig.harness.cluster.expand_host_expression", return_value=hosts),
            patch("rig.harness.cluster.subprocess.run", side_effect=completed) as run,
            patch("rig.harness.cluster.socket.gethostname", return_value="slice-w-0"),
        ):
            inventory = probe_cluster("slice-w-[0-3]", 4)

        self.assertEqual(inventory.local_host, "slice-w-0")
        self.assertEqual(inventory.remote_hosts, hosts[1:])
        self.assertEqual(inventory.reported_hostnames["slice-w-3"], "slice-w-3")
        self.assertEqual(run.call_count, 4)
        self.assertTrue(all(call.args[0][0] == "ssh" for call in run.call_args_list))
        self.assertTrue(
            all(
                any("ControlMaster=auto" in part for part in call.args[0])
                for call in run.call_args_list
            )
        )

    def test_probe_reports_short_ssh_key_guidance(self) -> None:
        hosts = ("slice-w-0", "slice-w-1")
        completed = subprocess.CompletedProcess([], 255, "", "permission denied")
        with (
            patch("rig.harness.cluster.expand_host_expression", return_value=hosts),
            patch("rig.harness.cluster.subprocess.run", return_value=completed),
            patch("rig.harness.cluster.time.sleep"),
        ):
            with self.assertRaisesRegex(ClusterAccessError, "authorized_keys") as raised:
                probe_cluster("slice-w-[0-1]", 2)
        self.assertEqual(str(raised.exception), SSH_SETUP_GUIDANCE)

    def test_probe_retries_transient_ssh_failure(self) -> None:
        hosts = ("slice-w-0", "slice-w-1")
        failed = subprocess.CompletedProcess([], 255, "", "key exchange failed")
        first = subprocess.CompletedProcess([], 0, "slice-w-0\n", "")
        second = subprocess.CompletedProcess([], 0, "slice-w-1\n", "")
        with (
            patch("rig.harness.cluster.expand_host_expression", return_value=hosts),
            patch(
                "rig.harness.cluster.subprocess.run",
                side_effect=(failed, first, second),
            ) as run,
            patch("rig.harness.cluster.socket.gethostname", return_value="slice-w-0"),
            patch("rig.harness.cluster.time.sleep") as sleep,
        ):
            inventory = probe_cluster("slice-w-[0-1]", 2)

        self.assertEqual(inventory.local_host, "slice-w-0")
        self.assertEqual(run.call_count, 3)
        sleep.assert_called_once_with(1.0)

    def test_pdsh_retries_only_transport_status(self) -> None:
        failed = subprocess.CompletedProcess([], 255, "", "key exchange failed")
        succeeded = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/pdsh"),
            patch(
                "rig.harness.cluster.subprocess.run",
                side_effect=(failed, succeeded),
            ) as run,
            patch("rig.harness.cluster.time.sleep") as sleep,
        ):
            run_pdsh(("slice-w-0", "slice-w-1"), "hostname", labels=False)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(1.0)

        remote_failure = subprocess.CompletedProcess([], 5, "", "setup failed")
        with (
            patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/pdsh"),
            patch("rig.harness.cluster.subprocess.run", return_value=remote_failure) as run,
            patch("rig.harness.cluster.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(ClusterError, "setup failed"):
                run_pdsh(("slice-w-0",), "false", labels=False)
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    def test_distributed_command_is_unlabelled_and_shell_quoted(self) -> None:
        with patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/pdsh"):
            command = build_distributed_launch_command(
                host_expression="slice-w-[0-3]",
                host_count=4,
                cwd=Path("/repo with space/submissions/reference"),
                command=("/repo with space/.venv/bin/python", "train.py", "--seed", "7"),
                environment={"SAFE_VALUE": "value with space", "RANK_COUNT": "4"},
            )

        self.assertEqual(command[:8], ["pdsh", "-S", "-R", "ssh", "-f", "4", "-w", "slice-w-[0-3]"])
        self.assertEqual(command[8], "-N")
        self.assertIn("cd '/repo with space/submissions/reference'", command[-1])
        self.assertIn("SAFE_VALUE='value with space'", command[-1])
        self.assertIn("'/repo with space/.venv/bin/python'", command[-1])

    def test_distributed_teardown_matches_one_exact_run(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/pdsh"),
            patch("rig.harness.cluster.subprocess.run", return_value=completed) as run,
        ):
            cleaned = terminate_distributed_workers(
                host_expression="slice-w-[0-3]",
                host_count=4,
                executable=Path("/repo/.venv/bin/python3"),
                script=Path("/repo/submissions/candidate/train.py"),
                output_dir=Path("/repo/runs/run-123"),
            )

        self.assertTrue(cleaned)
        command = run.call_args.args[0]
        self.assertEqual(
            command[:8],
            ["pdsh", "-S", "-R", "ssh", "-f", "4", "-w", "slice-w-[0-3]"],
        )
        self.assertIn("pkill -KILL -f --", command[-1])
        self.assertIn("candidate/train\\.py", command[-1])
        self.assertIn("run\\-123", command[-1])

    def test_workspace_copy_uses_parallel_incremental_rsync_without_delete(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/rsync"),
            patch("rig.harness.cluster.subprocess.run", return_value=completed) as run,
        ):
            _rsync_to_hosts(
                Path("/repo"),
                ("slice-w-1", "slice-w-2"),
                (".git/", "/runs/"),
                environment={},
            )
        self.assertEqual(run.call_count, 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(all(command[0] == "rsync" for command in commands))
        self.assertTrue(all("--delete" not in command for command in commands))
        self.assertTrue(all(".git/" in command and "/runs/" in command for command in commands))
        self.assertEqual(
            {command[-1] for command in commands},
            {"slice-w-1:/repo/", "slice-w-2:/repo/"},
        )

    def test_ram_cache_preflight_checks_mount_then_creates_link_on_all_hosts(self) -> None:
        inventory = ClusterInventory(
            host_expression="slice-w-[0-1]",
            hosts=("slice-w-0", "slice-w-1"),
            remote_hosts=("slice-w-1",),
            local_host="slice-w-0",
            reported_hostnames={"slice-w-0": "slice-w-0", "slice-w-1": "slice-w-1"},
        )
        with patch("rig.harness.cluster.run_pdsh") as run:
            prepare_ram_cache(Path("/repo"), inventory)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], inventory.hosts)
        self.assertIn("tmpfs|ramfs", run.call_args_list[0].args[1])
        setup = run.call_args_list[1].args[1]
        self.assertIn(str(RAM_CACHE_ROOT), setup)
        self.assertIn("sudo -n install", setup)
        self.assertIn(f"ln -s {RAM_CACHE_ROOT} /repo/shm", setup)
        self.assertIn("readlink -f /repo/shm", setup)

    def test_ram_cache_seal_protects_only_the_dedicated_cache(self) -> None:
        command = seal_ram_cache_command()
        self.assertIn(f"chown -R root:", command)
        self.assertIn(str(RAM_CACHE_ROOT), command)
        self.assertIn("sudo -n chown", command)
        self.assertIn("find /dev/shm/.speedrun-cache -type d", command)
        self.assertNotIn("chown -R root:\"$group\" -- /dev/shm ", command)

    def test_ram_cache_preflight_reports_mount_instruction(self) -> None:
        inventory = ClusterInventory(
            host_expression="slice-w-[0-1]",
            hosts=("slice-w-0", "slice-w-1"),
            remote_hosts=("slice-w-1",),
            local_host="slice-w-0",
            reported_hostnames={"slice-w-0": "slice-w-0", "slice-w-1": "slice-w-1"},
        )
        with (
            patch("rig.harness.cluster.run_pdsh", side_effect=ClusterError("mount failed")),
            self.assertRaisesRegex(ClusterError, "mount.*make prepare") as raised,
        ):
            prepare_ram_cache(Path("/repo"), inventory)
        self.assertEqual(str(raised.exception), RAM_CACHE_SETUP_GUIDANCE)

    def test_rsync_bootstrap_attempts_noninteractive_apt_get_on_every_host(self) -> None:
        inventory = ClusterInventory(
            host_expression="slice-w-[0-1]",
            hosts=("slice-w-0", "slice-w-1"),
            remote_hosts=("slice-w-1",),
            local_host="slice-w-0",
            reported_hostnames={"slice-w-0": "slice-w-0", "slice-w-1": "slice-w-1"},
        )
        with (
            patch("rig.harness.cluster.run_pdsh") as run,
            patch("rig.harness.cluster.shutil.which", return_value="/usr/bin/rsync"),
        ):
            bootstrap_rsync(inventory)
        self.assertEqual(run.call_args.args[0], inventory.hosts)
        self.assertIn("apt-get install -y rsync", run.call_args.args[1])
        self.assertIn("DEBIAN_FRONTEND=noninteractive", run.call_args.args[1])


class ClusterConfigTests(unittest.TestCase):
    def test_cluster_settings_round_trip_and_old_files_remain_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = LocalConfig(
                tpu_vm_count=4,
                tpu_vm_hosts="t1v-n-a09f5679-w-[0-3]",
            )
            save_config(configured, root)
            self.assertEqual(load_config(root), configured)

            (root / ".rig.toml").write_text(
                "[rig]\ndata_path = \"shm\"\nartifacts_path = \"runs\"\n",
                encoding="utf-8",
            )
            self.assertEqual(load_config(root).tpu_vm_count, 1)
            self.assertEqual(load_config(root).tpu_vm_hosts, "")

    def test_multiple_hosts_require_a_safe_expression(self) -> None:
        with self.assertRaisesRegex(ConfigError, "required"):
            LocalConfig(tpu_vm_count=4).validate()
        with self.assertRaisesRegex(ConfigError, "whitespace"):
            LocalConfig(tpu_vm_count=4, tpu_vm_hosts="host 1,host2").validate()
        with self.assertRaisesRegex(ConfigError, "positive integer"):
            LocalConfig(tpu_vm_count=True).validate()


if __name__ == "__main__":
    unittest.main()
