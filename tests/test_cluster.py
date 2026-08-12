from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.cluster import (
    SSH_SETUP_GUIDANCE,
    ClusterAccessError,
    build_distributed_launch_command,
    expand_host_expression,
    infer_host_expression,
    probe_cluster,
)
from harness.cluster import _copy_to_hosts
from speedrun.config import ConfigError, LocalConfig, load_config, save_config


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
            patch("harness.cluster.shutil.which", return_value="/usr/bin/pdsh"),
            patch("harness.cluster.subprocess.run", return_value=completed) as run,
        ):
            hosts = expand_host_expression("slice-w-[0-2]")
        self.assertEqual(hosts, ("slice-w-0", "slice-w-2", "slice-w-1"))
        self.assertEqual(run.call_args.args[0][-1], "/bin/echo")

    def test_probe_resolves_controller_and_peer_hosts(self) -> None:
        hosts = tuple(f"slice-w-{index}" for index in range(4))
        output = "".join(f"{host}: {host}\n" for host in hosts)
        completed = subprocess.CompletedProcess([], 0, output, "")
        with (
            patch("harness.cluster.expand_host_expression", return_value=hosts),
            patch("harness.cluster.subprocess.run", return_value=completed),
            patch("harness.cluster.socket.gethostname", return_value="slice-w-0"),
        ):
            inventory = probe_cluster("slice-w-[0-3]", 4)

        self.assertEqual(inventory.local_host, "slice-w-0")
        self.assertEqual(inventory.remote_hosts, hosts[1:])
        self.assertEqual(inventory.reported_hostnames["slice-w-3"], "slice-w-3")

    def test_probe_reports_short_ssh_key_guidance(self) -> None:
        hosts = ("slice-w-0", "slice-w-1")
        completed = subprocess.CompletedProcess([], 1, "", "permission denied")
        with (
            patch("harness.cluster.expand_host_expression", return_value=hosts),
            patch("harness.cluster.subprocess.run", return_value=completed),
        ):
            with self.assertRaisesRegex(ClusterAccessError, "authorized_keys") as raised:
                probe_cluster("slice-w-[0-1]", 2)
        self.assertEqual(str(raised.exception), SSH_SETUP_GUIDANCE)

    def test_distributed_command_is_unlabelled_and_shell_quoted(self) -> None:
        with patch("harness.cluster.shutil.which", return_value="/usr/bin/pdsh"):
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

    def test_workspace_copy_uses_parallel_scp_without_peer_pdcp(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch("harness.cluster.shutil.which", return_value="/usr/bin/scp"),
            patch("harness.cluster.subprocess.run", return_value=completed) as run,
        ):
            _copy_to_hosts(
                Path("/tmp/archive.tar.gz"),
                "/tmp/remote.tar.gz",
                ("slice-w-1", "slice-w-2"),
                environment={},
            )
        self.assertEqual(run.call_count, 2)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(all(command[0] == "scp" for command in commands))
        self.assertEqual(
            {command[-1] for command in commands},
            {"slice-w-1:/tmp/remote.tar.gz", "slice-w-2:/tmp/remote.tar.gz"},
        )


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

            (root / ".speedrun.toml").write_text(
                "[speedrun]\ndata_path = \"shm\"\nartifacts_path = \"runs\"\n",
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
