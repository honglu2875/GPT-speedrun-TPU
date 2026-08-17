"""Named cluster profiles and the per-cluster accelerator contract."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import unittest.mock

from rig.config import (
    ClusterProfile,
    ConfigError,
    LocalConfig,
    load_clusters,
    load_config,
    save_config,
)
from rig.doctor import check_devices


class _Device:
    def __init__(self, kind: str, platform: str = "tpu") -> None:
        self.device_kind = kind
        self.platform = platform


BOTH = """
[rig]
data_path = "shm"
active_cluster = "v5e-64"

[cluster.v4-32]
tpu_vm_count = 4
tpu_vm_hosts = "t1v-n-aaa-w-[0-3]"
accelerator = "TPU v4"
chips_per_host = 4

[cluster.v5e-64]
tpu_vm_count = 16
tpu_vm_hosts = "t1v-n-bbb-w-[0-15]"
accelerator = "TPU v5 lite"
chips_per_host = 4
"""


class ClusterProfileTests(unittest.TestCase):
    def test_active_cluster_is_overlaid_onto_the_base_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(BOTH, encoding="utf-8")
            config = load_config(root)
            self.assertEqual(config.tpu_vm_count, 16)
            self.assertEqual(config.accelerator, "TPU v5 lite")
            self.assertEqual(config.data_path, "shm")

    def test_explicit_selection_overrides_the_active_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(BOTH, encoding="utf-8")
            config = load_config(root, cluster="v4-32")
            self.assertEqual(config.tpu_vm_count, 4)
            self.assertEqual(config.accelerator, "TPU v4")

    def test_unknown_cluster_names_the_known_ones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(BOTH, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "v4-32"):
                load_config(root, cluster="nope")

    def test_defined_clusters_must_be_selected(self) -> None:
        # Silently falling back to the flat settings would run the wrong slice.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(
                BOTH.replace('active_cluster = "v5e-64"\n', ""), encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigError, "none is active"):
                load_config(root)

    def test_a_file_without_clusters_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(
                '[rig]\ntpu_vm_count = 4\ntpu_vm_hosts = "a-[0-3]"\n', encoding="utf-8"
            )
            config = load_config(root)
            self.assertEqual(config.tpu_vm_count, 4)
            self.assertEqual(config.accelerator, "TPU v4")

    def test_saving_one_cluster_does_not_drop_the_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".rig.toml").write_text(BOTH, encoding="utf-8")
            save_config(load_config(root), root)
            names = sorted(load_clusters(root))
            self.assertEqual(names, ["v4-32", "v5e-64"])
            self.assertEqual(load_config(root, cluster="v4-32").tpu_vm_count, 4)

    def test_cluster_tables_reject_unknown_and_invalid_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ".rig.toml"
            path.write_text("[rig]\n\n[cluster.x]\nwat = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unknown setting"):
                load_clusters(root)
            path.write_text(
                "[rig]\n\n[cluster.x]\ntpu_vm_count = 2\n"
                'tpu_vm_hosts = "a-[0-1]"\nchips_per_host = 0\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "chips_per_host"):
                load_clusters(root)


class AcceleratorContractTests(unittest.TestCase):
    def test_v5e_passes_its_own_contract_and_fails_the_v4_one(self) -> None:
        devices = [_Device("TPU v5 lite") for _ in range(4)]
        with (
            unittest.mock.patch("jax.devices", return_value=devices),
            unittest.mock.patch("jax.process_count", return_value=1),
            unittest.mock.patch("jax.local_device_count", return_value=4),
        ):
            ok = check_devices(
                require_tpu=True, accelerator="TPU v5 lite", chips_per_host=4
            )
            wrong = check_devices(
                require_tpu=True, accelerator="TPU v4", chips_per_host=4
            )
        self.assertEqual(ok.status, "ok")
        self.assertEqual(wrong.status, "error")
        self.assertIn("TPU v4", wrong.hint or "")

    def test_wrong_chip_count_is_rejected(self) -> None:
        devices = [_Device("TPU v5 lite") for _ in range(2)]
        with (
            unittest.mock.patch("jax.devices", return_value=devices),
            unittest.mock.patch("jax.process_count", return_value=1),
            unittest.mock.patch("jax.local_device_count", return_value=2),
        ):
            result = check_devices(
                require_tpu=True, accelerator="TPU v5 lite", chips_per_host=4
            )
        self.assertEqual(result.status, "error")


if __name__ == "__main__":
    unittest.main()
