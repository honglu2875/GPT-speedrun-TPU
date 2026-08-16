"""Corpora addressed by name, independently of saved profile settings."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rig.cli import _require_prepared_dataset, _route_for_config
from rig.config import ConfigError, LocalConfig
from rig.data_routing import DataError, dataset_names, named_preparation_route


class NamedRoutingTests(unittest.TestCase):
    def test_every_corpus_is_addressable_by_name(self) -> None:
        self.assertEqual(dataset_names(), ("2B", "4B", "8B", "hero"))
        for name in dataset_names():
            self.assertEqual(named_preparation_route(name).variant.name, name)

    def test_a_partial_download_is_requested_honestly(self) -> None:
        # 500M at 20 TPP needs 10.05B tokens: 101 of hero's 749 shards, not all
        # 749. Capacity routing could not express that.
        route = named_preparation_route("hero", train_shards=105)
        self.assertEqual(route.train_shards, 105)
        self.assertEqual(route.train_capacity, 10_500_000_000)

    def test_asking_for_more_shards_than_exist_is_refused(self) -> None:
        with self.assertRaisesRegex(DataError, "749 train shards"):
            named_preparation_route("hero", train_shards=1000)

    def test_an_unknown_name_lists_the_real_ones(self) -> None:
        with self.assertRaisesRegex(DataError, "hero"):
            named_preparation_route("enormous")


class ResolutionTests(unittest.TestCase):
    def _config(self, **kwargs) -> LocalConfig:
        base = dict(data_profile="official", training_tokens=5_000_000_000)
        base.update(kwargs)
        return LocalConfig(**base)

    def test_a_named_dataset_wins_over_capacity_routing(self) -> None:
        # training_tokens would select 8B; the name must decide instead.
        route = _route_for_config(self._config(dataset="hero"), "official")
        self.assertEqual(route.variant.name, "hero")

    def test_capacity_routing_still_works_when_unnamed(self) -> None:
        route = _route_for_config(self._config(), "official")
        self.assertEqual(route.variant.name, "8B")

    def test_non_official_profiles_ignore_the_name(self) -> None:
        # dev and smoke have their own fixed corpora.
        route = _route_for_config(self._config(dataset="hero"), "dev")
        self.assertIsNone(route.variant)


class PresenceGuardTests(unittest.TestCase):
    def test_a_missing_named_corpus_names_the_fixing_commands(self) -> None:
        config = LocalConfig(dataset="hero", data_profile="official")
        route = _route_for_config(config, "official")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConfigError) as caught:
                _require_prepared_dataset(
                    config, route, Path(directory), cluster="v6e-8"
                )
        message = str(caught.exception)
        self.assertIn("rig dataset prepare hero", message)
        self.assertIn("rig dataset ship hero --cluster v6e-8", message)

    def test_a_present_corpus_passes(self) -> None:
        config = LocalConfig(dataset="hero", data_profile="official")
        route = _route_for_config(config, "official")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / route.cache_subdirectory
            target.mkdir(parents=True)
            (target / "fineweb_train_000001.bin").write_bytes(b"x")
            _require_prepared_dataset(config, route, root, cluster=None)

    def test_unnamed_configurations_are_not_guarded(self) -> None:
        # Existing clusters keep capacity routing and must not start failing.
        config = LocalConfig(data_profile="official", training_tokens=5_000_000_000)
        route = _route_for_config(config, "official")
        with tempfile.TemporaryDirectory() as directory:
            _require_prepared_dataset(config, route, Path(directory), cluster=None)


if __name__ == "__main__":
    unittest.main()
