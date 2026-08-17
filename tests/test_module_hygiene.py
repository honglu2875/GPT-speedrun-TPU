from __future__ import annotations

import ast
import builtins
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
# Dunders the interpreter supplies at module scope.
_MODULE_GLOBALS = {"__file__", "__name__", "__doc__", "__package__", "__spec__"}


def _python_sources() -> list[Path]:
    sources = sorted(ROOT.joinpath("rig").rglob("*.py"))
    sources += sorted(ROOT.joinpath("recipes").rglob("train.py"))
    return [path for path in sources if "__pycache__" not in path.parts]


def _undefined_names(source: str) -> list[str]:
    """Names loaded but never bound anywhere in the module.

    Deliberately coarse: it ignores scoping and so cannot find a name that is
    bound in the wrong place. What it does find is a name that is bound
    *nowhere*, which is what a missing import looks like -- and a missing
    import inside a branch that only a multi-host run reaches will not show up
    in a CPU test suite any other way.
    """

    tree = ast.parse(source)
    bound = set(dir(builtins)) | _MODULE_GLOBALS
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound |= set(node.names)
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return sorted(loaded - bound)


class ModuleHygieneTests(unittest.TestCase):
    def test_no_module_references_a_name_it_never_binds(self) -> None:
        """Guards the failure mode that moving code between modules creates.

        Three real bugs of exactly this shape survived a green 335-test suite
        when the shared infrastructure moved out of train.py: `multihost_utils`,
        `PartitionSpec`, and `argparse` were used on paths that only execute
        with more than one process. The suite pins JAX to CPU and runs one
        process, so nothing evaluated those lines.
        """

        offenders = {}
        for path in _python_sources():
            missing = _undefined_names(path.read_text(encoding="utf-8"))
            if missing:
                offenders[str(path.relative_to(ROOT))] = missing
        self.assertEqual(
            offenders,
            {},
            "these modules load names they never bind, which is what a missing "
            f"import looks like: {offenders}",
        )

    def test_the_check_would_catch_a_missing_import(self) -> None:
        # A guard nobody has seen fail is a guard nobody should trust.
        self.assertEqual(
            _undefined_names("def f(mesh):\n    return multihost_utils.sync(mesh)\n"),
            ["multihost_utils"],
        )
        self.assertEqual(
            _undefined_names(
                "from jax.experimental import multihost_utils\n"
                "def f(mesh):\n    return multihost_utils.sync(mesh)\n"
            ),
            [],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
