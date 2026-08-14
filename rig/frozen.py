"""Import alias required by the byte-frozen FineWeb builder.

``rig/fineweb_builder.py`` must stay byte-identical forever. Its SHA-256 is
recorded as ``builder_module_sha256`` inside every published manifest under
``data/manifests/fineweb-scaled-gpt2/`` and is checked against
``rig.data_routing.SCALED_BUILDER_SHA256`` on every non-smoke run, so the file
is provenance for roughly 25 GB of already-published corpus. Editing it — even
to update an import — would invalidate that chain and make the scaled datasets
fail to load.

The frozen source still spells its dependency ``speedrun.data``, the module's
name before the package was renamed to ``rig``. Importing this module registers
that name so the frozen builder resolves it.

Import this before ``rig.fineweb_builder``::

    import rig.frozen  # noqa: F401
    from rig.fineweb_builder import BuildConfig
"""

from __future__ import annotations

import sys

from . import data as _data


sys.modules.setdefault("speedrun", sys.modules[__package__])
sys.modules.setdefault("speedrun.data", _data)


__all__: list[str] = []
