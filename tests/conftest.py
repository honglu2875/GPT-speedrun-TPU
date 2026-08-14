"""Pin the whole suite to the CPU backend before anything imports JAX.

These are CPU-only infrastructure tests. Several of them build real JAX
executables, and ``jax.jit`` compiles for the default backend — so on a TPU
host they would seize the accelerator instead of exercising the CPU path they
were written for. ``test_trainer_static`` in particular hangs indefinitely that
way while compiling the diagnostic training step on device.

pytest imports this file before collecting any test module, which is the only
point early enough to set the variable: JAX latches its backend on first use,
and by the time an individual test module runs, another module's import may
already have initialized it.

Individual modules also call ``os.environ.setdefault("JAX_PLATFORMS", "cpu")``;
those remain correct and keep working when a file is run on its own.
"""

from __future__ import annotations

import os


os.environ.setdefault("JAX_PLATFORMS", "cpu")
