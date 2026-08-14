"""Pin the whole suite to the CPU backend before anything imports JAX.

These are CPU-only infrastructure tests. Several of them build real JAX
executables, and ``jax.jit`` compiles for the default backend, so without this
they would claim the accelerator instead of exercising the CPU path they were
written for.

On a multi-host slice that is not merely wasteful, it hangs. A v4-32 is one
slice spanning four VMs, not four independent v4-8s, so a lone pytest process
cannot initialize the TPU: libtpu blocks trying to build the slice with peers
that are not running a matching process (``SliceBuilder grpc channel to
<peer>:8471``). The backend initializes on the first *execution*, not at
``jax.jit``, so the suite gets through the static tests and then wedges on the
first one that actually runs a compiled function — for us,
``test_trainer_static.py`` at ``test_diagnostic_executable_preserves_ordinary_
optimizer_trajectory``. On a single-host TPU box the same tests would have run,
just on the wrong device.

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
