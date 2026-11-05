"""Regression: the GPU training stack must import on a compute node.

Qwen3.5's modeling code imports fla.modules, fla imports triton, and triton's backend drivers import
triton.runtime.build, which does `import setuptools`. A uv-made venv has no setuptools, so the Qwen smoke
arm crashed at model load (job 11632177), while CPU tests never touched fla or triton.

- The CPU test imports setuptools and every triton backend driver (the exact failing chain) and runs in
  the CPU build job's pytest. It fails without setuptools.
- fla.modules itself needs an active GPU driver ("0 active drivers" on CPU nodes), so that import is
  checked only where CUDA is available, i.e. in a GPU job. Both are skipped where triton/fla are absent
  (the laptop).
"""
import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton not installed")
def test_setuptools_and_triton_backend_drivers_import():
    import setuptools  # noqa: F401
    import triton.runtime.build  # noqa: F401  the module that does `import setuptools`
    from triton.backends import backends  # imports each backend's compiler and driver module

    assert "nvidia" in backends
    assert all(b.driver is not None for b in backends.values())


@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="flash-linear-attention not installed")
def test_fla_modules_import_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("fla's kernels need an active GPU driver; checked in the GPU smoke job")
    from fla.modules import FusedRMSNormGated  # noqa: F401  exactly what modeling_qwen3_5 imports
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
