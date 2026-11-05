"""Regression: the GPU training stack must import on a compute node.

Qwen3.5's modeling code imports fla.modules, fla imports triton, and triton's backend drivers import
triton.runtime.build, which does `import setuptools`. A uv-made venv has no setuptools, so the Qwen smoke
arm crashed at model load (job 11632177), while CPU tests never touched fla or triton. This runs on CPU
(no GPU needed) and is skipped where fla isn't installed, e.g. the laptop.
"""
import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="flash-linear-attention not installed")
def test_setuptools_triton_and_fla_modules_import():
    import setuptools  # noqa: F401  triton.runtime.build imports it at triton import time
    import triton  # noqa: F401
    from fla.modules import FusedRMSNormGated  # noqa: F401  exactly what modeling_qwen3_5 imports
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
