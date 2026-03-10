#  Copyright 2024-2026 Bjorn Wehlin
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ctypes.util
import importlib.util
import pathlib

import pytest


def _find_cuda_extension(masspcf_dir: pathlib.Path) -> pathlib.Path | None:
    candidates = list(masspcf_dir.glob("_mpcf_cuda*.pyd")) + list(masspcf_dir.glob("_mpcf_cuda*.so"))
    return candidates[0] if candidates else None


def _module_name_from_path(path: pathlib.Path) -> str:
    # Handle .pyd filenames like _mpcf_cuda13.cp313-win_amd64.pyd
    return path.name.split(".")[0]


def test_cuda_module_loads_with_pip_cudart():
    # If no CUDA extension exists in this wheel, skip.
    spec = importlib.util.find_spec("masspcf")
    assert spec is not None and spec.origin, "masspcf package not importable"
    masspcf_dir = pathlib.Path(spec.origin).parent
    cuda_ext = _find_cuda_extension(masspcf_dir)
    if cuda_ext is None:
        pytest.skip("No CUDA extension module in this wheel")

    # Preload cudart (pip-installed) and import the CUDA extension directly.
    from masspcf._cudart_preload import _preload_cudart

    try:
        _preload_cudart()
        mod_name = _module_name_from_path(cuda_ext)
        cuda_spec = importlib.util.spec_from_file_location(mod_name, cuda_ext)
        assert cuda_spec is not None and cuda_spec.loader is not None
        cuda_mod = importlib.util.module_from_spec(cuda_spec)
        cuda_spec.loader.exec_module(cuda_mod)
    except Exception:
        has_pip_cuda = importlib.util.find_spec("nvidia") is not None
        has_system_cuda = ctypes.util.find_library("cudart") is not None
        if not has_pip_cuda and not has_system_cuda:
            pytest.skip("No cudart found (pip or system); skipping CUDA load test")
        raise

    assert cuda_mod is not None
