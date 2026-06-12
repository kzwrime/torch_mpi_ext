# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import glob
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "torch_mpi_ext"

this_dir = os.path.dirname(os.path.abspath(__file__))
this_path = Path(this_dir)


def find_torch_mcpu_dir():
    torch_mcpu_spec = importlib.util.find_spec("torch_mcpu")
    candidates = []
    if torch_mcpu_spec is not None and torch_mcpu_spec.origin is not None:
        candidates.append(Path(torch_mcpu_spec.origin).parent)
    candidates.append(this_path.parent / "torch_mcpu" / "torch_mcpu")
    return next((path for path in candidates if path.exists()), None)


torch_mcpu_dir = find_torch_mcpu_dir()


def get_torch_mcpu_compile_flags():
    candidate_files = []
    if torch_mcpu_dir is not None:
        candidate_files.append(torch_mcpu_dir / "_compile_flags.json")
    candidate_files.append(
        this_path.parent / "torch_mcpu" / "torch_mcpu" / "_compile_flags.json"
    )
    flags_file = next((path for path in candidate_files if path.exists()), None)
    if flags_file is None:
        return []

    with flags_file.open(encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("compile_flags", []))


def should_generate_aoti_wrappers() -> bool:
    build_commands = {
        "bdist_wheel",
        "build",
        "build_ext",
        "develop",
        "editable_wheel",
        "install",
    }
    return any(command in build_commands for command in sys.argv[1:])


def generate_aoti_wrappers() -> None:
    script = os.path.join(this_dir, "scripts", "generate_aoti_wrappers.py")
    print(f"Generating AOTI wrappers: {script}")
    subprocess.check_call([sys.executable, script], cwd=this_dir)


if should_generate_aoti_wrappers():
    generate_aoti_wrappers()

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    include_dirs = [os.path.join(this_dir, library_name, "include")]
    library_dirs = []
    libraries = []
    runtime_library_dirs = []

    if torch_mcpu_dir is not None:
        torch_mcpu_lib_dir = torch_mcpu_dir / "lib"
        torch_mcpu_lib = torch_mcpu_lib_dir / "libtorch_mcpu.so"
        include_dirs += [
            str(torch_mcpu_dir / "include"),
            str(torch_mcpu_dir),
        ]
        library_dirs.append(str(torch_mcpu_lib_dir))
        libraries.append("torch_mcpu")
        runtime_library_dirs.append(str(torch_mcpu_lib_dir))
        extra_link_args += [str(torch_mcpu_lib), f"-Wl,-rpath,{torch_mcpu_lib_dir}"]

    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
            *get_torch_mcpu_compile_flags(),
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = [
        os.path.relpath(path, this_dir)
        for path in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    ]

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = [
        os.path.relpath(path, this_dir)
        for path in glob.glob(os.path.join(extensions_cuda_dir, "*.cu"))
    ]

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            runtime_library_dirs=runtime_library_dirs,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    package_data={
        library_name: [
            "include/*.h",
            "include/*.hpp",
            "include/*.txt",
        ],
    },
    include_package_data=True,
    ext_modules=get_extensions(),
    install_requires=["torch", "torch_mcpu"],
    description="Example of PyTorch C++ and CUDA extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/extension-cpp",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
