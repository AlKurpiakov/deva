# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": ["-O2"]}
    define_macros = []

    # FORCE_CUDA=1 allows building without a visible GPU (e.g. on a login
    # node); the target architectures then come from TORCH_CUDA_ARCH_LIST.
    force_cuda = os.environ.get("FORCE_CUDA", "0") == "1"

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set and nvcc was not found. Run `module load cuda` "
            "(its major version must match torch.version.cuda = %s)."
            % torch.version.cuda
        )
    if torch.version.cuda is None:
        raise RuntimeError(
            "The installed PyTorch is a CPU-only build (torch.version.cuda is None). "
            "Install a CUDA build of PyTorch first, see INSTALL.md."
        )

    if torch.cuda.is_available() or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-O2",
        ]
    else:
        raise NotImplementedError(
            "CUDA is not available: torch.cuda.is_available() is False. Check the "
            "NVIDIA driver, or set FORCE_CUDA=1 together with TORCH_CUDA_ARCH_LIST "
            "to build without a visible GPU."
        )

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    # Only the compiled extension is installed. The Python wrappers
    # (functions/, modules/) are imported from the repository as
    # models.ops.*, so they are not copied into site-packages as generic
    # top-level "functions" and "modules" packages.
    packages=[],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
