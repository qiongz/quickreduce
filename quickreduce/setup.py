import os
import setuptools
import pathlib
import subprocess
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys

py_root = pathlib.Path(__file__).parent.resolve()
project_root = py_root.parent

# Gather
gpu_archs = os.environ.get("GPU_ARCHS", "gfx942")
rocm_arch = [f"--offload-arch={arch}" for arch in gpu_archs.split(";")]
arch_flags = [f"-D__{arch}__" for arch in gpu_archs.split(";")]

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-Wno-unused-function","-pthread"],
    "nvcc": [
        "-O3", "-std=c++17",
        "-Wno-unused-result", "-Wno-undefined-internal",
        "-mllvm", "-amdgpu-early-inline-all=true"
    ] + rocm_arch + arch_flags,
}

sources = [
    str(project_root / "csrc/quickreduce.hip"),
    str(project_root / "quickreduce/csrc/device.cpp"),
    str(project_root / "quickreduce/csrc/device_pybind.cpp"),
]

include = [
    str(project_root / "csrc"),
    str(project_root / "quickreduce/csrc"),
]

setuptools.setup(
    name='mk1-quickreduce',
    version='0.1.0',
    url="https://www.mk1.ai",
    description="MK1 QuickReduce: Inference-optimized allreduce for ROCm",
    packages=setuptools.find_packages(),
    ext_modules=[
        CUDAExtension(
            name="quickreduce.device",
            sources=sources,
            include_dirs=include,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
