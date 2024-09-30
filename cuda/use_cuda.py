# # Compiling CUDA programs on Modal
#
# In this tutorial, we'll demonstrate how to compile CUDA programs on Modal.
# This is intended as a beginner-friendly but detailed guide to working with CUDA on Modal.
#
# For a quick summary of the CUDA stack on Modal, see the
# [CUDA guide](https://modal.com/docs/guides/cuda).
# in the docs.
#
# As our test problem, we will run the classic
# fast reciprocal square root algorithm from the late 1900s
# first-person shooter game Quake III Arena
# on every single 32 bit floating point number.

# First, we'll do our Modal setup: importing the `modal` client library and
# defining our Modal App.

from pathlib import Path

import modal

app = modal.App("example-cuda-fast-invsqrt")

# ## Compute Capability and Streaming Multiprocessor Architecture

# One notable feature of GPUs is that they are essentially
# only used when CPUs are not able to handle a workload with sufficient performance.
# Just as no one writes multiprocess or multithread code
# for the CPU because it is so fun and easy,
# no one writes code to run on GPUs for the sheer delight.
#
# Because of this, GPU code is inherently performance-sensitive,
# and performance is the enemy of abstraction.
# So, for example, we frequently need to know the precise GPU
# architecture our code will run on at the time we write it --
# let alone compile it.
#
# The most important piece of information about a GPU
# is the _Compute Capability_, also known as the
# _Streaming Multiprocessor Architecture_.
# Streaming Multiprocessors (SMs), are the execution units of GPUs,
# akin to the cores of a CPU -- though they are themselves composed of cores.
# Different SM architectures have distinct capabilities
# and support distinct instruction sets.

GPU_CONFIG = modal.gpu.H100()  # highest CC on Modal
COMPILE_CONFIG = modal.gpu.T4()  # lowest CC on Modal


if isinstance(COMPILE_CONFIG, modal.gpu.T4):
    GPU_SM_ARCH = "75"  # Turing 12nm microarchitecture
elif isinstance(COMPILE_CONFIG, modal.gpu.A100):
    GPU_SM_ARCH = "80"  # Ampere 7nm microarchitecture
elif isinstance(COMPILE_CONFIG, modal.gpu.A10G):
    GPU_SM_ARCH = "86"  # Ampere 8nm microarchitecture
elif isinstance(COMPILE_CONFIG, modal.gpu.L4):
    GPU_SM_ARCH = "89"  # Lovelace 5nm microarchitecture
elif isinstance(COMPILE_CONFIG, modal.gpu.H100):
    GPU_SM_ARCH = "90"  # Hopper 5nm microarchitecture
else:
    raise ValueError(
        f"Not sure how to compile architecture-specific code for {COMPILE_CONFIG}"
    )

# ## Overview: Compiling, executing, and inspecting CUDA binaries on Modal
#
# In this example, we will compile a simple CUDA program,
# inspect its contents, and run it.
#
# This will all happen remotely on Modal,
# triggered by Python code running on your local machine
#
# This local logic is contained in the `local_entrypoint` below.


@app.local_entrypoint()
def main():
    print(f"🔥 showing nvidia-smi output for {COMPILE_CONFIG}")
    nvidia_smi.remote()

    print(f"🔥 compiling our CUDA program for {COMPILE_CONFIG}")
    prog = nvcc.remote()
    Path("invsqrt_demo").write_bytes(prog)

    path = Path("ptx") / f"invsqrt_demo_{GPU_SM_ARCH}.ptx"
    print("🔥 extracting kernel PTX and SASS from binary")
    print(ptx := cuobjdump.remote(prog))
    print(f"🔥 saving kernel code to {path}")
    path.write_text(ptx)

    print(f"🔥 running our CUDA program on the base Modal image on {GPU_CONFIG}")
    raw_cuda.remote(Path("invsqrt_demo").read_bytes())


# ## Using the CUDA Drivers, CUDA Driver API (`libcuda.so`), and Device Management (`nvidia-smi`)
#
# All Modal Functions run inside containers,
# which provide a sort of light-weight virtual machine
# for code to execute in.
#
# All Modal containers with a GPU attached have the NVIDIA CUDA drivers,
# CUDA Driver API, and device management utilities installed.
#
# This happens _outside_ the world of a container.
# This is actually very common for drivers:
# a container can't emulate the existence of a printer or a network,
# for example, unless the host has one.
#
# So we can interact with the GPU without installing anything else
# in our container image.

base_image = modal.Image.debian_slim(python_version="3.11")


@app.function(gpu=COMPILE_CONFIG, image=base_image)
def nvidia_smi():
    import subprocess

    # nvidia-smi runs, see the output in your terminal
    assert not subprocess.run(["nvidia-smi"]).returncode

    # let's check the output programmatically
    output = subprocess.run(
        ["nvidia-smi", "-q", "-u", "-x"], capture_output=True
    ).stdout

    driver_version, cuda_version = parse_nvidia_smi(output)
    # driver version and CUDA (driver API) version are set by the host
    # not by the container itself!
    assert driver_version.text.split(".")[0] == "550"
    assert cuda_version.text.split(".") == ["12", "4"]


# ## Compiling a CUDA program with `nvcc`

# Even with nothing but the drivers and their API,
# we can still run CUDA programs distributed as binaries.
#
# The function below does this in a silly, non-standard way,
# just to get the point across:
# we send the compiled CUDA program as a bunch of bytes, then run it.


@app.function(gpu=GPU_CONFIG, image=base_image)
def raw_cuda(prog: bytes):
    import subprocess

    filepath = Path("./prog")

    # write the program to a file
    filepath.write_bytes(prog)
    # make the program executable
    filepath.chmod(0o755)
    # run it
    subprocess.run(["./prog"])


# ### Installing `nvcc` and the CUDA Toolkit

# But to use that, we'll need a compiled CUDA program to run.
# So let's install the NVIDIA CUDA compiler (`nvcc`).
# We'll do this on a new image, to underscore that we don't need to install
# these dependencies just to run CUDA programs.

arch = (
    "x86_64"  # instruction set architecture for the CPU, all Modal machines are x86_64
)
distro = "debian11"  # the distribution and version number of our OS (GNU/Linux)
filename = "cuda-keyring_1.1-1_all.deb"  # NVIDIA signing key file
cuda_keyring_url = f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/{arch}/{filename}"

major, minor = 12, 4
max_cuda_version = f"{major}-{minor}"


cudatoolkit_image = (
    base_image.apt_install("wget")
    .run_commands(
        [  # we need to get hold of NVIDIA's CUDA keyring to verify the installation
            f"wget {cuda_keyring_url}",
            f"dpkg -i {filename}",
        ]  # otherwise we can't be sure the binaries are from NVIDIA
    )
    .apt_install(  # MUST BE <= 12.4!
        f"cuda-compiler-{max_cuda_version}",  # nvcc and dependencies
    )
    .env({"PATH": "/usr/local/cuda/bin:$PATH"})
)

# ### Inspecting CUDA binaries with `cuobjdump` and `nvdisasm`
#
# When working at this level, a compiler alone is insufficient.
# So the `cuda-compiler` package we installed includes
# a number of utilities for working with CUDA binaries.
#
# CUDA binaries on Linux are in a format similar to the ELF format used by Linux executables.
# But in addition to regular machine code for execution on the host,
# CUDA binaries contain code for execution on the GPU,
# which may appear in one of two formats: PTX and SASS.
#
# PTX, short for "Parallel Thread eXecution", is a sort of intermediate representation for CUDA code.
# It is a virtual instruction set that can be compiled to the actual instruction set of the GPU.
# At time of writing, certain advanced features of the Hopper microarchitecture are only accessible
# via writing inline PTX in CUDA code.
#
# SASS, short for "Streaming ASSembly", is the actual assembly code for the GPU --
# it is a textual representation that corresponds directly to the (undocumented) machine codes
# that the GPU executes. As such, it is specific to the targeted GPU microarchitecture/compute capability.
#
# PTX is forward-compatible: if you compile a CUDA program for a GPU with lower compute capability,
# the PTX will run on a GPU with higher compute capability. This is achieved by just-in-time (JIT) compilation:
# the PTX is compiled to SASS at runtime by the CUDA drivers.
#
# We can examine PTX with the `cuobjdump` tool that was installed along with the compiler.
# But we need another piece of the CUDA Toolkit to view both PTX and SASS:
# `nvdisasm`.


@app.function(
    gpu=COMPILE_CONFIG,
    image=cudatoolkit_image.apt_install(  # add CUDA disassembler
        f"cuda-nvdisasm-{max_cuda_version}"
    ),
)
def cuobjdump(prog: bytes) -> str:
    """Extracts the GPU assembly code using cuobjdump, a binary-munging tool from the CUDA Toolkit."""
    import subprocess

    filepath = Path("./prog")
    filepath.write_bytes(prog)

    result = subprocess.run(
        ["cuobjdump", "-ptx", "-sass", "./prog"], capture_output=True, text=True
    )

    if result.returncode:
        raise Exception(result.stderr)

    return result.stdout


# ### Compiling a CUDA program with `nvcc`
#
# Now we can use `nvcc` to compile a CUDA program.
#
# We've included a simple CUDA program with this example.
# It's a translation of the famous fast inverse square root algorithm
# from Quake III Arena into CUDA C. This is only intended as a fun illustrative example;
# contemporary GPUs include direct instruction-level support for inverse square roots.
#
# The CUDA "kernel" for this example, which contains all the code that execute on the GPU,
# is in the `invsqrt_kernel.cu`.
#
# To make the example more interesting, we include a CUDA program to run it at "GPU scale":
# we calculate the result of the algorithm on every single 32-bit floating point number.
# This logic is in `every_invsqrt.cu`.
#
# We add the source code to our Modal environment by `mount`ing the local files,
# adding them to the filesystem of the container running the function.
# We return the resulting binary as a sequence of bytes.
#
# Like most compilers, `nvcc` has lots of arguments.
# Review the comments next to the arguments for explanations.


@app.function(
    image=cudatoolkit_image,
    mounts=[
        modal.Mount.from_local_file("invsqrt_kernel.cu", "/root/invsqrt_kernel.cu"),
        modal.Mount.from_local_file("every_invsqrt.cu", "/root/every_invsqrt.cu"),
    ],
)
def nvcc():
    import subprocess
    from pathlib import Path

    # settings below are designed for easily-understood PTX, not performance!
    assert not subprocess.run(
        [
            "nvcc",  # run nvidia cuda compiler
            f"-arch=compute_{GPU_SM_ARCH}",  # generate PTX machine code compatible with the GPU architecture and future architectures
            f"-code=sm_{GPU_SM_ARCH},compute_{GPU_SM_ARCH}",  # and a SASS binary optimized for the GPU architecture
            "-g",  # include debug symbols
            "-O0",  # no optimization from nvcc, keep PTX assembly simple
            "-Xcompiler",  # and for the gcc toolchain
            "-Og",  # also limit the optimization
            "-lineinfo",  # add line numbers in machine code
            "-v",  # show verbose log output
            "-o",  # and send binary output to
            "invsqrt_demo",  # a binary called invsqrt_demo
            "invsqrt_kernel.cu",  # compiling the kernel
            "every_invsqrt.cu",  # and the host code
            "-lcuda",  # and linking in some symbols from the CUDA driver API
            # note that cudart is linked by default
        ],
        check=True,
    ).returncode

    return Path("./invsqrt_demo").read_bytes()


# ## Running the example
#
# That's all the code for this example.
# You can run it by executing the command
#
# ```bash
# modal run use_cuda.py
# ```
#
# The rest of the code in this example is utility code.


def parse_nvidia_smi(xml_output):
    """Utility for parsing version info from nvidia-smi's XML output"""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_output)

    return root.find("driver_version"), root.find("cuda_version")
