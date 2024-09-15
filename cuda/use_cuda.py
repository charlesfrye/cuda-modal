# # Using CUDA on Modal
#
# CUDA is not one library, but a stack with multiple layers:
#
# - The NVIDIA CUDA Drivers
#   - kernel-level components, like `nvidia.ko`
#   - the user-level CUDA Driver API, `libcuda.so`
# - The NVIDIA CUDA Toolkit
#   - the CUDA Runtime API (`libcudart.so`)
#   - the NVIDIA CUDA compiler (`nvcc`)
#   - and more goodies (`cuobjdump`, profilers, `cudnn`, etc.)
#
# Most folks running code on GPUs don't use these things directly,
# and instead use them via a higher-level library like TensorFlow or PyTorch.
#
# But when configuring environments to run ML applications,
# even using those higher-level libraries, issues with the CUDA stack
# still arise and cause major headaches.
#
# In this tutorial, we'll tour the NVIDIA CUDA stack layer by layer,
# showing how to use it (and break it!) on Modal.
#
# For a quick summary of the CUDA stack on Modal, see the
# [CUDA guide](https://modal.com/docs/guides/cuda).
# in the docs.
#
# As our test problem, we will run the classic
# fast inverse square root algorithm from the late 1900s
# first-person shooter game Quake III Arena
# on every single 32 bit floating point number.

from pathlib import Path

import modal

app = modal.App()

base_image = modal.Image.debian_slim(python_version="3.11")

GPU_CONFIG = modal.gpu.A10G()

if isinstance(GPU_CONFIG, modal.gpu.T4):
    raise Exception(
        "T4 GPUs don't have enough memory to run this example, try an A10G."
    )
elif isinstance(GPU_CONFIG, modal.gpu.A10G):
    GPU_CAPABILITY_CODE = "86"
elif isinstance(GPU_CONFIG, modal.gpu.L4):
    GPU_CAPABILITY_CODE = "89"
elif isinstance(GPU_CONFIG, modal.gpu.A100):
    GPU_CAPABILITY_CODE = "80"
elif isinstance(GPU_CONFIG, modal.gpu.H100):
    GPU_CAPABILITY_CODE = "90"
else:
    raise ValueError(
        f"Not sure how to compile architecture-specific code for {GPU_CONFIG}"
    )

# All Modal containers with a GPU attached have the NVIDIA CUDA drivers
# installed and the CUDA Driver API and management libraries available.
# This happens _outside_ the world of a container.
# Imagine trying to add a printer driver to a container,
# it just doesn't make sense.


@app.function(gpu=GPU_CONFIG, image=base_image)
def nvidia_smi():
    import subprocess

    # nvidia-smi runs, see the output in your terminal
    assert not subprocess.run(["nvidia-smi"]).returncode

    # let's check the output programmatically
    output = subprocess.run(
        ["nvidia-smi", "-q", "-u", "-x"], capture_output=True
    ).stdout

    driver_version, cuda_version = check_nvidia_smi(output)
    # driver version and CUDA (driver API) version are set by the host
    # not by the container itself!
    assert driver_version.text.split(".")[0] == "550"
    assert cuda_version.text.split(".") == ["12", "4"]


# Even if we remove the CUDA Driver API, the base NVIDIA drivers are still present.
# This is mostly a curiosity, but it underscores the difference between
# ther raw device drivers, which allow the OS to operate the device,
# and the user-level C API to those drivers.


@app.function(gpu=GPU_CONFIG, image=base_image)
def remove_libcuda(verbose: bool = True):
    import os
    import subprocess
    import xml.etree.ElementTree as ET
    from pathlib import Path

    root = Path("/")
    shared_user_level_dir = root / "usr"
    shared_library_dir = shared_user_level_dir / "lib"
    shared_x86_dir = shared_library_dir / "x86_64-linux-gnu"

    # remove libnvidia-ml.so and related files
    for libcuda_file in shared_x86_dir.glob("libcuda*"):
        if verbose:
            print("removing", libcuda_file)
        os.remove(libcuda_file)
    if verbose:
        print()  # empty line

    xml_output = subprocess.run(
        ["nvidia-smi", "-q", "-u", "-x"], capture_output=True, check=False
    ).stdout
    if verbose:
        print("nvidia-smi still runs!")

    root = ET.fromstring(xml_output)
    assert root.find("driver_version").text.split(".")[0] == "535"
    if verbose:
        print("the NVIDIA drivers are still present")
    assert root.find("cuda_version").text.lower() == "not found"
    if verbose:
        print("but the CUDA Driver API is gone:", "\n")
        subprocess.run(["nvidia-smi"])


# Even with nothing but the drivers and their API,
# we can still run packaged CUDA programs.
#
# The function below does this in a silly, non-standard way,
# just to get the point across:
# we send the compiled CUDA program as a bunch of bytes, then run it.


@app.function(gpu=GPU_CONFIG, image=base_image)
def raw_cuda(prog: bytes, with_libcuda: bool = True):
    import os
    import subprocess

    if not with_libcuda:
        remove_libcuda.local(verbose=False)

    # open the file in _w_ritable _b_inary mode
    with open("./prog", "wb") as f:
        f.write(prog)

    os.chmod("./prog", 0o755)  # make the program executable
    subprocess.run(["./prog"])


# But to use that, we'll need a compiled CUDA program to run.
# So let's install the NVIDIA CUDA compiler (nvcc) and the CUDA Toolkit.
# We'll do this on a new image, to underscore that we don't need to install
# these dependencies just to run CUDA programs.

arch = (
    "x86_64"  # instruction set architecture for the CPU, all Modal machines are x86_64
)
distro = "debian11"  # the distribution and version number of our OS (GNU/Linux)
filename = "cuda-keyring_1.1-1_all.deb"
cuda_keyring_url = f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/{arch}/{filename}"


cudatoolkit_image = (
    base_image.apt_install("wget")
    .run_commands(
        [  # we need to get hold of NVIDIA's CUDA keyring to verify the installation
            f"wget {cuda_keyring_url}",
            f"dpkg -i {filename}",
        ]  # otherwise we can't be sure the binaries are from NVIDIA
    )
    .apt_install("cuda-compiler-12-2")  # MUST BE <= 12.4!
    .env({"PATH": "/usr/local/cuda/bin:$PATH"})
)

# Now we can use nvcc to compile a CUDA program.
#
# We've included a simple CUDA program with this example.
# It's a translation of the famous fast inverse square root algorithm
# from Quake II Arena into CUDA C. This is only intended as a fun illustrative example;
# contemporary GPUs include direct instruction-level support for inverse square roots.
#
# To make the example more interesting, we've run it at "GPU scale":
# we calculate the result of the algorithm on every single 32-bit floating point number.
#
# We add the source code to our Modal environment by mounting the local files,
# adding them to the filesystem of the container running the function.
# We return the resulting binary as a stream of bytes.


@app.function(
    memory=1024 * 32,  # 32 GB of RAM
    gpu=GPU_CONFIG,
    image=cudatoolkit_image,
    mounts=[
        modal.Mount.from_local_file("invsqrt_kernel.cu", "/root/invsqrt_kernel.cu"),
        modal.Mount.from_local_file("every_invsqrt.cu", "/root/every_invsqrt.cu"),
    ],
)
def compile(and_run: bool = False):
    import subprocess
    from pathlib import Path

    cc_code = GPU_CAPABILITY_CODE

    # settings below are designed for easily-understood PTX, not performance!
    assert not subprocess.run(
        [
            "nvcc",  # run nvidia cuda compiler (which also calls the gcc toolchain)
            f"-arch=compute_{cc_code}",  # generate PTX machine code compatible with the GPU architecture
            f"-code=sm_{cc_code},compute_{cc_code}",  # and a binary optimized for the GPU architecture
            "-g",  # include debug symbols
            "-O0",  # no optimization from nvcc, keep PTX assembly simple
            "-Xcompiler",  # and for the gcc toolchain
            "-Og",  # also limit the optimization
            "-v",  # show verbose output
            "-lineinfo",  # and line numbers in machine code
            "-o",  # and send output to
            "invsqrt_demo",  # a binary called invsqrt_demo
            "invsqrt_kernel.cu",  # compiling the kernel
            "every_invsqrt.cu",  # and the host code
            "-lcuda",  # and linking in some symbols from the CUDA driver API
            # note that cudart is linked by default
        ],
        check=True,
    ).returncode

    assert not subprocess.run(["./invsqrt_demo"], capture_output=not and_run).returncode

    return Path("./invsqrt_demo").read_bytes()


@app.function(gpu=GPU_CONFIG, image=cudatoolkit_image)
def dump_ptx(prog: bytes) -> str:
    """Extracts the GPU assembly code using cuobjdump a binary-munging tool from the CUDA Toolkit."""
    import subprocess

    with open("./prog", "wb") as f:
        f.write(prog)

    result = subprocess.run(
        ["cuobjdump", "-ptx", "./prog"], capture_output=True, text=True
    )

    if result.returncode:
        raise Exception(result.stderr)

    return result.stdout


# The `main` function below puts this all together:
# it compiles the CUDA program, then shows the output of `nvidia-smi`
# in the environment it will be running in,
# then runs the program. Note that we run it on the base image,
# _without_ `libcudart` or any other dependencies installed.
#
# You can pass the flag `--no-with-libcuda` to see what happens
# when the CUDA Driver API is removed at run time.
# While `nvidia-smi` still runs, it no longer
# reports a CUDA version, and the program cannot run.


@app.local_entrypoint()
def main(with_libcuda: bool = True):
    print(f"🔥 compiling our CUDA kernel on {GPU_CONFIG}")
    prog = compile.remote()
    with open("invsqrt_demo", "wb") as f:
        f.write(prog)

    path = Path("ptx") / f"invsqrt_demo_{GPU_CAPABILITY_CODE}.ptx"
    print(f"🔥 dumping the PTX assembly code to {path}")
    ptx = dump_ptx.remote(prog)
    with open(path, "w") as f:
        f.write(ptx)

    if with_libcuda:
        print("🔥 showing nvidia-smi output")
        nvidia_smi.remote()
    else:
        print("🔥 removing libcuda to show what breaks")

    print("🔥 running our CUDA kernel")
    with open("invsqrt_demo", "rb") as f:
        prog = f.read()

    print("🔥 running on the base Modal image, no dependencies added")
    raw_cuda.remote(prog, with_libcuda)


def check_nvidia_smi(xml_output):
    """Utility for parsing version info from nvidia-smi's XML output"""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_output)

    return root.find("driver_version"), root.find("cuda_version")
