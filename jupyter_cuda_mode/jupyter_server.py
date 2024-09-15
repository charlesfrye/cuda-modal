"""Runs a GPU-accelerated Jupyter server on Modal.

See the example notebooks for samples of kernel-development workflows.

The only local dependency is the Modal client library, which can be installed with `pip install modal`.
If you've never used Modal before, run `modal setup` to configure your account.

Usage:
    - `modal run jupyter_server` to set up the remote filesystem
    - `modal deploy jupyter_server` to deploy the Jupyter server
    - `modal shell jupyter_server` to spin up a shell in a new replica (good for debugging the environment)

You can find the server at the URL printed in the logs when you run the deploy command.
It will end with `cuda-modal-serve.modal.run`.
"""
import os
from pathlib import Path
import subprocess

import modal

GPU = modal.gpu.A10G()
JUPYTER_TOKEN = "1234"  # Change me to something secret!

app = modal.App(
    "cuda-modal-jupyter-python",
    image=modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"  # pytorch + CUDA stack
        # "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"  # CUDA stack
    ).pip_install(
        "jupyter==1.0.0",
        "matplotlib==3.9.2",
        "notebook<7",
        "numba==0.60.0",  # write CUDA kernels in Python
        "numpy~=2.0.0",
        "wurlitzer==3.1.1",  # capture C stedrr/stdout info in notebooks
    ),
)
volume = modal.Volume.from_name("cuda-modal", create_if_missing=True)

WORKING_DIR = Path("/root") / "examples"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES
PORT = 8888


@app.function(
    concurrency_limit=1,
    volumes={WORKING_DIR: volume},
    timeout=30 * MINUTES,
    allow_concurrent_inputs=1000,
    gpu=GPU,
)
@modal.web_server(port=PORT, label="cuda-modal-serve")
def serve():
    _jupyter_process = subprocess.Popen(
        [
            "jupyter",
            "notebook",
            "--no-browser",
            "--allow-root",
            "--ip=0.0.0.0",
            f"--port={PORT}",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.allow_remote_access=1",
            "--NotebookApp.show_banner=False",  # hide migration banner
        ],
        env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
    )


if JUPYTER_TOKEN == "1234":
    print(
        "WARNING: Jupyter token set to default! Change JUPYTER_TOKEN in the deployment script."
    )


@app.local_entrypoint()
def main():
    seed_volume(volume, force=True)


def seed_volume(volume, force=False):
    """Adds starter files to the remote Volume on first run."""
    local_directory = Path(__file__).parent / "examples"
    with volume.batch_upload(force=force) as batch:
        batch.put_directory(local_directory / "lecture_003", "lecture_003")
        batch.put_directory(local_directory / "lecture_005", "lecture_005")
        batch.put_file(local_directory / "utils.py", "utils.py")
