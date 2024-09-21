"""Sample script for running a VS Code server on Modal.

Usage: modal run --detach vscode_server.py --gpu h100:2

And look for a URL and password in the output. Open the URL in your browser to access the VS Code server.

See https://modal.com/docs/guide/gpu#specifying-gpu-count for more information on GPU options.
"""
import os
import secrets
import socket
import subprocess
import threading
import time
import webbrowser
from typing import Any, Dict, Tuple

from modal import Queue, forward
import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

PORT = 8080
STARTUP_TIMEOUT = 30  # seconds

args: Dict[str, Any] = {
    "cpu": 4,
    "memory": 8000,
    "timeout": 8 * HOURS,
}

# see https://modal.com/docs/guide/cuda for details on using CUDA on Modal
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

app = modal.App(  # see https://modal.com/docs/guide/custom-container for details on how to define container images
    image=modal.Image.from_registry(  # base image w CUDA stack
        f"nvidia/cuda:{tag}",
        add_python="3.11",
    )
    .apt_install("curl")  # install system dependencies with `apt_install`
    .run_commands(  # run arbitrary bash commands with `run_commands`
        "curl -fsSL https://code-server.dev/install.sh | sh"
    )
    .dockerfile_commands(  # run arbitrary Dockerfile commands with `dockerfile_commands`
        "ENTRYPOINT []"
    )
    .pip_install("torch")  # install python dependencies with `pip_install`
)


@app.function(
    cpu=args.get("cpu"),
    gpu=args.get("gpu"),
    memory=args.get("memory"),
    timeout=args.get("timeout"),
)
def run_vscode(q: Queue):
    # os.chdir("/home/coder")
    token = secrets.token_urlsafe(13)
    with forward(PORT) as tunnel:
        url = tunnel.url
        threading.Thread(target=wait_for_port, args=((url, token), q)).start()
        subprocess.run(
            ["/usr/bin/code-server", "--bind-addr", f"0.0.0.0:{PORT}", "."],
            env={**os.environ, "SHELL": "/bin/bash", "PASSWORD": token},
        )
    q.put("done")


@app.local_entrypoint()
def main(gpu=None):
    args["gpu"] = gpu
    with Queue.ephemeral() as q:
        run_vscode.spawn(q)  # spawn remote worker
        url, token = q.get()  # await URL and token
        time.sleep(1)  # Give VS Code a chance to start up
        print("\nVS Code on Modal, opening in browser...")
        print(f"   -> {url}")
        print(f"   -> password: {token}\n")
        webbrowser.open(url)
        assert q.get() == "done"


def wait_for_port(data: Tuple[str, str], q: Queue):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", PORT), timeout=STARTUP_TIMEOUT):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= STARTUP_TIMEOUT:
                raise TimeoutError(
                    f"Waited too long for port {PORT} to accept connections"
                ) from exc
    q.put(data)
