# modified from original by cmpadden, https://github.com/cmpadden/modal-playground/blob/main/modal-cupy-conway.py
"""CuPy Conway's Game of Life

USAGE

    modal run modal-cupy-conway.py
"""

import modal

app = modal.App("example-cupy-conway")

cuda_version = "12.4.0"
flavor = "devel"
os = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install("cupy-cuda12x")
    .entrypoint([])
)


ITERATIONS = 10
UNIVERSE_DIMENSION_M = 10
UNIVERSE_DIMENSION_N = 25


@app.function(gpu="any", image=image)
def run_cupy_conways_game_of_life():
    import cupy as cp
    import cupyx.scipy.ndimage as nd

    kernel = cp.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=cp.uint8)

    universe = cp.random.randint(2, size=(UNIVERSE_DIMENSION_M, UNIVERSE_DIMENSION_N))

    for _ in range(ITERATIONS):
        neighbor_counts = nd.convolve(universe, kernel, mode="constant")

        universe = (
            universe & cp.isin(neighbor_counts, cp.array([2, 3]))  # cell survived
        ) | (
            ~universe & (neighbor_counts == 3)  # cell was birthed
        )

        print(universe)
