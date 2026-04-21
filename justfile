# rod-sim3d — quick recipes.
# Run `just` (no arguments) to see the list.

set shell := ["bash", "-eu", "-c"]

default:
    @just --list --unsorted

# --- Setup -----------------------------------------------------------------

# Install everything (project + dev deps).
sync:
    uv sync

# --- Quality ---------------------------------------------------------------

# Format, lint, type-check, measure complexity, run tests. One button, six checks.
qa:
    uv run ruff format .
    uv run ruff check .
    uv run ty check src tests
    uv run radon cc -s -a src
    uv run radon mi -s src
    uv run pytest

# Faster subset: just format + lint + tests (skips radon/ty).
ci:
    uv run ruff format --check .
    uv run ruff check .
    uv run pytest

test:
    uv run pytest

# --- Pass-through cylinder mode (NONE) -------------------------------------
#
# Cylinders have a visible radius and bounce off the walls, but pass through
# each other. The viewer highlights the convex intersection polytope of every
# overlapping pair in translucent magenta. See configs/cylinders.toml.

# Run the pass-through simulation offline and store trajectory + overlaps to SQLite.
compute config="configs/cylinders.toml" db="runs/cylinders.db" frames="1200" quadrature="16":
    uv run rod-sim3d compute --config {{config}} --db {{db}} --frames {{frames}} --overlap-quadrature {{quadrature}}

# Replay a previously computed DB interactively (PyVista).
replay db="runs/cylinders.db" backend="pyvista":
    uv run rod-sim3d replay --db {{db}} --backend {{backend}}

# Compute + replay: the canonical pass-through demo.
demo:
    just compute
    just replay

# Short compute + headless replay, suitable for smoke tests (CI-friendly).
demo-short:
    uv run rod-sim3d compute --config configs/cylinders.toml --db runs/cylinders_short.db --frames 60
    uv run rod-sim3d replay --db runs/cylinders_short.db --backend none

# --- Hard-contact (rigid-body) cylinder mode ------------------------------
#
# Capsules collide impulsively with a configurable restitution coefficient.
# See configs/cylinders_hard.toml for knobs.

# Compute a hard-contact simulation.
hard-compute config="configs/cylinders_hard.toml" db="runs/cylinders_hard.db" frames="1500":
    uv run rod-sim3d compute --config {{config}} --db {{db}} --frames {{frames}}

# Compute + replay the hard-contact demo.
hard:
    just hard-compute
    uv run rod-sim3d replay --db runs/cylinders_hard.db --backend pyvista

# --- Housekeeping ---------------------------------------------------------

clean:
    rm -rf runs/*.db runs/*.npz runs/*.json
