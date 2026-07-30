# Code Design
Use the Python Domain, Port, and Adapters architecture pattern. Entities (python dataclasses
should go in domains module), Port are the abstraction concepts/interfaces of the overall
solution. Adapters are the specific implementations of the Ports.

The main script to run the model should be in main.py. main.py should import everything it needs
from the domain, port, and adapters modules.

## Adapter / Port category organization

`adapters/` and `ports/` are organized into matching category subdirectories. The category
name on both sides identifies the same domain concept — opening `ports/logging/` shows the
contract, opening `adapters/logging/` shows alternative implementations of it.

Categories:

- `logging/` — MLflow logging strategies (the variants of how models get registered)
- `storage/` — persistence of the run plan and artifact bundles
- `training/` — reference model trainer and training simulator
- `data_generation/` — synthetic demand data generation
- `serving/` — inference-time per-SKU model lookup (adapters only — no port yet, intentionally
  per the port-adapter-scope discipline: ports come with their second adapter, not ahead of it)

`adapters/__init__.py` and `ports/__init__.py` re-export every public class flat, so
`from dbrx_multimodel_registration.adapters import UCTableLoggingStrategy` still works.
Prefer the subdir-aware import (`from ...adapters.logging import UCTableLoggingStrategy`)
in new code.

## Why this architecture (the operating principle)

The point of ports/domains/adapters here is **composability under changing details**.
Ports own the *orchestration and shape* of the solution; adapters own the *implementation
details* that we expect to change. When the underlying mechanism changes (a new serving
topology, a different storage backend, another logging strategy), we add or swap an
adapter — the port and the code that composes ports stay put. That stability at the seams
is the whole payoff.

Consequences that follow from that principle:

- **Domains are the shared vocabulary.** Entities (frozen dataclasses) are the value
  objects passed across seams — inputs and return types of port methods. Keep behavior
  that is pure and mechanism-independent in domains so both build-time and inference-time
  code can share it without dragging in heavy dependencies.
- **A port names a responsibility, not a topology.** Name the contract for what it
  guarantees (e.g. "deploy a serving topology"), not for how the first adapter happens to
  do it (e.g. "a fleet"). This keeps the second adapter — which may choose a different
  shape — a natural fit rather than a contortion.
- **Introduce a port when the second adapter is real, not speculative.** One adapter with
  no credible sibling stays adapter-only. When a genuine alternative implementation is
  anticipated, the port ahead of it is justified — it's what makes the swap cheap.
- **Keep the hot path import-light.** Adapters that run in both a heavy context (Spark /
  MLflow deploy) and a light one (an inference endpoint) should keep heavy imports lazy
  (inside the methods that need them), so the light path can use the class without paying
  for the build-time dependency graph.

## Constitution
- Be concise, unless specified otherwise. 
- Response should be in markdown, my terminal can render rich text. I need headings for major points, and subheadings for subpoints. Feel free to use bullet points for granualr details. 
  This is so I can close the loop faster and sift to the relevant details quicker. 
