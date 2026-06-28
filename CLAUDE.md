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
