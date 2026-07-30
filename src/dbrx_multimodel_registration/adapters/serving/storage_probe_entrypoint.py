"""Models-from-code entrypoint for the serving storage-ceiling probe.

This file is passed to `mlflow.pyfunc.log_model(python_model=<this path>)` — NOT
imported and pickled. At serving load time the container EXECUTES this script and
uses the object registered via `set_model(...)`, so nothing here needs the project
package to be importable (that pickle-by-reference path is what caused
`ModuleNotFoundError: dbrx_multimodel_registration` on the endpoint).

Deliberately trivial: the model does nothing with its baked artifact. The whole
point of the test is whether Databricks/MLflow can materialize an N-GB `artifacts=`
payload onto the endpoint's disk at deploy time — the model just needs to load and
answer. So `predict` returns a constant and `load_context` is a no-op. Only
dependency is `mlflow` (stdlib otherwise), matching `pip_requirements=["mlflow"]`.
"""
import mlflow
from mlflow.models import set_model

# The artifact key the baked payload is logged under (see stress.log_probe_model).
# Duplicated here (not imported) so this script stays self-contained at load time.
ARTIFACT_KEY = "payload"


class HelloProbeModel(mlflow.pyfunc.PythonModel):
    """Trivial probe: proves the model deploys; ignores the baked artifact."""

    def predict(self, context, model_input):
        # One row in → one row out. Content is irrelevant; a successful response
        # is the signal that the endpoint loaded with its N-GB artifact attached.
        return ["hello world"]


set_model(HelloProbeModel())
