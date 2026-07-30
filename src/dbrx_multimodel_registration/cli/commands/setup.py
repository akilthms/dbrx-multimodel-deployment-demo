"""dbrx-mmd CLI — project setup / bundle deployment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from dotenv import dotenv_values

app = typer.Typer(help="Project setup and bundle deployment.")


@app.command()
def deploy(
    target: str = typer.Option("dev", "-t", "--target", help="Deployment target (e.g. dev, staging)."),
    validate: bool = typer.Option(True, help="Run `bundle validate` before deploying."),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Pass --auto-approve to `bundle deploy`."),
) -> None:
    """Load .env, then validate and deploy the Databricks bundle.

    Reproduces the `set -a; source .env; databricks bundle deploy` flow: the
    profile named in DATABRICKS_CONFIG_PROFILE is passed as --profile (supplying
    host + credentials), and BUNDLE_VAR_* entries are exported into the
    subprocess environment so ${var.*} values resolve.
    """
    project_root = Path(__file__).resolve().parents[4]
    values = dotenv_values(project_root / ".env")

    profile = values.get("DATABRICKS_CONFIG_PROFILE")
    if not profile:
        typer.echo("Error: DATABRICKS_CONFIG_PROFILE not set in .env", err=True)
        raise typer.Exit(code=1)

    env = {**os.environ, **{k: v for k, v in values.items() if v is not None}}

    if validate:
        _run(["databricks", "bundle", "validate", "-t", target, "--profile", profile], project_root, env)

    deploy_cmd = ["databricks", "bundle", "deploy", "-t", target, "--profile", profile]
    if auto_approve:
        deploy_cmd.append("--auto-approve")
    _run(deploy_cmd, project_root, env)

    typer.echo(f"Deployed target '{target}'.")


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    """Echo and run a subprocess; exit the CLI with its return code on failure."""
    typer.echo(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
