"""dbrx-mmd CLI — unified command-line interface."""

import typer

from dbrx_multimodel_registration.cli.commands import capacity as capacity_commands
from dbrx_multimodel_registration.cli.commands import fleet as fleet_commands
from dbrx_multimodel_registration.cli.commands import setup as setup_commands

cli = typer.Typer(name="dbrx-mmd", help="dbrx-mmd — tools for the multi-model registration demo.")
cli.add_typer(setup_commands.app, name="setup")
cli.add_typer(capacity_commands.app, name="capacity")
cli.add_typer(fleet_commands.app, name="fleet")


def main() -> None:
    try:
        cli()
    except SystemExit as e:
        # Only re-raise on non-zero exit (actual error). Databricks serverless
        # treats SystemExit(0) as a failure.
        if e.code != 0:
            raise


if __name__ == "__main__":
    main()
