"""Entry point for running the Kedro project as a package (`python -m fraud_detection`)."""

import sys
from pathlib import Path

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs):
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    kwargs["standalone_mode"] = not hasattr(sys, "ps1")
    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
