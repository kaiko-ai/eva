"""NOX lint and test automation API."""
import os

import nox

PACKAGE = "eva"
"""The name of the library."""

PYTHON_VERSIONS = ["3.10.13"]
"""The python versions to test on."""

LOCATIONS = "src", "tests", "noxfile.py"
"""The locations to add to nox."""

nox.options.sessions = "fmt", "lint", "check", "test"
"""List of all available sessions."""

nox.options.reuse_existing_virtualenvs = True
"""Whether to re-use the virtualenvs between runs."""

nox.options.stop_on_first_error = True
"""Whether to abort as soon as the first session fails."""

# so that PDM pick up the Python in the virtualenv correctly
os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

# unexpected behavior in some cases due to __pypackages__
os.environ.pop("PYTHONPATH", None)


@nox.session(python=PYTHON_VERSIONS[-1], tags=["fmt", "format"])
def fmt(session: nox.Session) -> None:
    """Fixes the source code format."""
    args = session.posargs or LOCATIONS
    session.run("pdm", "install", "--group", "lint", external=True)
    session.run("black", *args)
    session.run("isort", *args)
    session.run("ruff", "--fix", *args)


@nox.session(python=PYTHON_VERSIONS[-1], tags=["lint"])
def lint(session: nox.Session) -> None:
    """Checks the source code for programmatic and stylistic errors."""
    args = session.posargs or LOCATIONS
    session.run("pdm", "install", "--group", "lint", external=True)
    session.run("isort", "--check-only", *args)
    session.run("black", "--check", *args)
    session.run("ruff", *args)
    session.run("yamllint", *args)


@nox.session(python=PYTHON_VERSIONS[-1], tags=["check"])
def check(session: nox.Session) -> None:
    """Performs statically type checking of the source code."""
    args = session.posargs or LOCATIONS
    session.run("pdm", "install", "--group", "dev", external=True)
    session.run("pyright", *args)


@nox.session(python=PYTHON_VERSIONS, tags=["test"])
def test(session: nox.Session) -> None:
    """Runs the unit tests of the source code."""
    session.run("pdm", "install", "--group", "dev", external=True)
    session.run("pdm", "run", "pytest", "--cov")
