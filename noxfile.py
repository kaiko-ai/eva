"""Packaging, testing and release process automation API.

The automation is build using `nox` (https://nox.thea.codes/en/stable/).

Quick guide:

To run all the sessions:
>>> nox

To run only the code-quality check tagged session:
>>> nox -t quality

To run only the unit-test tagged session:
>>> nox -t tests

To run a session (fmt, lint, check, test):
>>> nox -s {fmt,lint,check,test}

To run a session and pass extra arguments:
>>> nox -s test -- tests/eva/metrics/test_average_loss.py
"""

import os

import nox

PACKAGE = "eva"
"""The name of the library."""

PYTHON_VERSIONS = ["3.10"]
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


@nox.session(tags=["fmt", "format", "quality"])
def fmt(session: nox.Session) -> None:
    """Formats the source code format."""
    args = session.posargs or LOCATIONS
    session.run_always("pdm", "install", "--no-default", "--group", "lint", external=True)
    session.run("isort", *args)
    session.run("black", *args)
    session.run("ruff", "check", "--fix-only", *args)


@nox.session(tags=["lint", "quality"])
def lint(session: nox.Session) -> None:
    """Checks the source code for programmatic, stylistic and security violations."""
    args = session.posargs or LOCATIONS
    session.run_always("pdm", "install", "--no-default", "--group", "lint", external=True)
    session.run("isort", "--check-only", *args)
    session.run("black", "--check", *args)
    session.run("ruff", "check", *args)
    session.run("yamllint", *args)
    session.run("bandit", "-q", "-c", "pyproject.toml", "-r", *args)


@nox.session(tags=["check", "quality"])
def check(session: nox.Session) -> None:
    """Performs statically type checking of the source code."""
    args = session.posargs or LOCATIONS
    session.run_always("pdm", "install", "--group", "typecheck", "--group", "all", external=True)
    session.run("pyright", *args)


@nox.session(python=PYTHON_VERSIONS, tags=["unit-tests", "tests"])
def test(session: nox.Session) -> None:
    """Runs the tests and code coverage analysis session of all the source code."""
    session.notify("test_core")
    session.notify("test_vision")


@nox.session(python=PYTHON_VERSIONS, tags=["unit-tests", "tests"])
def test_core(session: nox.Session) -> None:
    """Runs the tests and code coverage analysis session of the core source code."""
    args = session.posargs or ["--cov"]
    session.run_always("pdm", "install", "--group", "test", external=True)
    session.run("pytest", os.path.join("tests", "eva", "core"), *args)


@nox.session(python=PYTHON_VERSIONS, tags=["unit-tests", "tests"])
def test_vision(session: nox.Session) -> None:
    """Runs the tests and code coverage analysis session of the vision source code."""
    args = session.posargs or ["--cov"]
    session.run_always("pdm", "install", "--group", "test", "--group", "vision", external=True)
    session.run("pytest", os.path.join("tests", "eva", "vision"), *args)


@nox.session(python=PYTHON_VERSIONS, tags=["unit-tests", "tests"])
def test_all(session: nox.Session) -> None:
    """Runs the tests and code coverage analysis session of all the source code."""
    args = session.posargs or ["--cov"]
    session.run_always("pdm", "install", "--group", "test", "--group", "all", external=True)
    session.run("pytest", *args)


@nox.session(python=PYTHON_VERSIONS, tags=["ci"])
def ci(session: nox.Session) -> None:
    """Runs the CI workflow."""
    session.notify("lint")
    session.notify("check")
    session.notify("test")


@nox.session
def bump(session: nox.Session) -> None:
    """Bumps the version of the library.

    Usage:
      Update patch (0.0.1 -> 0.0.2)
      >>> nox -s bump -- micro

      Update minor (0.0.1 -> 0.1.0)
      >>> nox -s bump -- minor

      Update major (0.0.1 -> 1.0.0)
      >>> nox -s bump -- minor

      Update dev (0.0.1 -> 0.0.1.dev1)
      >>> nox -s bump -- dev
    """
    session.run_always("pdm", "self", "add", "pdm-bump", external=True)
    session.run("pdm", "bump", *session.posargs, external=True)


@nox.session
def docs(session: nox.Session) -> None:
    """Builds and deploys the code documentation."""
    args = session.posargs or []
    session.run_always("pdm", "install", "--no-default", "--group", "docs", external=True)
    session.run("pdm", "run", "mike", *args)


@nox.session
def build(session: nox.Session) -> None:
    """Builds the source and wheel distributions."""
    session.run("pdm", "build")


@nox.session
def publish(session: nox.Session) -> None:
    """Builds and publishes the source and wheel distributions."""
    session.run("pdm", "publish")
