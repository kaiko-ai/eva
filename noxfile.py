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
def version(session: nox.Session) -> None:
    """Fetches and prints the version of the library."""
    session.run_always("pdm", "self", "add", "pdm-version", external=True)
    session.run("pdm", "version", external=True)


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
    session.run_always("pdm", "install", "--group", "docs", external=True)
    session.run("pdm", "run", "mike", *args)


@nox.session
def build(session: nox.Session) -> None:
    """Builds the source and wheel distributions."""
    session.run("pdm", "build")


@nox.session
def publish(session: nox.Session) -> None:
    """Builds and publishes the source and wheel distributions."""
    session.run("pdm", "publish")


@nox.session
def release(session: nox.Session) -> None:
    """Prepares a release by bumping version, building, and creating a commit.
    
    Usage:
      Make a patch release (0.2.0 -> 0.2.1):
      >>> nox -s release -- patch
      
      Make a minor release (0.2.1 -> 0.3.0):
      >>> nox -s release -- minor
      
      Make a major release (0.3.0 -> 1.0.0):
      >>> nox -s release -- major
      
      Prepare a specific version:
      >>> nox -s release -- to 1.2.3
    """
    if not session.posargs:
        session.error("You must specify a release type: patch, minor, major, or 'to X.Y.Z'")
    
    bump_cmd = session.posargs
    
    # Run version command to see current version
    session.run_always("pdm", "self", "add", "pdm-version", external=True)
    session.run("pdm", "version", external=True)
    
    # Bump version according to args
    session.run_always("pdm", "self", "add", "pdm-bump", external=True)
    session.run("pdm", "bump", *bump_cmd, external=True)
    
    # Run lint and tests
    session.run("nox", "-s", "lint", external=True)
    session.run("nox", "-s", "check", external=True)
    session.run("nox", "-s", "test", external=True)
    
    # Build artifacts
    session.run("nox", "-s", "build", external=True)
    
    # Get the new version for commit message
    import configparser
    config = configparser.ConfigParser()
    config.read("pyproject.toml", encoding="utf-8")
    version = config["project"]["version"].strip('"')
    
    # Create git commit with version
    session.run("git", "add", "pyproject.toml", external=True)
    session.run("git", "commit", "-m", f"chore: bump version to {version}", external=True)
    
    # Create a tag
    session.run("git", "tag", "-a", f"v{version}", "-m", f"Release v{version}", external=True)
    
    # Instructions for next steps
    print(f"\n\nRelease v{version} prepared!\n")
    print("Next steps:")
    print("1. Review the changes: git show")
    print("2. Push the commit: git push origin HEAD")
    print("3. Push the tag to trigger the release: git push origin v{version}")
    print("   Or run the Release workflow in GitHub Actions\n")
