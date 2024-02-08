# EVA Style Guide

This document contains our style guides used in `eva`.

Our priority is _consistency_, so that developers can quickly
ingest and understand the entire codebase without being distracted
by style idiosyncrasies.

## General coding principles

_Q: How to keep code readable and maintainable?_
- [Don't Repeat Yourself (DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
- Use the lowest possible visibility for a variable or method (i.e. make private if possible) -- see [Information Hiding / Encapsulation](https://pynative.com/python-encapsulation/)

_Q: How big should a function be?_
- [Single Level of Abstraction Principle (SLAP)](http://principles-wiki.net/principles:single_level_of_abstraction)
- [High Cohesion](http://principles-wiki.net/principles:high_cohesion) and [Low Coupling](http://principles-wiki.net/principles:low_coupling)

    TL;DR: functions should usually be quite small, and _do one thing_

## Python Style Guide

In general we follow the following regulations:
[PEP8](https://peps.python.org/pep-0008/), the 
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
and we expect [type hints/annotations](https://peps.python.org/pep-0484/).

### Docstrings

Our docstring style is derived from
[Google Python style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).


```py
def example_function(variable: int, optional: str | None = None) -> str:
    """An example docstring that explains what this functions do.

    Docs sections can be referenced via :ref:`custom text here <anchor-link>`.

    Classes can be referenced via :class:`eva.data.datamodules.DataModule`.

    Functions can be referenced via :func:`eva.data.datamodules.call.call_method_if_exists`.

    Example:

        >>> from torch import nn
        >>> import eva
        >>> eva.models.modules.HeadModule(
        >>>     head=nn.Linear(10, 2),
        >>>     criterion=nn.CrossEntropyLoss(),
        >>> )

    Args:
        variable: A required argument.
        optional: An optional argument.

    Returns:
        A description of the output string.
    """
    pass
```

#### Module docstrings

[PEP-8](https://peps.python.org/pep-0008/#documentation-strings) and [PEP-257](https://peps.python.org/pep-0257/) indicate docstrings should have very specific syntax:

```py
"""One line docstring that shouldn't wrap onto next line."""
```

```py
"""First line of multiline docstring that shouldn't wrap.

Subsequent line or paragraphs.
"""
```

#### Constants docstrings
Public constants should usually have docstrings. Optional on
private constants. Docstrings on constants go underneath

```
SOME_CONSTANT = 3
"""Either a single-line docstring or multiline as per above."""
```

#### Function docstrings

All public functions should have docstrings following the
pattern shown below.

Each section can be omitted if there are no inputs, outputs,
or no notable exceptions raised, respectively.

```py
def fake_datamodule(
    n_samples: int, random: bool = True
) -> eva.data.datamodules.DataModule:
    """Generates a fake DataModule.

    It builds a :class:`eva.data.datamodules.DataModule` by generating
    a fake dataset with generated data while fixing the seed. It can
    be useful for debugging purposes.

    Args:
        n_samples: The number of samples of the generated datasets.
        random: Whether to generated randomly.

    Returns:
        A :class:`eva.data.datamodules.DataModule` with generated random data.

    Raises:
        ValueError: If `n_samples` is `0`.
    """
    pass
```

#### Class docstrings

All public classes should have class docstrings following the
pattern shown below.

```py
class DataModule(pl.LightningDataModule):
    """DataModule encapsulates all the steps needed to process data.

    It will initialize and create the mapping between dataloaders and
    datasets. During the `prepare_data`, `setup` and `teardown`, the
    datamodule will call the respectively methods from all the datasets,
    given that they are defined.
    """

    def __init__(
        self,
        datasets: schemas.DatasetsSchema | None = None,
        dataloaders: schemas.DataloadersSchema | None = None,
    ) -> None:
        """Initializes the datamodule.

        Args:
            datasets: The desired datasets. Defaults to `None`.
            dataloaders: The desired dataloaders. Defaults to `None`.
        """
        pass
```
