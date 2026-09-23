"""The JSON contract of the result files (``docs/SPECIFICATIONS/05-metrics.md`` §9).

Every metric file of the experiment is written through :func:`write_json` and read back through
:func:`read_json`, so that the files of the 30 runs are byte-comparable: keys sorted, floats
rounded to six significant digits, no ``NaN`` or ``Infinity`` tokens (which are not JSON and
which no result of `05` §9 is allowed to be).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.metrics.errors import MetricError

__all__ = ["read_json", "write_json"]

#: Significant digits kept for every float. ``05`` §9 asks for "floats rounded to 6 digits";
#: significant digits rather than decimal places, so that small but meaningful quantities
#: (permutation p-values, inherited shares at sigma_max = 96) do not round to zero.
SIGNIFICANT_DIGITS: int = 6


def _plain(obj: Any) -> Any:
    """Return ``obj`` with numpy scalars, numpy arrays and paths turned into JSON types.

    Floats are rounded to :data:`SIGNIFICANT_DIGITS` significant digits; dictionary keys are
    coerced to ``str`` so that the integer step keys of ``lsd_by_step`` survive a round trip as
    strings, which is what JSON allows.

    Parameters
    ----------
    obj : Any
        The object to convert.

    Returns
    -------
    Any
        A structure of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``.

    Raises
    ------
    MetricError
        If a float is not finite, or a value has no JSON representation.
    """
    if isinstance(obj, dict):
        return {str(key): _plain(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        return _plain(obj.item())
    if isinstance(obj, bool | str | type(None)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise MetricError(f"refusing to write a non-finite float: {obj!r}")
        return float(f"{obj:.{SIGNIFICANT_DIGITS}g}")
    if isinstance(obj, Path):
        return str(obj)
    raise MetricError(f"cannot serialise {type(obj).__name__} to JSON: {obj!r}")


def write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` to ``path`` under the result-file contract of ``05-metrics.md`` §9.

    Parameters
    ----------
    path : str or Path
        Destination file; its parent directory is created if needed.
    obj : Any
        The object to write. Dictionaries, sequences, numpy arrays and numpy scalars are
        converted; every float is rounded to six significant digits.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    MetricError
        If the object holds a non-finite float or a value with no JSON representation.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_plain(obj), sort_keys=True, indent=2, allow_nan=False)
    destination.write_text(text + "\n")
    return destination


def read_json(path: str | Path) -> Any:
    """Read a result file written by :func:`write_json`.

    Parameters
    ----------
    path : str or Path
        The file to read.

    Returns
    -------
    Any
        The parsed object.

    Raises
    ------
    MetricError
        If the file is missing or does not parse as JSON.
    """
    source = Path(path)
    try:
        return json.loads(source.read_text())
    except FileNotFoundError as error:
        raise MetricError(f"missing result file: {source}") from error
    except json.JSONDecodeError as error:
        raise MetricError(f"{source}: not valid JSON ({error})") from error
