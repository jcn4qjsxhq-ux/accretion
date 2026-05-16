"""Standalone shared imports, constants, and optional dependency helpers.

This file is the importable source of truth for the notebook functions so they
can be reused from scripts and other notebooks.
"""

from __future__ import annotations

import importlib
import os
import glob
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from datetime import datetime
from functools import lru_cache

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))


class _MissingDependency:
    def __init__(self, package_name: str):
        self.package_name = package_name

    def __getattr__(self, name: str):
        raise ImportError(
            f"Optional dependency '{self.package_name}' is required for this function"
        )

    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"Optional dependency '{self.package_name}' is required for this function"
        )


def _optional_import(module_name: str, package_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return _MissingDependency(package_name or module_name)


def _optional_attr(module_name: str, attr_name: str, package_name: str | None = None):
    module = _optional_import(module_name, package_name)
    if isinstance(module, _MissingDependency):
        return module
    return getattr(module, attr_name)


mpl = _optional_import("matplotlib")
plt = _optional_import("matplotlib.pyplot", "matplotlib")
figure = _optional_attr("matplotlib.pyplot", "figure", "matplotlib")
colors = _optional_import("matplotlib.colors", "matplotlib")
Normalize = _optional_attr("matplotlib.colors", "Normalize", "matplotlib")
LogNorm = _optional_attr("matplotlib.colors", "LogNorm", "matplotlib")
animation = _optional_import("matplotlib.animation", "matplotlib")
Axes3D = _optional_attr("mpl_toolkits.mplot3d", "Axes3D", "matplotlib")
DateFormatter = _optional_attr("matplotlib.dates", "DateFormatter", "matplotlib")
FuncFormatter = _optional_attr("matplotlib.ticker", "FuncFormatter", "matplotlib")
cm = _optional_attr("matplotlib", "cm", "matplotlib")

np = _optional_import("numpy")
Polynomial = _optional_attr("numpy.polynomial", "Polynomial", "numpy")

constants = _optional_import("scipy.constants", "scipy")
optimize = _optional_import("scipy.optimize", "scipy")
spectrogram = _optional_attr("scipy.signal", "spectrogram", "scipy")
find_peaks = _optional_attr("scipy.signal", "find_peaks", "scipy")
curve_fit = _optional_attr("scipy.optimize", "curve_fit", "scipy")
minimize = _optional_attr("scipy.optimize", "minimize", "scipy")

fits = _optional_import("astropy.io.fits", "astropy")
Time = _optional_attr("astropy.time", "Time", "astropy")

px = _optional_import("plotly.express", "plotly")
pio = _optional_import("plotly.io", "plotly")
go = _optional_import("plotly.graph_objects", "plotly")
make_subplots = _optional_attr("plotly.subplots", "make_subplots", "plotly")
sample_colorscale = _optional_attr("plotly.colors", "sample_colorscale", "plotly")
tls = _optional_import("plotly.tools", "plotly")
display_html = _optional_attr("IPython.display", "display_html", "IPython")

pd = _optional_import("pandas")

_numba_njit = _optional_attr("numba", "njit", "numba")
if isinstance(_numba_njit, _MissingDependency):
    def njit(function=None, *args, **kwargs):
        if function is None:
            def decorator(inner_function):
                return inner_function

            return decorator
        return function
else:
    njit = _numba_njit


warnings.filterwarnings('ignore')

if not isinstance(pio, _MissingDependency):
    pio.renderers.default = 'notebook'

R_sun = 6.95500e8
M_sun = 1.989e30
year = 365.25 * 24 * 60 * 60
c = 299792458.0

RED_EXCESS_DEFAULT_T_BB = 1500.0
RED_EXCESS_DEFAULT_R_BB = 2 * 149597870700.0
SPECTRAL_REDE_THRESHOLD_MICRON = 2.0
SPECTRAL_REDE_BB_PLOT_FLOOR_JY = 1e-5
SPECTRAL_REDE_MAX_BLUE_BB_FRACTION = 0.10
SPECTRAL_REDE_BLUE_PENALTY_WEIGHT = 25.0
