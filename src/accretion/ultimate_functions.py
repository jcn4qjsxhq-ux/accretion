"""Compatibility re-export module for the split Ultimate helpers."""

from __future__ import annotations

try:
    from . import ultimate_file_organisers as _ultimate_file_organisers
    from . import ultimate_older_functions as _ultimate_older_functions
    from . import ultimate_synthetic_data as _ultimate_synthetic_data
    from . import ultimate_physics as _ultimate_physics
    from . import ultimate_helpers as _ultimate_helpers
    from . import ultimate_fitting as _ultimate_fitting
    from . import ultimate_fitting_types as _ultimate_fitting_types
    from . import ultimate_visualisation as _ultimate_visualisation
    from .ultimate_common import R_sun, M_sun, year, c, RED_EXCESS_DEFAULT_T_BB, RED_EXCESS_DEFAULT_R_BB
    from .ultimate_file_organisers import *
    from .ultimate_older_functions import *
    from .ultimate_synthetic_data import *
    from .ultimate_physics import *
    from .ultimate_helpers import *
    from .ultimate_fitting import *
    from .ultimate_fitting_types import *
    from .ultimate_visualisation import *
except ImportError:
    import ultimate_file_organisers as _ultimate_file_organisers
    import ultimate_older_functions as _ultimate_older_functions
    import ultimate_synthetic_data as _ultimate_synthetic_data
    import ultimate_physics as _ultimate_physics
    import ultimate_helpers as _ultimate_helpers
    import ultimate_fitting as _ultimate_fitting
    import ultimate_fitting_types as _ultimate_fitting_types
    import ultimate_visualisation as _ultimate_visualisation
    from ultimate_common import R_sun, M_sun, year, c, RED_EXCESS_DEFAULT_T_BB, RED_EXCESS_DEFAULT_R_BB
    from ultimate_file_organisers import *
    from ultimate_older_functions import *
    from ultimate_synthetic_data import *
    from ultimate_physics import *
    from ultimate_helpers import *
    from ultimate_fitting import *
    from ultimate_fitting_types import *
    from ultimate_visualisation import *

__all__ = [
    'R_sun', 'M_sun', 'year', 'c', 'RED_EXCESS_DEFAULT_T_BB', 'RED_EXCESS_DEFAULT_R_BB',
]
__all__ += _ultimate_file_organisers.__all__
__all__ += _ultimate_older_functions.__all__
__all__ += _ultimate_synthetic_data.__all__
__all__ += _ultimate_physics.__all__
__all__ += _ultimate_helpers.__all__
__all__ += _ultimate_fitting.__all__
__all__ += _ultimate_fitting_types.__all__
__all__ += _ultimate_visualisation.__all__
