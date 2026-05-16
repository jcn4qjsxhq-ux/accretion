"""Compatibility imports for functions that now live in titled modules.

New code should import daily-data helpers from ``ultimate_file_organisers`` and
plotting/date helpers from ``ultimate_visualisation``.
"""

from __future__ import annotations

try:
    from .ultimate_file_organisers import (
        assign_jd_day,
        repair_flux_from_magnitude,
        average_daily_measurements,
        get_daily_data,
    )
    from .ultimate_visualisation import (
        julian_to_calendar,
        add_gregorian_top_axis,
        get_model_predictions,
    )
except ImportError:
    from ultimate_file_organisers import (
        assign_jd_day,
        repair_flux_from_magnitude,
        average_daily_measurements,
        get_daily_data,
    )
    from ultimate_visualisation import (
        julian_to_calendar,
        add_gregorian_top_axis,
        get_model_predictions,
    )


__all__ = [
    'assign_jd_day',
    'repair_flux_from_magnitude',
    'average_daily_measurements',
    'get_daily_data',
    'julian_to_calendar',
    'add_gregorian_top_axis',
    'get_model_predictions',
]
