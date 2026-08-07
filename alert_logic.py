"""
alert_logic.py

Helpers for changing alert state and recording timeline events only when
those states actually transition.
"""

from event_log_io import log_event
from robot_state_io import read_field, write_field


def set_alert_state(field_name, active, active_message, clear_message, metadata=None):
    previous = bool(read_field(field_name, max_age_seconds=86400, default=False))
    active = bool(active)
    write_field(field_name, active)

    if active == previous:
        return None

    return log_event(
        kind=field_name,
        severity="critical" if active else "info",
        message=active_message if active else clear_message,
        metadata=metadata,
    )
