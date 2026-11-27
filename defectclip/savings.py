"""Parametric labor-time arithmetic for an inspection line.

Nothing in this module was measured on a factory. The caller passes every
input. Model throughput may be a number you measured elsewhere (this repo
records one CPU measurement in results/throughput.json); the dollar figure is
still just the formula below.

    manual_hours = parts_per_hour * manual_seconds_per_part / 3600
    model_hours  = parts_per_hour / model_images_per_second / 3600
    hour_delta   = manual_hours - model_hours
    labor_delta  = hour_delta * labor_usd_per_hour

hour_delta is positive when the manual look takes more time than the encode.
It ignores misses, false alarms, review of the queue, and rework. It is not
a measured saving.
"""

from __future__ import annotations


def estimate_labor_delta(
    parts_per_hour: float,
    manual_seconds_per_part: float,
    labor_usd_per_hour: float,
    model_images_per_second: float,
) -> dict:
    if parts_per_hour < 0 or manual_seconds_per_part < 0 or labor_usd_per_hour < 0:
        raise ValueError("parts, manual time, and labor cost must be non-negative")
    if model_images_per_second <= 0:
        raise ValueError("model throughput must be positive")
    manual_hours = parts_per_hour * manual_seconds_per_part / 3600.0
    model_seconds = 1.0 / model_images_per_second
    model_hours = parts_per_hour * model_seconds / 3600.0
    hour_delta = manual_hours - model_hours
    return {
        "parts_per_hour": parts_per_hour,
        "manual_seconds_per_part": manual_seconds_per_part,
        "labor_usd_per_hour": labor_usd_per_hour,
        "model_images_per_second": model_images_per_second,
        "model_seconds_per_part": model_seconds,
        "manual_labor_hours_per_clock_hour": manual_hours,
        "model_compute_hours_per_clock_hour": model_hours,
        "hours_difference_per_clock_hour": hour_delta,
        "labor_cost_difference_usd_per_clock_hour": hour_delta * labor_usd_per_hour,
        "kind": "parametric illustration, not a measured saving",
    }
