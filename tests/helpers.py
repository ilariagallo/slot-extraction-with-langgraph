from datetime import datetime

import arrow


def resolve_date_from_day_of_week(day_of_week):
    """Helper to resolves day of week into the closest date in the future.
    Returns a date in the format DD/MMM/YYYY"""
    day_of_week_mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    day_of_week_int = day_of_week_mapping[day_of_week.lower()]
    today = datetime.now()
    today_day_of_week_int = today.weekday()

    # Compute difference between today and day of the week provided by the user
    diff = (day_of_week_int - today_day_of_week_int) % 7

    # Add 7 days to diff if day of week provided is equal to current day of week
    if diff == 0:
        diff = diff + 7

    # Resolve date given the difference
    resolved_date = arrow.now().shift(days=diff).format("DD/MMM/YYYY")
    return resolved_date