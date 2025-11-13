# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime, timedelta

from ..models.models import MyCandle


def check_if_it_is_a_full_candle(
    all_daily_data: dict[datetime, MyCandle],
    start_moment: datetime,
    end_moment: datetime,
    full_candle_threshold: float,
) -> bool:
    assert start_moment in all_daily_data.keys()
    assert end_moment in all_daily_data.keys()
    assert start_moment <= end_moment
    all_moments = [start_moment]
    i = 1
    while True:
        new_moment = start_moment + timedelta(minutes=i)
        if new_moment > end_moment:
            break
        all_moments.append(new_moment)
        i += 1

    all_minute_data: list[MyCandle] = [all_daily_data[x] for x in all_moments]

    open_val = all_minute_data[0].open
    close_val = all_minute_data[-1].close
    high_val = max([x.high for x in all_minute_data])
    low_val = min([x.low for x in all_minute_data])

    full_height = high_val - low_val
    if full_height == 0:
        return True

    return (abs(open_val - close_val) / full_height) >= full_candle_threshold
