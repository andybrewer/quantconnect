# region imports
from AlgorithmImports import *
# endregion
from datetime import date, datetime
from pydantic import BaseModel
import numpy as np

from ..utils.other import shift_to_anchor


class OpeningRange(BaseModel):
    high: float
    low: float


class OpeningRangeFactory:
    def __init__(self, name: str, opening_ranges_window_minutes: int) -> None:
        self.name: str = name
        self.opening_ranges_by_date: dict[date, OpeningRange] = {}
        self.opening_ranges_window_minutes: int = opening_ranges_window_minutes

    def add_new_opening_range_by_date(
        self, new_opening_range: OpeningRange, new_date: date
    ) -> None:
        assert new_date not in self.opening_ranges_by_date.keys()
        self.opening_ranges_by_date[new_date] = new_opening_range

    def update_opening_range_feature(
        self,
        current_moment: datetime,
        last_market_open_moment: datetime | None,
        market_high_by_date: dict[date, float],
        market_low_by_date: dict[date, float],
        day_opening_price: float,
        use_midpoint_logic: bool = False,
    ) -> None:
        if last_market_open_moment is None:
            return
        difference_between_last_market_open = current_moment - last_market_open_moment

        if (
            difference_between_last_market_open.total_seconds()
            != 60 * self.opening_ranges_window_minutes
        ):
            return

        current_date = current_moment.date()

        assert current_date in market_high_by_date.keys()
        assert current_date in market_low_by_date.keys()
        assert current_date not in self.opening_ranges_by_date.keys()

        high_of_the_day = market_high_by_date[current_date]
        low_of_the_day = market_low_by_date[current_date]

        if use_midpoint_logic:
            high_of_the_day, low_of_the_day = shift_to_anchor(
                high_of_the_day, low_of_the_day, day_opening_price
            )

        new_opening_range = OpeningRange(high=high_of_the_day, low=low_of_the_day)

        self.add_new_opening_range_by_date(new_opening_range, current_date)


def get_average_opening_range_thickness(
    opening_ranges_by_date: dict[date, OpeningRange], average_size: int
) -> float | None:
    all_values_to_consider: list[OpeningRange] = list(opening_ranges_by_date.values())[
        -1 * average_size :
    ]
    if len(all_values_to_consider) == 0:
        return None
    return np.average([(x.high - x.low) / x.low for x in all_values_to_consider])
