# region imports
from AlgorithmImports import *
# endregion
import copy
from datetime import date, datetime, timedelta
from enum import Enum
import uuid

from pydantic import BaseModel

from src.models.models import MyCandle
from .opening_range import OpeningRange
from copy import deepcopy

class BoxBreakExitDirection(Enum):
    UP = 1
    DOWN = -1
    NONE = 0

class BoxBreakInsideClassification(str, Enum):
    UP = "up"
    DOWN = "down"
    MARKET_CLOSE = "market_close"


class BoxBreakOpeningRangeClassification(str, Enum):
    ABOVE = "above"
    BELLOW = "bellow"
    INSIDE = "inside"
    ACROSS = "across"


class ConsolidationBox(BaseModel):
    unique_id: str
    start_moment: datetime
    end_moment: datetime
    moment_of_creation: datetime
    high: float
    high_moment: datetime | None  # TODO: remove this optional type in the future
    low: float
    low_moment: datetime | None  # TODO: remove this optional type in the future
    box_break_inside: BoxBreakInsideClassification | None = None
    box_break_according_to_opening_range: BoxBreakOpeningRangeClassification | None = (
        None
    )
    had_a_hod_after_x_minutes: bool | None = None
    had_a_lod_after_x_minutes: bool | None = None

    had_a_hod_before_x_minutes: bool | None = None
    had_a_lod_before_x_minutes: bool | None = None

    broke_due_to_last_5_min_top25: bool = False
    broke_due_to_last_5_min_bottom25: bool = False

    exit_classification: BoxBreakExitDirection = BoxBreakExitDirection.NONE


class ConsolidationBoxFactory:
    def __init__(
        self,
        name: str,
        consolidation_box_min_minutes: int,
        consolidation_boxes_max_overlap_threshold: float,
        consolidation_box_post_minimum_minutes_tolerance_coeficient: float,
        accumulated_candle_length_for_consolidation_box_exit_evaluation: int,  # minutes
        minutes_to_consider_a_hod_or_lod_after_close: int,
        minutes_to_consider_a_hod_or_lod_before_close: int,
        allow_consolidation_boxes_to_exist_on_opening_range_area: bool,
    ) -> None:
        self.name: str = name
        self.closed_consolidation_boxes: list[ConsolidationBox] = []
        self.open_consolidation_box: ConsolidationBox | None = None
        self.consolidation_box_min_minutes: int = consolidation_box_min_minutes
        self.consolidation_box_max_thickness: float | None = None
        self.consolidation_box_max_thickness_last_set_at: date | None = None
        self.consolidation_boxes_max_overlap_threshold: float = (
            consolidation_boxes_max_overlap_threshold
        )
        self.consolidation_box_post_minimum_minutes_tolerance_coeficient: float = (
            consolidation_box_post_minimum_minutes_tolerance_coeficient
        )
        self.minutes_to_consider_a_hod_or_lod_after_close: int = (
            minutes_to_consider_a_hod_or_lod_after_close
        )
        self.minutes_to_consider_a_hod_or_lod_before_close: int = (
            minutes_to_consider_a_hod_or_lod_before_close
        )
        self.allow_consolidation_boxes_to_exist_on_opening_range_area: bool = (
            allow_consolidation_boxes_to_exist_on_opening_range_area
        )

    def update_max_thickness(self, new_max_thickness: float) -> None:
        self.consolidation_box_max_thickness = new_max_thickness

    def update_consolidation_boxes_according_to_closing_hod_or_lod(
        self,
        current_datetime: datetime,
        closing_hod: list[datetime],
        closing_lod: list[datetime],
    ) -> None:
        is_closing_hod = current_datetime in closing_hod
        is_closing_lod = current_datetime in closing_lod

        for closed_box in self.closed_consolidation_boxes:
            if (
                current_datetime
                - timedelta(minutes=self.minutes_to_consider_a_hod_or_lod_after_close)
                <= closed_box.end_moment
            ):
                if is_closing_hod:
                    closed_box.had_a_hod_after_x_minutes = True
                if is_closing_lod:
                    closed_box.had_a_lod_after_x_minutes = True


    def erase_all_consolidation_boxes(self) -> None:
        self.open_consolidation_box = None
        self.closed_consolidation_boxes = []

    def set_had_hod_or_lod_before_x_minutes_attribute(
        self,
        current_datetime: datetime,
        closing_hod: list[datetime],
        closing_lod: list[datetime],
    ) -> None:
        assert self.open_consolidation_box

        for i in range(self.minutes_to_consider_a_hod_or_lod_before_close + 1):
            this_moment = current_datetime - timedelta(minutes=i)
            if not self.open_consolidation_box.had_a_hod_before_x_minutes:
                self.open_consolidation_box.had_a_hod_before_x_minutes = (
                    this_moment in closing_hod
                )
            if not self.open_consolidation_box.had_a_lod_before_x_minutes:
                self.open_consolidation_box.had_a_lod_before_x_minutes = (
                    this_moment in closing_lod
                )

    def set_breaking_type(
            self,
            opening_range: OpeningRange,
            current_candle: MyCandle,
            is_at_market_close: bool,
            considering_closing_candle_value_instead_of_highs_and_lows: bool,
            all_daily_data: dict[datetime, MyCandle],
        ) -> None:
            box = self.open_consolidation_box
            if box is None:
                return

            if is_at_market_close:
                box.box_break_inside = BoxBreakInsideClassification.MARKET_CLOSE
            else:
                box_midpoint = (box.high + box.low) / 2
                box_range = box.high - box.low
                top_third = box.low + (3 / 5) * box_range
                bottom_third = box.low + (2 / 5) * box_range

                closing_moments = [
                    box.start_moment + timedelta(minutes=i)
                    for i in range(int((box.end_moment - box.start_moment).total_seconds() / 60) + 1)
                ]
                above_count = 0
                below_count = 0
                for moment in closing_moments:
                    if moment in all_daily_data:
                        close = all_daily_data[moment].close
                        if close > box_midpoint:
                            above_count += 1
                        elif close < box_midpoint:
                            below_count += 1

                total_count = above_count + below_count
                proportion_above = above_count / total_count if total_count > 0 else 0
                proportion_below = below_count / total_count if total_count > 0 else 0

                breakout_value = current_candle.close if considering_closing_candle_value_instead_of_highs_and_lows else current_candle.low
                broke_up = breakout_value >= box.high
                broke_down = breakout_value <= box.low

                last_n = max(1, len(closing_moments) // 3)
                compression_moments = closing_moments[-last_n:]
                tolerance = box_range * 0.25

                all_in_bottom_third = all(
                    moment in all_daily_data and
                    all_daily_data[moment].high <= bottom_third + tolerance and
                    all_daily_data[moment].low >= box.low - tolerance
                    for moment in compression_moments
                )
                all_in_top_third = all(
                    moment in all_daily_data and
                    all_daily_data[moment].low >= top_third - tolerance and
                    all_daily_data[moment].high <= box.high + tolerance
                    for moment in compression_moments
                )

                top_25_threshold = box.low + 0.65 * box_range
                last_5_mins = [box.end_moment - timedelta(minutes=i) for i in range(4, 0, -1)]
                lowest_low_last_5 = min(
                    (all_daily_data[m].low for m in last_5_mins if m in all_daily_data),
                    default=None
                )
                compressed_in_top_25 = (
                    lowest_low_last_5 is not None and lowest_low_last_5 >= top_25_threshold
                )

                bottom_25_threshold = box.low + 0.35 * box_range
                highest_high_last_5 = max(
                    (all_daily_data[m].high for m in last_5_mins if m in all_daily_data),
                    default=None
                )
                compressed_in_bottom_25 = (
                    highest_high_last_5 is not None and highest_high_last_5 <= bottom_25_threshold
                )

                box.broke_due_to_last_5_min_top25 = compressed_in_top_25
                box.broke_due_to_last_5_min_bottom25 = compressed_in_bottom_25

                if broke_up and breakout_value > box.low:
                    if proportion_above >= 0.5 or all_in_top_third or compressed_in_top_25:
                        box.box_break_inside = BoxBreakInsideClassification.UP
                    else:
                        box.box_break_inside = None
                elif broke_down and breakout_value < box.high:
                    if proportion_below >= 0.7 or all_in_bottom_third or compressed_in_bottom_25:
                        box.box_break_inside = BoxBreakInsideClassification.DOWN
                    else:
                        box.box_break_inside = None
                else:
                    box.box_break_inside = None

                if box.box_break_inside == BoxBreakInsideClassification.UP:
                    box.exit_classification = BoxBreakExitDirection.UP
                elif box.box_break_inside == BoxBreakInsideClassification.DOWN:
                    box.exit_classification = BoxBreakExitDirection.DOWN
                else:
                    box.exit_classification = BoxBreakExitDirection.NONE

                print(f"[BREAK TYPE] Box {box.unique_id} | breakout: {breakout_value:.2f}, low: {box.low:.2f}, high: {box.high:.2f}")
                print(f"[BREAK TYPE] broke_up={broke_up}, broke_down={broke_down}, proportion_below={proportion_below:.2f}")
                print(f"[BREAK TYPE] exit_classification: {box.exit_classification.name}")

            if (
                box.low >= opening_range.low and box.high <= opening_range.high
            ):
                box.box_break_according_to_opening_range = BoxBreakOpeningRangeClassification.INSIDE
            elif (
                box.high >= opening_range.high and box.low <= opening_range.low
            ):
                box.box_break_according_to_opening_range = BoxBreakOpeningRangeClassification.ACROSS
            elif box.low <= opening_range.low:
                box.box_break_according_to_opening_range = BoxBreakOpeningRangeClassification.BELLOW
            elif box.high >= opening_range.high:
                box.box_break_according_to_opening_range = BoxBreakOpeningRangeClassification.ABOVE
            else:
                raise ValueError("Unexpected box vs opening range relationship")

    def update_consolidation_boxes(
        self,
        current_datetime: datetime,
        current_candle: MyCandle,
        closing_hod: dict[datetime],
        closing_lod: dict[datetime],
        this_day_opening_range: OpeningRange | None,
        meta_opening_range: OpeningRange | None,
        all_daily_data: dict[datetime, MyCandle],
        last_market_open_moment: datetime,
        meta_opening_ranges_window_minutes: int,
        considering_closing_candle_value_instead_of_highs_and_lows: bool,
        is_at_market_close: bool,
    ) -> None:

        if self.consolidation_box_max_thickness is None:
            return

        if this_day_opening_range is None:
            return

        if self.open_consolidation_box is not None:

            needs_to_close_the_box = False

            if (
                is_candle_inside_opening_range(
                    current_candle,
                    considering_closing_candle_value_instead_of_highs_and_lows,
                    meta_opening_range,
                )
                and not self.allow_consolidation_boxes_to_exist_on_opening_range_area
            ):
                needs_to_close_the_box = True
            else:
                expected_next_minute = self.open_consolidation_box.end_moment + timedelta(minutes=1)

                if current_datetime != expected_next_minute:
                    if current_datetime <= self.open_consolidation_box.end_moment:
                        return  # Already processed or behind — skip
                    else:
                        if self.logger:
                            self.logger.warning(
                                f"[LIVE GUARD] Unexpected datetime gap at {current_datetime}, "
                                f"expected {expected_next_minute} — closing box defensively."
                            )
                        needs_to_close_the_box = True
                else:
                    all_candles = []
                    index_of_creation: int | None = None

                    for i in range(
                        int((current_datetime - self.open_consolidation_box.start_moment).total_seconds() / 60) + 1
                    ):
                        iterative_moment = self.open_consolidation_box.start_moment + timedelta(minutes=i)
                        candle = all_daily_data.get(iterative_moment)
                        if candle is None:
                            if self.logger:
                                self.logger.warning(f"Missing candle for {iterative_moment} during box update.")
                            needs_to_close_the_box = True
                            break
                        all_candles.append(candle)
                        if iterative_moment == self.open_consolidation_box.moment_of_creation:
                            assert index_of_creation is None
                            index_of_creation = i

                    if not needs_to_close_the_box and index_of_creation is not None:
                        it_qualifies, new_highest_value, new_lowest_value = qualifies_as_consolidation_box(
                            how_many_minutes_of_data_considered=len(all_candles),
                            min_minutes_period=self.consolidation_box_min_minutes,
                            consolidation_box_max_thickness=self.consolidation_box_max_thickness,
                            consolidation_box_threshold_tolerance_per_minute=self.consolidation_box_post_minimum_minutes_tolerance_coeficient,
                            all_candles=all_candles,
                            considering_closing_candle_value_instead_of_highs_and_lows=considering_closing_candle_value_instead_of_highs_and_lows,
                            index_of_creation=index_of_creation,
                        )

                        if it_qualifies:
                            assert new_highest_value
                            assert new_lowest_value
                            self.open_consolidation_box.high = new_highest_value
                            self.open_consolidation_box.low = new_lowest_value
                            self.open_consolidation_box.end_moment = current_datetime
                        else:
                            needs_to_close_the_box = True
                    else:
                        needs_to_close_the_box = True

            if needs_to_close_the_box:
                box = self.open_consolidation_box  # store reference BEFORE setting to None

                self.set_breaking_type(
                    opening_range=meta_opening_range,
                    current_candle=current_candle,
                    is_at_market_close=is_at_market_close,
                    considering_closing_candle_value_instead_of_highs_and_lows=considering_closing_candle_value_instead_of_highs_and_lows,
                    all_daily_data=all_daily_data,
                )

                self.set_had_hod_or_lod_before_x_minutes_attribute(
                    current_datetime=current_datetime,
                    closing_hod=closing_hod,
                    closing_lod=closing_lod,
                )

                # ➕ Midpoint fallback logic (after set_breaking_type)
                box_midpoint = (box.high + box.low) / 2
                closing_candle_moments = [
                    box.start_moment + timedelta(minutes=i)
                    for i in range(int((box.end_moment - box.start_moment).total_seconds() // 60) + 1)
                ]
                above_count = 0
                below_count = 0
                for moment in closing_candle_moments:
                    if moment in all_daily_data:
                        candle = all_daily_data[moment]
                        if candle.close > box_midpoint:
                            above_count += 1
                        elif candle.close < box_midpoint:
                            below_count += 1
                total_count = above_count + below_count
                if total_count > 0:
                    if above_count / total_count >= 0.99:
                        box.box_break_inside = BoxBreakInsideClassification.UP
                    elif below_count / total_count >= 0.99:
                        box.box_break_inside = BoxBreakInsideClassification.DOWN

                self.closed_consolidation_boxes.append(box)
                self.open_consolidation_box = None

        else:
            found_consolidation_box: ConsolidationBox | None = (
                get_most_recent_consolidation_box_from_now_to_the_past(
                    current_moment=current_datetime,
                    all_daily_data=all_daily_data,
                    opening_range=meta_opening_range,
                    opening_range_end_time=last_market_open_moment + timedelta(minutes=meta_opening_ranges_window_minutes),
                    min_minutes_period=self.consolidation_box_min_minutes,
                    consolidation_box_max_thickness=self.consolidation_box_max_thickness,
                    consolidation_box_threshold_tolerance_per_minute=self.consolidation_box_post_minimum_minutes_tolerance_coeficient,
                    considering_closing_candle_value_instead_of_highs_and_lows=considering_closing_candle_value_instead_of_highs_and_lows,
                    allow_consolidation_boxes_to_exist_on_opening_range_area=self.allow_consolidation_boxes_to_exist_on_opening_range_area,
                )
            )

            if found_consolidation_box is None:
                return

            if self.closed_consolidation_boxes:
                last_closed_box = self.closed_consolidation_boxes[-1]

                if not allowed_regarding_overlap_of_consolidation_boxes(
                    last_closed_box,
                    found_consolidation_box,
                    self.consolidation_boxes_max_overlap_threshold,
                ):
                    return

            self.open_consolidation_box = found_consolidation_box

    def close_all_open_consolidation_boxes(  # ← Paste here
            self,
            all_daily_data: dict[datetime, MyCandle],
            closing_hod: dict[datetime],
            closing_lod: dict[datetime],
            current_moment: datetime,
            meta_opening_range: OpeningRange,
            considering_closing_candle_value_instead_of_highs_and_lows: bool,
            is_at_market_close: bool = False,
        ) -> None:
            if self.open_consolidation_box is not None:
                self.set_breaking_type(
                    opening_range=meta_opening_range,
                    current_candle=all_daily_data[current_moment],
                    is_at_market_close=is_at_market_close,
                    considering_closing_candle_value_instead_of_highs_and_lows=considering_closing_candle_value_instead_of_highs_and_lows,
                    all_daily_data=all_daily_data,
                )

                self.set_had_hod_or_lod_before_x_minutes_attribute(
                    current_datetime=current_moment,
                    closing_hod=closing_hod,
                    closing_lod=closing_lod,
                )

                self.closed_consolidation_boxes.append(self.open_consolidation_box)
                self.open_consolidation_box = None


def qualifies_as_consolidation_box(
    how_many_minutes_of_data_considered: int,
    min_minutes_period: int,
    consolidation_box_max_thickness: float,
    consolidation_box_threshold_tolerance_per_minute: float,
    all_candles: list[MyCandle],
    considering_closing_candle_value_instead_of_highs_and_lows: bool,
    index_of_creation: int | None = None,
) -> tuple[bool, float | None, float | None]:
    if how_many_minutes_of_data_considered < min_minutes_period:
        return False, None, None

    current_max_thickness = deepcopy(consolidation_box_max_thickness)

    highest_value: float | None = None
    # highest_moment: datetime | None
    lowest_value: float | None = None
    # lowest_moment: datetime | None

    for index, candle in enumerate(all_candles):

        if index_of_creation and index >= index_of_creation:
            # increase the thickness overtime
            current_max_thickness += (
                consolidation_box_threshold_tolerance_per_minute
                * consolidation_box_max_thickness
            )

        high_value: float = (
            candle.close
            if considering_closing_candle_value_instead_of_highs_and_lows
            else candle.high
        )
        low_value: float = (
            candle.close
            if considering_closing_candle_value_instead_of_highs_and_lows
            else candle.low
        )

        if highest_value is None or high_value > highest_value:
            highest_value = high_value
            # highest_moment = deepcopy(candle_moment)
        if lowest_value is None or low_value < lowest_value:
            lowest_value = low_value
            # lowest_moment = deepcopy(candle_moment)

        if (highest_value - lowest_value) / lowest_value >= current_max_thickness:
            return False, None, None

    return True, highest_value, lowest_value


def allowed_regarding_overlap_of_consolidation_boxes(
    box_a: ConsolidationBox, box_b: ConsolidationBox, threshold: float
) -> float:
    assert box_a.start_moment < box_b.start_moment
    assert box_a.end_moment < box_b.end_moment

    if box_a.end_moment < box_b.start_moment:
        return True

    total_seconds_of_box_a = (box_a.end_moment - box_a.start_moment).total_seconds()
    total_seconds_of_box_a_overlaped_area = (
        box_a.end_moment - box_b.start_moment
    ).total_seconds()

    is_too_much_of_box_a_covered = (
        total_seconds_of_box_a_overlaped_area / total_seconds_of_box_a > threshold
    )

    total_seconds_of_box_b = (box_b.end_moment - box_b.start_moment).total_seconds()
    total_seconds_of_box_b_non_overlaped_area = (
        box_b.end_moment - box_a.end_moment
    ).total_seconds()

    is_box_b_sufficiently_uncovered = (
        total_seconds_of_box_b_non_overlaped_area / total_seconds_of_box_b > threshold
    )

    return (not is_too_much_of_box_a_covered) and is_box_b_sufficiently_uncovered


def get_most_recent_consolidation_box_from_now_to_the_past(
    current_moment: datetime,
    all_daily_data: dict[datetime, MyCandle],
    opening_range: OpeningRange | None,
    opening_range_end_time: datetime,
    min_minutes_period: int,
    consolidation_box_max_thickness: float,
    consolidation_box_threshold_tolerance_per_minute: float,
    considering_closing_candle_value_instead_of_highs_and_lows: bool,
    allow_consolidation_boxes_to_exist_on_opening_range_area: bool,
) -> ConsolidationBox | None:

    if opening_range is None:
        # no open range yet
        return None

    minutes_passed_since_opening: int = int(
        (current_moment - opening_range_end_time).total_seconds() / 60
    )

    if minutes_passed_since_opening < min_minutes_period:
        # it hasnt passed enough time since the end of the opening range
        return None

    # now we get all the data from the last (min_minutes_period)
    all_data_since_end_of_opening_range: list[tuple[datetime, MyCandle]] = []
    for i in range(minutes_passed_since_opening):
        key_moment = current_moment - timedelta(minutes=i)
        all_data_since_end_of_opening_range.append(
            (key_moment, all_daily_data[key_moment])
        )

    current_open_consolidation_box: ConsolidationBox | None = None

    end_moment = all_data_since_end_of_opening_range[0][0]

    current_set_of_candles: list[MyCandle] = []

    # we are iterating from the most recent candle to the oldest candle
    for current_index, candle_tuple in enumerate(all_data_since_end_of_opening_range):
        current_candle_moment, current_candle = candle_tuple
        # check if any candle is intercepting the Opening Range

        current_set_of_candles.insert(0, current_candle)

        if (
            is_candle_inside_opening_range(
                current_candle,
                considering_closing_candle_value_instead_of_highs_and_lows,
                opening_range,
            )
            and not allow_consolidation_boxes_to_exist_on_opening_range_area
        ):
            break

        does_it_qualify, highest_value, lowest_value = qualifies_as_consolidation_box(
            current_index + 1,
            min_minutes_period,
            consolidation_box_max_thickness,
            consolidation_box_threshold_tolerance_per_minute,
            current_set_of_candles,
            considering_closing_candle_value_instead_of_highs_and_lows=considering_closing_candle_value_instead_of_highs_and_lows,
        )

        if does_it_qualify:
            assert highest_value
            assert lowest_value

            current_open_consolidation_box = ConsolidationBox(
                unique_id=str(uuid.uuid4()),
                start_moment=current_candle_moment,
                moment_of_creation=end_moment,
                end_moment=end_moment,
                high=highest_value,
                high_moment=None,  # TODO: not important, but maybe re-add this
                low=lowest_value,
                low_moment=None,  # TODO: not important, but maybe re-add this
            )
        else:
            if current_open_consolidation_box:
                return current_open_consolidation_box

    return current_open_consolidation_box


def is_candle_inside_opening_range(
    current_candle: MyCandle,
    considering_closing_candle_value_instead_of_highs_and_lows: bool,
    opening_range: OpeningRange,
) -> bool:

    if considering_closing_candle_value_instead_of_highs_and_lows:
        return (
            current_candle.close >= opening_range.low
            and current_candle.close <= opening_range.high
        )

    if (
        current_candle.high >= opening_range.low
        and current_candle.high <= opening_range.high
    ):
        return True

    if (
        current_candle.low >= opening_range.low
        and current_candle.low <= opening_range.high
    ):
        return True

    if (
        current_candle.high >= opening_range.high
        and current_candle.low <= opening_range.low
    ):
        return True

    return False
