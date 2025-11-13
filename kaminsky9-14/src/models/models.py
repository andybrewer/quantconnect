from AlgorithmImports import *
from datetime import datetime, timedelta
from pydantic import BaseModel

class MyCandle(BaseModel):
    high: float
    low: float
    open: float
    close: float
    price: float
    volume: float
    moment: datetime

class TakeProfitArea:
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        initial_value: float,
        distance: float,
        min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer: int,
        number_of_layers: int,
        incremental_tunnel_closing_coeficient: float,
        exit_worsening_threshold_percent: float = 50,
        min_sequence_length_minutes: int = 1000,
        max_sequences_before_exit: int = 1000,
        exit_threshold_percent: float = 50,
        max_consecutive_minutes_outside_profit_area: int = 1000,
        min_loss_threshold_percent_when_outside_profit_area: float = -0.20,
        traded_long: bool = True,
        max_loss_threshold_percent: float = None,
        logger=None,
    ) -> None:
        assert start_time <= end_time
        self.start_time = start_time
        self.end_time = end_time
        self.traded_long = traded_long
        self.distance = distance
        self.incremental_tunnel_closing_coeficient = incremental_tunnel_closing_coeficient
        self.number_of_layers = number_of_layers
        self.min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer = min_consecutive_candles_to_perform_exit_when_entering_a_worse_layer
        self.exit_worsening_threshold_percent = exit_worsening_threshold_percent
        self.exit_threshold_percent = exit_threshold_percent
        self.min_sequence_length_minutes = min_sequence_length_minutes
        self.max_sequences_before_exit = max_sequences_before_exit
        self.max_consecutive_minutes_outside_profit_area = max_consecutive_minutes_outside_profit_area
        self.min_loss_threshold_percent_when_outside_profit_area = min_loss_threshold_percent_when_outside_profit_area
        self.max_loss_threshold_percent = max_loss_threshold_percent
        self.logger = logger
        self.entry_price = initial_value
        self.take_profit_area_at_moment = {}
        self._total_negative_closes = 0  # ⬅️ ADDED
        self.reset_state_for_new_trade()

        layer_values = [initial_value]
        for i in range(number_of_layers - 1):
            delta = distance / (2 + i)
            new_val = delta + layer_values[-1] if traded_long else layer_values[-1] - delta
            layer_values.append(new_val)

        current_moment = start_time
        while current_moment <= end_time:
            shifted_layer_values = []
            minutes_elapsed = (current_moment - start_time).total_seconds() / 60.0
            for base_value in layer_values:
                shift = distance * incremental_tunnel_closing_coeficient * minutes_elapsed * (-1 if traded_long else 1)
                shifted_value = base_value + shift
                shifted_layer_values.append(shifted_value)
            self.take_profit_area_at_moment[current_moment] = shifted_layer_values
            current_moment += timedelta(minutes=1)

    def reset_state_for_new_trade(self):
        self.last_layer_present_index = None
        self.last_layer_present_consecutive_candles = 0
        self.consecutive_minutes_outside_profit_area = 0
        self.consecutive_minutes_inside_profit_area = 0
        self.has_ever_entered_profit_area = False
        self.best_close_in_profit_area = None
        self.in_profit_sequence = False
        self.sequence_start_time = None
        self.inside_profit_area_sequences = 0
        self.exit_worsening_check_active = False
        self.waiting_for_worsening_confirmation = False
        self.worsening_confirmation_check_time = None
        self.worst_price_during_worsening_check = None
        self.final_exit_trigger_active = False
        self.final_exit_reference_price = None
        self.max_profit_percent_seen_since_entry = 0
        self.cumulative_profitable_closes = 0
        self.consecutive_losing_closes = 0
        self._first_15_closes = []
        self._checked_first_15_closes = False
        self.worst_drawdown_percent_seen = 0
        self._first_20_closes = []
        self._checked_first_20_closes = False
        self._consecutive_profit_minutes = 0
        self._has_reached_140_consecutive_profit = False
        self._negative_close_counter = 0
        self._total_negative_closes = 0  # ⬅️ ADDED
        self.best_price_during_trade = self.entry_price

    def _current_profit_percent(self, price: float) -> float:
        return ((price - self.entry_price) / self.entry_price * 100.0) if self.traded_long else ((self.entry_price - price) / self.entry_price * 100.0)

    def what_layer_is_candle_in(self, candle: MyCandle, candle_moment: datetime, only_considering_closing_values=False):
        if candle_moment not in self.take_profit_area_at_moment:
            return None
        val = candle.close if only_considering_closing_values else (candle.low if self.traded_long else candle.high)
        valid_layer_index = None
        for layer_index in range(self.number_of_layers):
            level = self.take_profit_area_at_moment[candle_moment][layer_index]
            if self.traded_long and val > level:
                valid_layer_index = layer_index
                continue
            if not self.traded_long and val < level:
                valid_layer_index = layer_index
                continue
            break
        return valid_layer_index

    def should_exit_and_reason(self, candle: MyCandle, candle_moment: datetime, only_considering_closing_values=False):
        current_profit = self._current_profit_percent(candle.close)

        if self.traded_long:
            if candle.high > self.best_price_during_trade:
                self.best_price_during_trade = candle.high
        else:
            if candle.low < self.best_price_during_trade:
                self.best_price_during_trade = candle.low

        if self.traded_long:
            profit_from_best_price = ((self.best_price_during_trade - self.entry_price) / self.entry_price) * 100.0
        else:
            profit_from_best_price = ((self.entry_price - self.best_price_during_trade) / self.entry_price) * 100.0

        if profit_from_best_price > self.max_profit_percent_seen_since_entry:
            self.max_profit_percent_seen_since_entry = profit_from_best_price

        if (self.max_profit_percent_seen_since_entry - current_profit > 0.75):
            return True, "trailing_profit_giveback_exceeded_1.25"

        if self.entry_price is not None and self.max_loss_threshold_percent is not None:
            stop_price = (
                self.entry_price * (1 - self.max_loss_threshold_percent / 100)
                if self.traded_long else
                self.entry_price * (1 + self.max_loss_threshold_percent / 100)
            )
            if (self.traded_long and candle.close <= stop_price) or (not self.traded_long and candle.close >= stop_price):
                return True, "max_loss_threshold_hit"

        dynamic_exit, reason = self.should_exit_due_to_dynamic_loss(candle, candle_moment)
        if dynamic_exit:
            return True, reason

        exit_avg_close, reason = self.should_exit_due_to_avg_close_after_20min(candle, candle_moment)
        if exit_avg_close:
            return True, reason

        exit_140min_flip, reason = self.should_exit_due_to_140min_in_profit_then_flip_negative(candle, candle_moment)
        if exit_140min_flip:
            return True, reason

        exit_200_negative_closes, reason = self.should_exit_due_to_200_negative_closes(candle, candle_moment)
        if exit_200_negative_closes:
            return True, reason

        return False, None

    def is_candle_in_take_profit_area_long_enough(self, candle: MyCandle, candle_moment: datetime, only_considering_closing_values=False):
        result, _ = self.should_exit_and_reason(candle, candle_moment, only_considering_closing_values)
        return result

    def should_exit_due_to_dynamic_loss(self, candle: MyCandle, candle_moment: datetime):
        minutes_held = int((candle_moment - self.start_time).total_seconds() / 60.0)
        current_profit = self._current_profit_percent(candle.close)

        if current_profit < 0:
            if current_profit < self.worst_drawdown_percent_seen:
                self.worst_drawdown_percent_seen = current_profit

        if minutes_held > 80:
            if current_profit < -0.45:
                if abs(self.worst_drawdown_percent_seen) > self.max_profit_percent_seen_since_entry:
                    return True, "dynamic_loss_exceeds_max_profit_after_45min"

        if (
            not self.traded_long
            and minutes_held >= 75
            and self.max_profit_percent_seen_since_entry < 0.1
            and current_profit <= -0.35
        ):
            return True, "short_flat_then_drop_exit"

        return False, None

    def should_exit_due_to_avg_close_after_20min(self, candle: MyCandle, candle_moment: datetime):
        current_profit = self._current_profit_percent(candle.close)

        if len(self._first_20_closes) < 20:
            self._first_20_closes.append(current_profit)

        if len(self._first_20_closes) == 20:
            avg_profit = sum(self._first_20_closes) / 20.0

            if not self._checked_first_20_closes:
                if avg_profit <= -0.05:
                    self._checked_first_20_closes = True

            if self._checked_first_20_closes:
                exit_threshold = -0.25 if self.traded_long else -0.2
                if current_profit <= exit_threshold:
                    return True, f"avg_close_after_20min_flagged_and_current_profit_under_{exit_threshold:.2f}%"

        return False, None

    def should_exit_due_to_140min_in_profit_then_flip_negative(self, candle: MyCandle, candle_moment: datetime):
        current_profit = self._current_profit_percent(candle.close)

        if current_profit > 0:
            self._consecutive_profit_minutes += 1
        else:
            if self._has_reached_140_consecutive_profit and current_profit < 0:
                return True, "in_profit_140min_then_flipped_negative"
            self._consecutive_profit_minutes = 0

        if self._consecutive_profit_minutes >= 95:
            self._has_reached_140_consecutive_profit = True

        return False, None

    def should_exit_due_to_200_negative_closes(self, candle: MyCandle, candle_moment: datetime):
        current_profit = self._current_profit_percent(candle.close)

        if current_profit < 0:
            self._total_negative_closes += 1
        if self._total_negative_closes >= 140 and current_profit <= -0.1:
            return True, "200_total_negative_closes_and_trade_is_more_than_-0.1_percent"

        return False, None
