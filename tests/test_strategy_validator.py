import numpy as np
import pandas as pd
import pytest

from strategy.strategy_builder import StrategyBaseClass
from strategy.strategy_validator import StrategyValidator


def _make_ohlcv_data(n_bars: int = 60, symbols=("AAA", "BBB"), seed: int = 7):
    """Builds a small synthetic multi-symbol OHLCV dataset in the shape
    strategy.run() expects: Dict[symbol, DataFrame(index=timestamp,
    columns=[open, high, low, close, volume])]."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="D")
    data = {}
    for i, symbol in enumerate(symbols):
        close = 100 + i * 10 + np.cumsum(rng.normal(loc=0.1, scale=1.0, size=n_bars))
        open_ = close + rng.normal(scale=0.2, size=n_bars)
        high = np.maximum(open_, close) + rng.uniform(0, 0.5, size=n_bars)
        low = np.minimum(open_, close) - rng.uniform(0, 0.5, size=n_bars)
        volume = rng.integers(1_000, 10_000, size=n_bars).astype(float)
        data[symbol] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    return data


class CleanEmaStrategy(StrategyBaseClass):
    """A well-behaved EMA-crossover strategy: every signal at time T only
    uses close prices up to and including T (via ewm + shift(1)), so it
    should pass both the schema and look-ahead checks."""

    def __init__(self, fast: int = 5, slow: int = 15):
        super().__init__(name="Clean EMA Strategy")
        self.fast = fast
        self.slow = slow

    def run(self, ohlcv_data):
        close_data = pd.DataFrame({s: df["close"] for s, df in ohlcv_data.items()})
        open_data = pd.DataFrame({s: df["open"] for s, df in ohlcv_data.items()})

        fast_ema = close_data.ewm(span=self.fast, adjust=False).mean()
        slow_ema = close_data.ewm(span=self.slow, adjust=False).mean()

        entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))

        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits, close_data, open_data


class LeakyStrategy(StrategyBaseClass):
    """A deliberately broken strategy that peeks at tomorrow's close price
    (shift(-1)) to decide today's entry — a classic look-ahead bias bug.
    Used to prove StrategyValidator actually catches this."""

    def __init__(self):
        super().__init__(name="Leaky Strategy")

    def run(self, ohlcv_data):
        close_data = pd.DataFrame({s: df["close"] for s, df in ohlcv_data.items()})
        open_data = pd.DataFrame({s: df["open"] for s, df in ohlcv_data.items()})

        future_close = close_data.shift(-1)  # <-- the bug: tomorrow's price
        entries = (future_close > close_data).fillna(False)
        exits = (future_close < close_data).fillna(False)

        return entries.astype(bool), exits.astype(bool), close_data, open_data


class BadSchemaStrategy(StrategyBaseClass):
    """A strategy whose output violates the schema contract: entries and
    exits are both True on the same bar for the same symbol."""

    def __init__(self):
        super().__init__(name="Bad Schema Strategy")

    def run(self, ohlcv_data):
        close_data = pd.DataFrame({s: df["close"] for s, df in ohlcv_data.items()})
        open_data = pd.DataFrame({s: df["open"] for s, df in ohlcv_data.items()})

        entries = pd.DataFrame(True, index=close_data.index, columns=close_data.columns)
        exits = pd.DataFrame(True, index=close_data.index, columns=close_data.columns)

        return entries, exits, close_data, open_data


class TestValidateOutputSchema:
    def test_clean_strategy_passes_schema(self):
        ohlcv_data = _make_ohlcv_data()
        strategy = CleanEmaStrategy()
        entries, exits, close_data, open_data = strategy.run(ohlcv_data)

        issues = StrategyValidator().validate_output_schema(entries, exits, close_data, open_data)
        assert issues == []

    def test_conflicting_entry_exit_is_flagged(self):
        ohlcv_data = _make_ohlcv_data()
        strategy = BadSchemaStrategy()
        entries, exits, close_data, open_data = strategy.run(ohlcv_data)

        issues = StrategyValidator().validate_output_schema(entries, exits, close_data, open_data)
        assert any("both True on the same bar" in issue for issue in issues)

    def test_mismatched_index_is_flagged(self):
        ohlcv_data = _make_ohlcv_data()
        strategy = CleanEmaStrategy()
        entries, exits, close_data, open_data = strategy.run(ohlcv_data)

        truncated_entries = entries.iloc[:-1]  # drop the last row -> index mismatch

        issues = StrategyValidator().validate_output_schema(
            truncated_entries, exits, close_data, open_data
        )
        assert any("index does not match" in issue for issue in issues)


class TestDetectLookaheadBias:
    def test_clean_strategy_has_no_lookahead_bias(self):
        ohlcv_data = _make_ohlcv_data()
        strategy = CleanEmaStrategy()

        issues = StrategyValidator().detect_lookahead_bias(strategy, ohlcv_data)
        assert issues == []

    def test_leaky_strategy_is_caught(self):
        ohlcv_data = _make_ohlcv_data()
        strategy = LeakyStrategy()

        issues = StrategyValidator().detect_lookahead_bias(strategy, ohlcv_data)
        assert len(issues) > 0
        assert any("look-ahead bias" in issue for issue in issues)


class TestValidateEndToEnd:
    def test_clean_strategy_report_passes(self):
        ohlcv_data = _make_ohlcv_data()
        report = StrategyValidator().validate(CleanEmaStrategy(), ohlcv_data)
        assert report.passed is True
        assert report.issues == []

    def test_leaky_strategy_report_fails(self):
        ohlcv_data = _make_ohlcv_data()
        report = StrategyValidator().validate(LeakyStrategy(), ohlcv_data)
        assert report.passed is False
        assert len(report.lookahead_issues) > 0

    def test_bad_schema_strategy_report_fails(self):
        ohlcv_data = _make_ohlcv_data()
        report = StrategyValidator().validate(BadSchemaStrategy(), ohlcv_data)
        assert report.passed is False
        assert len(report.schema_issues) > 0

    def test_report_str_is_readable(self):
        ohlcv_data = _make_ohlcv_data()
        report = StrategyValidator().validate(BadSchemaStrategy(), ohlcv_data)
        text = str(report)
        assert "FAILED" in text
        assert "[schema]" in text
