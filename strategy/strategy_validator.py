"""
Strategy verification utilities.

Provides `StrategyValidator`, which checks a StrategyBaseClass implementation
for two classes of problems before it is trusted for backtesting/live trading:

1. Output schema validity — entries/exits/close_data/open_data must be
   boolean-aligned DataFrames sharing the same index and columns, with no
   NaNs and no bar where entries and exits are both True for the same symbol.

2. Look-ahead bias (future data leaking into past signals) — the strategy is
   run once on the full OHLCV history, then re-run on progressively truncated
   prefixes of that same history. A correct strategy's signal at time T can
   only depend on data up to and including T, so the truncated run's signals
   must exactly match the full run's signals over the overlapping period. Any
   mismatch means the strategy peeked at data that would not yet have existed
   at that point in time.

See discussion: https://github.com/himanshu2406/Algo.Py/issues/9
"""
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from strategy.strategy_builder import StrategyBaseClass


@dataclass
class ValidationReport:
    """Result of running StrategyValidator.validate()."""

    passed: bool
    schema_issues: List[str] = field(default_factory=list)
    lookahead_issues: List[str] = field(default_factory=list)

    @property
    def issues(self) -> List[str]:
        return self.schema_issues + self.lookahead_issues

    def __str__(self) -> str:
        if self.passed:
            return "ValidationReport: PASSED"
        lines = ["ValidationReport: FAILED"]
        lines += [f"  [schema] {msg}" for msg in self.schema_issues]
        lines += [f"  [lookahead] {msg}" for msg in self.lookahead_issues]
        return "\n".join(lines)


class StrategyValidator:
    """Validates a StrategyBaseClass instance's output schema and checks for
    look-ahead bias (data leaking from the future into past signals)."""

    def validate_output_schema(
        self,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        close_data: pd.DataFrame,
        open_data: pd.DataFrame,
    ) -> List[str]:
        """Structural checks on a strategy's raw output. Returns a list of
        human-readable issue descriptions; empty list means all checks passed."""
        issues: List[str] = []

        frames = {
            "entries": entries,
            "exits": exits,
            "close_data": close_data,
            "open_data": open_data,
        }
        for name, df in frames.items():
            if not isinstance(df, pd.DataFrame):
                issues.append(f"'{name}' is not a pandas DataFrame (got {type(df).__name__}).")
        if issues:
            # Can't safely compare shapes/indexes if a frame isn't even a DataFrame.
            return issues

        reference_index = close_data.index
        reference_columns = set(close_data.columns)
        for name, df in frames.items():
            if not df.index.equals(reference_index):
                issues.append(f"'{name}' index does not match 'close_data' index.")
            if set(df.columns) != reference_columns:
                issues.append(f"'{name}' columns {set(df.columns)} do not match 'close_data' columns {reference_columns}.")

        if not reference_index.is_monotonic_increasing:
            issues.append("close_data index is not sorted in chronological (ascending) order.")

        for name in ("entries", "exits"):
            df = frames[name]
            if df.isna().any().any():
                issues.append(f"'{name}' contains NaN values (should be filled with False).")
            elif df.dtypes.apply(lambda dt: dt != bool).any():
                issues.append(f"'{name}' contains non-boolean columns.")

        if entries.index.equals(exits.index) and set(entries.columns) == set(exits.columns):
            conflicting = entries & exits
            if conflicting.to_numpy().any():
                bad_cells = conflicting.stack()
                bad_cells = bad_cells[bad_cells].index.tolist()
                issues.append(
                    f"entries and exits are both True on the same bar for {len(bad_cells)} "
                    f"(timestamp, symbol) cell(s), e.g. {bad_cells[:3]}."
                )

        return issues

    def detect_lookahead_bias(
        self,
        strategy: StrategyBaseClass,
        ohlcv_data: Dict[str, pd.DataFrame],
        checkpoints: Optional[List[int]] = None,
        timestamp_column: str = "timestamp",
    ) -> List[str]:
        """Runs `strategy` on the full history, then on truncated prefixes of
        it, and checks that signals over the overlapping period are identical.

        :param strategy: an instance of a StrategyBaseClass subclass
        :param ohlcv_data: dict of {symbol: OHLCV DataFrame}, as passed to strategy.run()
        :param checkpoints: bar indices at which to truncate the data and re-run;
            defaults to the 50%, 75% and 90% marks of the shortest symbol's history
        :param timestamp_column: name of the timestamp column, if present as a
            column rather than already set as the index
        :return: list of human-readable issue descriptions; empty means no leak detected
        """
        issues: List[str] = []

        full_entries, full_exits, full_close, _ = strategy.run(ohlcv_data)

        min_length = min(len(df) for df in ohlcv_data.values())
        if checkpoints is None:
            checkpoints = sorted(
                {max(2, int(min_length * frac)) for frac in (0.5, 0.75, 0.9)}
            )

        for cutoff in checkpoints:
            if cutoff >= min_length:
                continue

            truncated_data = {
                symbol: df.iloc[:cutoff].copy() for symbol, df in ohlcv_data.items()
            }
            trunc_entries, trunc_exits, trunc_close, _ = strategy.run(truncated_data)

            common_index = full_entries.index.intersection(trunc_entries.index)
            common_columns = [c for c in full_entries.columns if c in trunc_entries.columns]
            if len(common_index) == 0 or len(common_columns) == 0:
                issues.append(
                    f"Truncated run at cutoff={cutoff} produced no overlapping "
                    f"index/columns with the full run — cannot compare."
                )
                continue

            full_slice = full_entries.loc[common_index, common_columns]
            trunc_slice = trunc_entries.loc[common_index, common_columns]
            mismatch = full_slice != trunc_slice
            if mismatch.to_numpy().any():
                bad_cells = mismatch.stack()
                bad_cells = bad_cells[bad_cells].index.tolist()
                issues.append(
                    f"entries differ between full-history and truncated (cutoff={cutoff}) "
                    f"runs at {len(bad_cells)} cell(s), e.g. {bad_cells[:3]} — "
                    f"likely look-ahead bias (signal depends on future data)."
                )

            full_exit_slice = full_exits.loc[common_index, common_columns]
            trunc_exit_slice = trunc_exits.loc[common_index, common_columns]
            exit_mismatch = full_exit_slice != trunc_exit_slice
            if exit_mismatch.to_numpy().any():
                bad_cells = exit_mismatch.stack()
                bad_cells = bad_cells[bad_cells].index.tolist()
                issues.append(
                    f"exits differ between full-history and truncated (cutoff={cutoff}) "
                    f"runs at {len(bad_cells)} cell(s), e.g. {bad_cells[:3]} — "
                    f"likely look-ahead bias (signal depends on future data)."
                )

        return issues

    def validate(
        self,
        strategy: StrategyBaseClass,
        ohlcv_data: Dict[str, pd.DataFrame],
        checkpoints: Optional[List[int]] = None,
    ) -> ValidationReport:
        """Runs both the output-schema and look-ahead-bias checks and returns
        a combined ValidationReport."""
        entries, exits, close_data, open_data = strategy.run(ohlcv_data)
        schema_issues = self.validate_output_schema(entries, exits, close_data, open_data)
        lookahead_issues = self.detect_lookahead_bias(strategy, ohlcv_data, checkpoints=checkpoints)
        return ValidationReport(
            passed=not (schema_issues or lookahead_issues),
            schema_issues=schema_issues,
            lookahead_issues=lookahead_issues,
        )
