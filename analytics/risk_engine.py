"""
analytics/risk_engine.py — Quantitative Portfolio Risk Analytics Engine
Calculates institutional-grade risk & performance metrics:
  - Daily returns series & Return Distribution (Skewness, Kurtosis, VaR, CVaR)
  - Annualized Volatility (252 trading days)
  - Maximum Drawdown & Underwater Drawdown Series
  - Sharpe Ratio (with clearly documented RBI 91-day T-Bill risk-free rate)
  - Sortino Ratio (Downside deviation)
  - Calmar Ratio (Return / Max Drawdown)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# =============================================================================
# DOCUMENTED FINANCIAL BENCHMARK ASSUMPTIONS
# =============================================================================
# Risk-Free Rate: 6.5% per annum (Reflects benchmark RBI 91-Day Treasury Bill yield for Indian equities)
RISK_FREE_RATE_ANNUAL = 0.065
TRADING_DAYS_PER_YEAR = 252
INTRADAY_MINUTES_PER_DAY = 375


@dataclass
class PortfolioRiskReport:
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95_pct: float             # Value-at-Risk (95% confidence)
    cvar_95_pct: float            # Conditional VaR / Expected Shortfall (95%)
    win_days_pct: float
    loss_days_pct: float
    skewness: float
    kurtosis: float
    assumptions: Dict[str, Any]
    daily_returns: List[float]
    equity_curve: List[float]
    drawdown_series: List[float]
    return_distribution: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return_pct': round(self.total_return_pct, 2),
            'annualized_return_pct': round(self.annualized_return_pct, 2),
            'annualized_volatility_pct': round(self.annualized_volatility_pct, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'calmar_ratio': round(self.calmar_ratio, 2),
            'var_95_pct': round(self.var_95_pct, 2),
            'cvar_95_pct': round(self.cvar_95_pct, 2),
            'win_days_pct': round(self.win_days_pct, 1),
            'loss_days_pct': round(self.loss_days_pct, 1),
            'skewness': round(self.skewness, 2),
            'kurtosis': round(self.kurtosis, 2),
            'assumptions': self.assumptions,
            'distribution_summary': {
                'bins': self.return_distribution.get('bins', []),
                'counts': self.return_distribution.get('counts', [])
            }
        }


class RiskAnalyticsEngine:
    """
    Computes risk and return metrics for any equity curve or trade return series.
    """

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
        trading_days: int = TRADING_DAYS_PER_YEAR
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def analyze_equity_curve(
        self,
        equity_series: List[float],
        timestamps: Optional[List[Any]] = None,
        is_intraday_minutes: bool = True
    ) -> PortfolioRiskReport:
        """
        Calculates all risk metrics from an equity curve series.
        """
        arr = np.array(equity_series, dtype=float)
        if len(arr) < 2:
            return self._empty_report(arr[0] if len(arr) > 0 else 100000.0)

        initial_equity = arr[0]
        final_equity = arr[-1]
        total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100.0

        # Percentage period-to-period returns
        returns = np.diff(arr) / (arr[:-1] + 1e-10)

        # 1. High-water mark & Drawdown series
        running_max = np.maximum.accumulate(arr)
        drawdowns = (running_max - arr) / (running_max + 1e-10) * 100.0
        max_drawdown_pct = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # 2. Annualization factor
        # If intraday 1-minute candles:
        # Effective days = len(arr) / INTRADAY_MINUTES_PER_DAY
        effective_days = max(1.0, len(arr) / (INTRADAY_MINUTES_PER_DAY if is_intraday_minutes else 1.0))
        years = effective_days / self.trading_days

        if years > 0 and final_equity > 0 and initial_equity > 0:
            ann_return = ((final_equity / initial_equity) ** (1.0 / max(0.01, years)) - 1.0) * 100.0
        else:
            ann_return = total_return_pct

        # Daily / Annualized Volatility
        daily_std = float(np.std(returns)) if len(returns) > 0 else 0.0
        # Scale to annual volatility
        scale_factor = np.sqrt(self.trading_days * (INTRADAY_MINUTES_PER_DAY if is_intraday_minutes else 1.0))
        ann_volatility_pct = float(daily_std * scale_factor * 100.0)

        # 3. Sharpe Ratio
        # (Annualized Return - Risk Free Rate) / Annualized Volatility
        excess_return = (ann_return / 100.0) - self.risk_free_rate
        ann_vol = max(1e-5, ann_volatility_pct / 100.0)
        sharpe = float(excess_return / ann_vol)

        # 4. Sortino Ratio (Downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = float(np.std(downside_returns))
            ann_downside_vol = max(1e-5, downside_std * scale_factor)
            sortino = float(excess_return / ann_downside_vol)
        else:
            sortino = sharpe * 1.5 if sharpe > 0 else 1.0

        # 5. Calmar Ratio
        calmar = (ann_return / max(0.5, max_drawdown_pct)) if max_drawdown_pct > 0 else ann_return

        # 6. VaR & CVaR (95% confidence)
        if len(returns) >= 10:
            var_95_pct = float(abs(np.percentile(returns, 5)) * 100.0)
            tail_losses = returns[returns <= np.percentile(returns, 5)]
            cvar_95_pct = float(abs(np.mean(tail_losses)) * 100.0) if len(tail_losses) > 0 else var_95_pct
        else:
            var_95_pct = 1.0
            cvar_95_pct = 1.5

        # 7. Distribution Shape
        win_days = sum(1 for r in returns if r > 0)
        loss_days = sum(1 for r in returns if r < 0)
        total_p = max(1, win_days + loss_days)
        win_days_pct = (win_days / total_p) * 100.0
        loss_days_pct = (loss_days / total_p) * 100.0

        skewness = float(self._calc_skew(returns))
        kurtosis = float(self._calc_kurt(returns))

        # Histogram bins for UI chart rendering
        hist_counts, bin_edges = np.histogram(returns * 100.0, bins=15)
        bin_labels = [f"{bin_edges[i]:.2f}%" for i in range(len(bin_edges) - 1)]

        assumptions = {
            'risk_free_rate_pct': round(self.risk_free_rate * 100.0, 2),
            'risk_free_benchmark': "RBI 91-Day Treasury Bill Yield (India)",
            'trading_days_per_year': self.trading_days,
            'timeframe': "1-Minute Intraday" if is_intraday_minutes else "Daily",
            'confidence_level_var': "95%"
        }

        return PortfolioRiskReport(
            total_return_pct=total_return_pct,
            annualized_return_pct=ann_return,
            annualized_volatility_pct=ann_volatility_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            var_95_pct=var_95_pct,
            cvar_95_pct=cvar_95_pct,
            win_days_pct=win_days_pct,
            loss_days_pct=loss_days_pct,
            skewness=skewness,
            kurtosis=kurtosis,
            assumptions=assumptions,
            daily_returns=[float(r * 100.0) for r in returns],
            equity_curve=[float(e) for e in arr],
            drawdown_series=[float(d) for d in drawdowns],
            return_distribution={
                'bins': bin_labels,
                'counts': hist_counts.tolist()
            }
        )

    def _calc_skew(self, x: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        std = np.std(x)
        if std == 0:
            return 0.0
        return float(np.mean(((x - np.mean(x)) / std) ** 3))

    def _calc_kurt(self, x: np.ndarray) -> float:
        if len(x) < 4:
            return 3.0
        std = np.std(x)
        if std == 0:
            return 3.0
        return float(np.mean(((x - np.mean(x)) / std) ** 4))

    def _empty_report(self, capital: float) -> PortfolioRiskReport:
        return PortfolioRiskReport(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            annualized_volatility_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            var_95_pct=0.0,
            cvar_95_pct=0.0,
            win_days_pct=0.0,
            loss_days_pct=0.0,
            skewness=0.0,
            kurtosis=3.0,
            assumptions={'risk_free_rate_pct': 6.5, 'trading_days': 252},
            daily_returns=[],
            equity_curve=[capital],
            drawdown_series=[0.0],
            return_distribution={'bins': [], 'counts': []}
        )
