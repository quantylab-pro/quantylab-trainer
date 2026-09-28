"""Point-in-time quote features; decision_at must be the real decision time."""
import pandas as pd


def join_quotes(decisions, snapshots, tolerance_seconds=90):
    """No backdating provider timestamps. Missing snapshots remain explicitly missing.

    decisions: code, decision_at (timezone-aware). snapshots: code,
    quote_received_at, quote_fresh, spread_bps, imbalance. All times require zones.
    """
    result = decisions.copy()
    result['_order'] = range(len(result))
    result['decision_at'] = pd.to_datetime(result.decision_at, utc=True)
    if snapshots.empty:
        result['quote_available'] = False
        result['spread_bps'] = float('nan')
        result['imbalance'] = float('nan')
        return result.drop(columns='_order')
    quotes = snapshots[snapshots.quote_fresh.eq(True)].copy()
    quotes['quote_received_at'] = pd.to_datetime(quotes.quote_received_at, utc=True)
    joined = pd.merge_asof(result.sort_values('decision_at'),
        quotes[['code','quote_received_at','spread_bps','imbalance']].sort_values('quote_received_at'),
        left_on='decision_at', right_on='quote_received_at', by='code', direction='backward',
        tolerance=pd.Timedelta(seconds=tolerance_seconds))
    joined['quote_available'] = joined.quote_received_at.notna()
    return joined.sort_values('_order').drop(columns='_order')
