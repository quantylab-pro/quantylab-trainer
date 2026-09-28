# Stock single intraday v1: complete minute data status

Checked 2026-09-28 after the market opened. The preceding trading day was
2026-09-23; 2026-09-24 and 2026-09-25 were Chuseok holidays.

## Sources and point-in-time contract

- Minute OHLCV: `stock_minute_candle`.
- Ten-level quote and depth snapshots: `stock_bid.content` and `stock_bid.extra.quote`.
- Minute execution strength, trade quantity, and cumulative volume:
  `stock_bid.extra.intensity` from Kiwoom `ka10046`. These are API trend rows,
  not individual exchange prints or actual brokerage fills.
- `minute_bid_volume` and daily `trade_intensity` are empty and are not used.
- `microstructure.py` keeps only intraday snapshots with `received_at`, checks
  provider-time freshness, joins by market event minute, and records the later
  of candle close and API receipt as feature availability. It does not fill
  missing minutes. Complete five-minute bars require all five minute records
  available before the bar closes.

## Coverage on 2026-09-23

| Measure | Count |
| --- | ---: |
| Target stocks | 30 |
| Continuous-session minute slots, 09:00–15:19 | 11,400 |
| Valid candle minutes | 11,395 |
| Quote source minutes | 11,336 |
| Fresh execution-strength minutes | 11,319 |
| Complete joined minutes | 11,315 (99.3%) |
| Complete five-minute bars | 2,200 / 2,280 (96.5%) |

The Prefect `acquisition-stock-microstructure-deployment` runs every minute,
and `acquisition-stock-intraday-deployment` runs every five minutes during the
regular market. Both were deployed and scheduled as of this check. The
microstructure flow's health file reported 30/30 captures at 15:29 on
2026-09-25, but those were stale 20:00 provider quotes on a holiday. A
successful flow state alone does not prove usable market data; the new loader
rejects those rows.

At 09:11 on 2026-09-28, live quote and candle collection both had 330
stock/minute rows (30 stocks over 09:00–09:10). The new complete-data loader
joined 309 minutes across all 30 stocks. The 21 missing joins are opening
execution-strength observations; 39 complete five-minute bars were available
for features at that point.

Kiwoom sometimes encodes a down-moving stock's OHLC prices as negative
numbers. `data.py` now converts OHLC to absolute prices while retaining the
signed day-on-day change. The old positive-price filter omitted 68 of the
first 180 fresh minutes on 2026-09-28. A database audit of the 30-stock
universe found no additional omitted distinct minutes from 2025-09 through
2026-08 because positive duplicate rows existed. Existing historical
backtest results are preserved; this live-data correction by itself does not
establish that they changed.

Only one complete combined trading day exists so far. No quote/strength model
is trained or promoted from this sample. The earlier candle-only v1 experiment
results remain intact. The next experiment should stream complete days through
`iter_complete_stock_days`, use chronological training/validation/held-out
periods, and measure actual quoted spread and depth in its execution rules.
