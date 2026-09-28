"""Minute-by-minute signals, next-open orders and marked intraday equity."""
import numpy as np
import pandas as pd

BASE = ['ret1', 'ret3', 'ret5', 'ret15', 'vol5', 'vol15', 'range',
        'body', 'vwap_gap', 'vwap_slope', 'session_ret', 'elapsed', 'remaining']
EXTRA = ['same_time_volume', 'same_time_volatility', 'same_time_cum_volume',
         'seasonal_available', 'or5_high', 'or5_low', 'or15_high', 'or15_low',
         'or5_ready', 'or15_ready', 'vwap_cross', 'volume_trend', 'market_ret5']


def features(candles):
    parts = []
    for (_, _), group in candles.groupby(['code', 'date'], sort=True):
        d = group.sort_values('time').copy()
        minute = d.time.astype(int)//100*60 + d.time.astype(int)%100
        d['minute'] = minute
        c, v = d.close, d.volume
        contiguous = minute.diff().eq(1)
        for n in (1, 3, 5, 15):
            d[f'ret{n}'] = c.pct_change(n).where(minute.diff(n).eq(n), 0).fillna(0)
        for n in (5, 15):
            d[f'vol{n}'] = d.ret1.rolling(n, min_periods=2).std(ddof=0).fillna(0)
        d['range'] = (d.high-d.low)/c
        d['body'] = (c-d.open)/d.open
        d['cum_volume'] = v.cumsum()
        # Close*volume is a bar-based VWAP approximation, not trade-level VWAP.
        vw = (c*v).cumsum()/v.cumsum().replace(0, np.nan)
        d['vwap_gap'] = c/vw-1
        d['vwap_slope'] = vw.pct_change(5).where(minute.diff(5).eq(5), 0)
        d['vwap_cross'] = np.sign(d.vwap_gap).diff().fillna(0)/2
        d['session_ret'] = c/d.open.iloc[0]-1
        d['elapsed'] = (minute-540)/390
        d['remaining'] = (930-minute)/390
        for n in (5, 15):
            opening = d[(minute >= 540) & (minute < 540+n)]
            ready = (len(opening) == n)
            active = (minute >= 540+n) & ready
            d[f'or{n}_ready'] = active.astype(float)
            d[f'or{n}_high'] = np.where(active, c/opening.high.max()-1, 0)
            d[f'or{n}_low'] = np.where(active, c/opening.low.min()-1, 0)
        d['volume_trend'] = d.ret5 * (v/v.rolling(20, min_periods=1).mean().replace(0,np.nan)).clip(0,5)
        d['capacity'] = v.rolling(15, min_periods=15).median().fillna(0)*.01
        for h in (3,5,10,15,30):
            valid = (minute.shift(-(h+1))-minute).eq(h+1) & minute.shift(-(h+1)).le(915)
            d[f'y{h}'] = (d.open.shift(-(h+1))/d.open.shift(-1)-1).where(valid)
        parts.append(d)
    data = pd.concat(parts).sort_values(['code','time','date']).reset_index(drop=True)
    for source, target in [('volume','same_time_volume'), ('vol5','same_time_volatility'),
                           ('cum_volume','same_time_cum_volume')]:
        ref = data.groupby(['code','time'])[source].transform(
            lambda x: x.shift(1).rolling(20,min_periods=5).median())
        data[target] = (data[source]/ref.replace(0,np.nan)).clip(0,10).fillna(1)
        if source == 'volume':
            data['seasonal_available'] = ref.notna().astype(float)
    data['market_ret5'] = data.groupby(['date','time']).ret5.transform('mean')
    data[BASE+EXTRA] = data[BASE+EXTRA].replace([np.inf,-np.inf],np.nan).fillna(0)
    return data.sort_values(['date','code','time']).reset_index(drop=True)


def simulate(data, prediction, horizon, threshold, dynamic=False, cost=.0018,
             delay=1, detail=False):
    """No overlapping positions per ETF; decisions use completed bar only.

    All exits execute at next available open, including risk exits. A known
    15:15 time exit is scheduled beforehand. Last-observed-close liquidation
    on truncated sessions is recorded separately, not hidden as a normal fill.
    """
    data = data.copy()
    data['prediction'] = prediction
    codes = sorted(data.code.unique())
    cash = {c: 1e7 for c in codes}
    peak = cash.copy()
    worst = {c: 0. for c in codes}
    trades, daily = [], []
    truncations = 0
    for (date, code), d in data.groupby(['date','code'],sort=True):
        d = d.sort_values('minute')
        op, cl, minute, pred, cap, vol, vw = [d[x].to_numpy() for x in
            ('open','close','minute','prediction','capacity','vol5','vwap_gap')]
        q=0; entry=0.; entry_min=0; pending=None; cooldown=0
        entry_cash=0.; max_gain=0.; max_loss=0.; max_close=0.; entry_vol=0.
        start_cash=cash[code]
        for i in range(len(d)):
            m=minute[i]
            # The session deadline is known in advance, so signal latency
            # must not postpone it. Missing deadline bars use the first
            # observed open afterwards, overriding any pending risk exit.
            if q and m >= 915:
                pending=(i,'sell',0,'session')
            if pending and i >= pending[0]:
                _, action, amount, reason = pending
                pending=None
                if action == 'buy' and not q and m < 900:
                    q=int(min(amount,cash[code]*.25/(op[i]*(1+cost/2))))
                    if q>0:
                        entry=op[i]; entry_min=m; entry_cash=q*entry*(1+cost/2)
                        cash[code]-=entry_cash; max_gain=max_loss=0.;max_close=entry
                        entry_vol=max(float(vol[max(0,i-1)]),.0005)
                elif action == 'sell' and q:
                    proceeds=q*op[i]*(1-cost/2); cash[code]+=proceeds
                    trades.append(dict(code=code,date=date,entry_minute=int(entry_min),
                        exit_minute=int(m),holding=int(m-entry_min),pnl=float(proceeds-entry_cash),
                        net_bps=float((proceeds/entry_cash-1)*10000),reason=reason,
                        mfe=max_gain,mae=max_loss))
                    q=0;cooldown=m+3
            equity=cash[code]+q*cl[i]*(1-cost/2)
            peak[code]=max(peak[code],equity)
            worst[code]=min(worst[code],equity/peak[code]-1)
            if q:
                gain=cl[i]/entry-1
                max_gain=max(max_gain,gain);max_loss=min(max_loss,gain)
                max_close=max(max_close,cl[i])
                reason=None
                if m-entry_min >= horizon-1: reason='time'
                if dynamic and m-entry_min>=2:
                    if gain <= -max(.002,3*entry_vol): reason='stop'
                    elif max_gain>.002 and cl[i]/max_close-1 <= -max(.001,2*entry_vol): reason='trail'
                    elif pred[i]<0 and vw[i]<0: reason='signal'
                if reason and pending is None and i+delay<len(d):
                    pending=(i+delay,'sell',0,reason)
            elif pending is None and m>=555 and m<900 and m>=cooldown and pred[i]>threshold:
                # Missing next minute bars cannot be treated as immediate fills.
                if i+delay<len(d) and minute[i+delay]-m==delay and cap[i]>=1:
                    pending=(i+delay,'buy',int(cap[i]),'entry')
        if q:
            proceeds=q*cl[-1]*(1-cost/2);cash[code]+=proceeds;truncations+=1
            trades.append(dict(code=code,date=date,entry_minute=int(entry_min),exit_minute=int(minute[-1]),
                holding=int(minute[-1]-entry_min),pnl=float(proceeds-entry_cash),
                net_bps=float((proceeds/entry_cash-1)*10000),reason='truncated',mfe=max_gain,mae=max_loss))
        daily.append(dict(code=code,date=date,return_=cash[code]/start_cash-1))
    t=pd.DataFrame(trades)
    result=dict(mean_return_pct=float(np.mean([v/1e7-1 for v in cash.values()])*100),
        worst_intraday_drawdown_pct=float(min(worst.values())*100),trades=len(t),
        win_rate=float((t.pnl>0).mean()) if len(t) else 0.,
        mean_trade_bps=float(t.net_bps.mean()) if len(t) else 0.,
        mean_holding_minutes=float(t.holding.mean()) if len(t) else 0.,
        truncated_exits=truncations,etfs=len(codes),days=int(data.date.nunique()))
    if detail: return result,t,pd.DataFrame(daily)
    return result
