"""Causal, fixed-horizon single-ETF model selection with untouched forward dates.

Run with python -m quantylab.trainer.etf_single_intraday.research --output DIR.
Each ETF has its own cash account. Report averages are not portfolio returns.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..intraday_features import load_minute_candles, build_intraday_features, FEATURE_COLUMNS


def samples(candles, horizon=15):
    features = build_intraday_features(candles)
    parts = []
    for (code, date), day in features.groupby(['code', 'date']):
        day = day.sort_values('time').reset_index(drop=True)
        minute = day.time.astype(int) // 100 * 60 + day.time.astype(int) % 100
        for i in range(30, len(day) - horizon, horizon):
            end = i + horizon
            # Sparse bars are not treated as consecutive minutes.
            if not np.all(np.diff(minute.iloc[i:end + 1]) == 1):
                continue
            if minute.iloc[end] > 15 * 60 + 15:
                continue
            entry = float(day.open.iloc[i + 1])
            exit_price = float(day.close.iloc[end])
            row = day.loc[i, FEATURE_COLUMNS].to_dict()
            row.update(code=code, date=date, time=day.time.iloc[i],
                       gross=exit_price / entry - 1,
                       entry=entry, exit_price=exit_price,
                       capacity=float(day.volume.iloc[max(0, i-14):i+1].median()) * .01)
            parts.append(row)
    return pd.DataFrame(parts)


def evaluate(data, prediction, threshold, codes, dates, cost=.0009):
    cash = {code: 10_000_000.0 for code in codes}
    daily = {(code, date): 0. for code in codes for date in dates}
    trades = 0
    for date in dates:
        before = cash.copy()
        indices = np.flatnonzero(data.date.to_numpy() == date)
        for i in indices:
            row = data.iloc[i]
            if prediction[i] <= threshold:
                continue
            # Integer shares, at most 25% exposure and 1% of prior median volume.
            shares = int(min(cash[row.code] * .25 / (row.entry * (1+cost/2)), row.capacity))
            if shares <= 0:
                continue
            pnl = shares * (row.exit_price * (1-cost/2) - row.entry * (1+cost/2))
            cash[row.code] += pnl
            trades += 1
        for code in codes:
            daily[code, date] = cash[code] / before[code] - 1
    returns = pd.DataFrame([[daily[c,d] for d in dates] for c in codes], index=codes, columns=dates)
    equity = (1 + returns).cumprod(axis=1)
    equity = pd.concat([pd.Series(1., index=codes, name='initial'), equity], axis=1)
    drawdown = equity / equity.cummax(axis=1) - 1
    result = dict(mean_return_pct=float(np.mean([v/1e7-1 for v in cash.values()])*100),
                  worst_etf_drawdown_pct=float(drawdown.min().min()*100), trades=trades,
                  profitable_etfs=int(sum(v>1e7 for v in cash.values())), etfs=len(codes), days=len(dates),
                  daily_equal_weight_return=returns.mean(axis=0).to_dict())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon', type=int, choices=[15,30,60], default=15)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    candles = load_minute_candles('20260709', '20260910')
    train_candles = candles[candles.date <= '20260819']
    counts = train_candles.groupby(['code','date']).size().groupby('code').median()
    codes = sorted(counts[counts >= 330].index)
    candles = candles[candles.code.isin(codes)].copy()
    print('liquid ETFs selected using train only:', len(codes), flush=True)
    data = samples(candles, args.horizon).sort_values(['date','code','time']).reset_index(drop=True)
    masks = {'train': data.date <= '20260819',
             'validation': data.date.between('20260820','20260827'),
             'historical_audit': data.date.between('20260828','20260907'),
             'forward_test': data.date >= '20260908'}
    data.to_csv(output/'samples.csv', index=False)
    train = data[masks['train']]
    val = data[masks['validation']].reset_index(drop=True)
    val_dates = sorted(candles.loc[candles.date.between('20260820','20260827'),'date'].unique())
    candidates = []
    best = None
    # Compact predeclared grid: test dates are not used for parameter selection.
    for leaves in (7, 15):
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=leaves,
                    min_samples_leaf=100, l2_regularization=10., learning_rate=.04,
                    early_stopping=False, random_state=42)
        with threadpool_limits(limits=2):
            model.fit(train[FEATURE_COLUMNS], train.gross.clip(-.05,.05))
            pred = model.predict(val[FEATURE_COLUMNS])
        for threshold in (.0009,.0015,.0025,.004):
            metrics = evaluate(val, pred, threshold, codes, val_dates)
            stress = evaluate(val, pred, threshold, codes, val_dates, cost=.0018)
            score = stress['mean_return_pct']
            metrics['selection_score_double_cost'] = score
            candidates.append(dict(leaves=leaves, threshold=threshold, **metrics))
            if best is None or score > best[0]:
                best = score, model, threshold, leaves
    score, model, threshold, leaves = best
    report = {'horizon':args.horizon, 'protocol': 'Single ETF independent cash accounts; train-only liquid universe; next-bar entry; fixed-horizon exit; no overlapping positions; 9bp roundtrip assumption; 1% prior-volume cap; 25% exposure cap; drawdown measured at daily closes; previous-session AI ETF numeric features joined as-of with same-day analysis excluded',
              'data_days': sorted(candles.date.unique()), 'codes': codes,
              'train_rows': len(train), 'selection': {'leaves':leaves,'threshold':threshold},
              'validation_candidates':candidates,
              'historical_audit_warning': 'Dates through 20260907 were used in earlier PPO training; only forward_test dates are new.',
              'baseline_cash_return_pct':0., 'evaluation':{}}
    for name in ('validation','historical_audit','forward_test'):
        part = data[masks[name]].reset_index(drop=True)
        dates = sorted(part.date.unique())
        if part.empty:
            report['evaluation'][name] = {'status':'missing_data'}
            continue
        with threadpool_limits(limits=2):
            pred = model.predict(part[FEATURE_COLUMNS])
        report['evaluation'][name] = evaluate(part,pred,threshold,codes,dates)
        report['evaluation'][name+'_double_cost'] = evaluate(part,pred,threshold,codes,dates,cost=.0018)
        report['evaluation'][name+'_always_long'] = evaluate(part,np.ones(len(part)),0,codes,dates)
    forward = report['evaluation']['forward_test']
    report['deployment_approved'] = False
    report['paper_action'] = 'CASH' if report['evaluation']['validation_double_cost']['mean_return_pct'] <= 0 else 'RESEARCH_CANDIDATE'
    report['status'] = 'research_only: require at least 20 new forward sessions and positive cost-stressed returns before deployment'
    joblib.dump(dict(model=model,features=FEATURE_COLUMNS,threshold=threshold,codes=codes,
                     horizon=args.horizon,training_end='20260819',
                     ai_features=True, deployment_approved=False),output/'candidate.joblib')
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps({'selection':report['selection'],'evaluation':report['evaluation'],'status':report['status']},indent=2),flush=True)


if __name__ == '__main__':
    main()
