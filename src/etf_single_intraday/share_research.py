"""Reproducible recent-date audit with a fixed pre-validation universe."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from ..intraday_features import (load_minute_candles, CORE_FEATURE_COLUMNS,
    AI_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, _add_previous_ai_features,
    _add_previous_market_features)
from .research import samples, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--snapshot', help='Reuse a trusted local sample snapshot; audit AI timestamps anew')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if args.snapshot:
        data = pd.read_pickle(args.snapshot)
        codes = sorted(data.code.unique())
        data = data.drop(columns=AI_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS, errors='ignore')
        # Only score/direction/factor fields are used in this branch. The
        # synthetic denominator is discarded along with both price-gap fields.
        data['close'] = 1.
        data = _add_previous_ai_features(data, '20240101', '20260910')
        data = _add_previous_market_features(data, '20240101', '20260910')
        data = data.drop(columns=['close', 'ai_upside_gap', 'ai_downside_gap'])
    else:
        candles = load_minute_candles('20260129', '20260910')
        early = candles[candles.date <= '20260529']
        counts = early.groupby(['code', 'date']).size().groupby('code').median()
        codes = sorted(counts[counts >= 330].index)
        print('fixed universe', len(codes), flush=True)
        data = samples(candles[candles.code.isin(codes)], 15)
    data = data.sort_values(['date', 'code', 'time']).reset_index(drop=True)
    # Causal feature interactions: price/volume confirmation and session regime.
    data['trend_volume'] = data.intra_ret_15 * data.volume_ratio_20.clip(0, 5)
    data['vwap_reversal'] = -data.session_vwap_gap * data.volatility_15
    data['overnight_followthrough'] = data.overnight_gap * data.session_return
    data['multi_day_alignment'] = data.prior_day_ret_5 * data.intra_ret_15
    extra = ['trend_volume', 'vwap_reversal', 'overnight_followthrough', 'multi_day_alignment']
    data.to_pickle(out / 'samples.pkl')
    folds = [('20260529', '20260601', '20260612'),
             ('20260630', '20260701', '20260715'),
             ('20260731', '20260803', '20260814')]
    # AI date alone cannot establish publication availability. Until that
    # provenance is audited these shareable candidates use candle features.
    variants = [('base', CORE_FEATURE_COLUMNS, 15, 'squared_error'),
                ('interactions', CORE_FEATURE_COLUMNS + extra, 15, 'squared_error'),
                ('shallow', CORE_FEATURE_COLUMNS + extra, 7, 'squared_error'),
                ('median', CORE_FEATURE_COLUMNS + extra, 7, 'absolute_error')]
    if args.snapshot:
        safe_ai = [c for c in AI_FEATURE_COLUMNS if c not in ('ai_upside_gap', 'ai_downside_gap')]
        variants += [('safe_etf', CORE_FEATURE_COLUMNS + safe_ai, 15, 'squared_error'),
                     ('safe_market', CORE_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS, 15, 'squared_error'),
                     ('safe_both', CORE_FEATURE_COLUMNS + safe_ai + MARKET_FEATURE_COLUMNS, 15, 'squared_error')]
    rows = []
    for name, columns, leaves, loss in variants:
        scores = {t: [] for t in (.0009, .0015, .0025, .004)}
        for end, start, stop in folds:
            train = data[data.date <= end]
            val = data[data.date.between(start, stop)].reset_index(drop=True)
            model = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=leaves,
                min_samples_leaf=120, l2_regularization=15, learning_rate=.035,
                early_stopping=False, random_state=42, loss=loss)
            with threadpool_limits(limits=2):
                model.fit(train[columns], train.gross.clip(-.03, .03))
                pred = model.predict(val[columns])
            for threshold in scores:
                scores[threshold].append(evaluate(val, pred, threshold, codes,
                    sorted(val.date.unique()), cost=.0018))
        for threshold, metrics in scores.items():
            returns = [m['mean_return_pct'] for m in metrics]
            rows.append(dict(variant=name, threshold=threshold, folds=metrics,
                score=float(np.mean(returns) - .5*np.std(returns)),
                eligible=all(m['trades'] >= 10 for m in metrics)))
        joblib.dump(dict(model=model, features=columns, codes=codes,
            horizon=15, training_end='20260731', deployment_approved=False), out / f'{name}.joblib')
        print('completed', name, flush=True)
    eligible = [r for r in rows if r['eligible']]
    selected = max(eligible, key=lambda r: r['score']) if eligible else None
    # All recent results are audit only; selection above is already frozen.
    recent = data[data.date >= '20260818'].reset_index(drop=True)
    for row in rows:
        bundle = joblib.load(out / (row['variant'] + '.joblib'))
        with threadpool_limits(limits=2):
            pred = bundle['model'].predict(recent[bundle['features']])
        row['recent'] = evaluate(recent, pred, row['threshold'], codes,
                                sorted(recent.date.unique()), cost=.0018)
    report = dict(rows=rows, selected=selected, codes=codes,
        warning='Recent dates were inspected in previous research; retrospective audit.',
        ai_status=('Conservative creation/update timestamp filter' if args.snapshot else
                   'Excluded pending timestamp provenance audit'), deployment_approved=False)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'selected': selected}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
