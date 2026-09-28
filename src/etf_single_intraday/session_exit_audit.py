"""Replay frozen candidates after the scheduled session-exit correction."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .minute_engine import simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for run in args.runs:
        root = Path(run)
        bundle = joblib.load(root / 'selected_candidate.joblib')
        data = pd.read_pickle(root / 'minute_samples.pkl')
        data = data[data.date >= '20260818']
        with threadpool_limits(limits=2):
            prediction = bundle['model'].predict(data[bundle['features']])
        if bundle.get('transform') == 'risk_scaled':
            prediction *= data.vol5.clip(lower=.0005).to_numpy()
        previous = json.loads((root / 'report.json').read_text())['recent']
        for label, cost, delay in [('double_cost', .0018, 1),
                                   ('delay_2min', .0018, 2),
                                   ('cost_36bp', .0036, 1)]:
            metric, trades, daily = simulate(
                data, prediction, bundle['horizon'], bundle['threshold'],
                bundle['dynamic'], cost=cost, delay=delay, detail=True)
            prefix = root.name + '_' + label
            trades.to_csv(out / (prefix + '_trades.csv'), index=False)
            daily.to_csv(out / (prefix + '_daily.csv'), index=False)
            result = dict(run=str(root), scenario=label, previous=previous[label],
                          corrected=metric,
                          return_change_pp=metric['mean_return_pct']
                          - previous[label]['mean_return_pct'])
            results.append(result)
            print(json.dumps(result), flush=True)
    (out / 'report.json').write_text(json.dumps(dict(
        results=results, deployment_approved=False,
        warning='Frozen models and thresholds; retrospective replay, not a new holdout.'
    ), indent=2))


if __name__ == '__main__':
    main()
