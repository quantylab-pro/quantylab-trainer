"""Scheduled minute refresh with a failing exit status on partial API failure."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from .auto_collect import ROOT
from quantylab.acquisition.kiwoom.etf_minute_candle import run


def main():
    result=run.fn(n=5, tic_scope='1')
    ROOT.mkdir(parents=True,exist_ok=True)
    record=dict(checked_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), **result)
    temp=ROOT/'bars-health.tmp'
    temp.write_text(json.dumps(record,indent=2))
    temp.replace(ROOT/'bars-health.json')
    if result['errors']:
        raise SystemExit(1)


if __name__=='__main__': main()
