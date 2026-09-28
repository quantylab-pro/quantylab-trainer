"""Append-only local quote/intensity snapshots using existing read-only auth.

Historical minute features must use received_at, never just the provider time
of day. This collector sends no orders and does not alter existing DB records.
"""
import argparse
from datetime import datetime
from pathlib import Path
import json
import time
from zoneinfo import ZoneInfo
from quantylab.clients.kiwoom_rest import KiwoomRestClient


def number(value):
    try: return abs(float(value))
    except (ValueError,TypeError): return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--codes',nargs='+',required=True)
    p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--interval',type=float,default=60)
    args=p.parse_args()
    if args.cycles<1 or args.interval<1: raise ValueError('positive cycles/interval required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    client=KiwoomRestClient(real=True)
    for cycle in range(args.cycles):
        begun=time.monotonic()
        for code in args.codes:
            record={'code':code}
            try:
                quote=client.get_stock_bid(code)
                if str(quote.get('return_code', '0')) != '0':
                    raise ValueError('quote API failure')
                record['quote_received_at']=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
                received=datetime.fromisoformat(record['quote_received_at'])
                source=str(quote.get('bid_req_base_tm','')).zfill(6)
                try:
                    source_seconds=int(source[:2])*3600+int(source[2:4])*60+int(source[4:6])
                    age=received.hour*3600+received.minute*60+received.second-source_seconds
                    record['quote_age_seconds']=age
                    record['quote_fresh']=0<=age<=120
                except (ValueError, TypeError):
                    record['quote_fresh']=False
                ask,bid=number(quote.get('sel_fpr_bid')),number(quote.get('buy_fpr_bid'))
                aq,bq=number(quote.get('sel_fpr_req')),number(quote.get('buy_fpr_req'))
                record['quote']=quote
                record['spread_bps']=(ask-bid)/((ask+bid)/2)*10000 if ask and bid and ask>=bid else None
                record['imbalance']=(bq-aq)/(bq+aq) if aq is not None and bq is not None and aq+bq else None
                intensity=client.get_trade_intensity(code)
                record['intensity_received_at']=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
                record['intensity']=[] if intensity is None else json.loads(intensity.to_json(orient='records',force_ascii=False))
                record['status']='captured'
            except Exception as exc:
                # Do not serialize exception payloads that may include auth.
                record['status']='error';record['error_type']=type(exc).__name__
            with (out/f'{cycle:05d}_{code}.json').open('x') as handle:
                json.dump(record,handle,ensure_ascii=False,indent=2)
            print(code,record['status'],flush=True)
        if cycle+1<args.cycles:
            time.sleep(max(0,args.interval-(time.monotonic()-begun)))


if __name__=='__main__': main()
