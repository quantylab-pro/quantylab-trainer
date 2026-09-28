"""Small read-only NAV/program-flow feasibility sample, with receipt timestamps."""
import argparse
import json
import signal
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from quantylab.clients.kiwoom_rest import KiwoomRestClient


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    def timeout_handler(signum,frame):raise TimeoutError('bounded pilot timeout')
    signal.signal(signal.SIGALRM,timeout_handler)
    signal.alarm(30)
    client=KiwoomRestClient(real=True)
    signal.alarm(0)
    requests=[('nav_102110','/api/dostk/etf','ka40006',{'stk_cd':'102110'},'etftisl_trnsn'),
        ('nav_396500','/api/dostk/etf','ka40006',{'stk_cd':'396500'},'etftisl_trnsn'),
        ('program_102110_20260911','/api/dostk/mrkcond','ka90008',
         {'stk_cd':'102110','amt_qty_tp':'1','date':'20260911'},'stk_tm_prm_trde_trnsn')]
    for name,url,api_id,data,key in requests:
        try:
            signal.alarm(20)
            payload=client._get_single_data(url=url,api_id=api_id,data=data)
            if str(payload.get('return_code','0'))!='0':raise ValueError('API failure')
            rows=payload.get(key,[])
            result=dict(api_id=api_id,request=data,received_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                rows=rows,status='captured',first_page_only=True)
            print(name,len(rows),list(rows[0]) if rows else [],flush=True)
        except Exception as exc:
            result=dict(status='error',error_type=type(exc).__name__);print(name,result,flush=True)
        finally:
            signal.alarm(0)
        (out/f'{name}.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
