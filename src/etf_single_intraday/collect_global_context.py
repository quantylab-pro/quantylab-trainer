"""Public historical context. Downloads are not historical receipt timestamps."""
import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import zipfile
import pandas as pd
import requests


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['all','us','crypto_september'],default='all');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for symbol in ['BTCUSDT','ETHUSDT']:
        for month in range(1,9):
            name=f'{symbol}-1m-2026-{month:02d}.zip'
            jobs.append((name,f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{name}','crypto'))
    for series in ['SP500','NASDAQCOM','VIXCLS','DGS10','DTWEXBGS']:
        jobs.append((series+'.csv',f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2026-01-01&coed=2026-09-10','us'))
    if args.kind=='us':jobs=[j for j in jobs if j[2]=='us']
    if args.kind=='crypto_september':
        jobs=[]
        for symbol in ['BTCUSDT','ETHUSDT']:
            for day in range(1,11):
                name=f'{symbol}-1m-2026-09-{day:02d}.zip'
                jobs.append((name,f'https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{name}','crypto'))
    def fetch(job):
        name,url,kind=job
        record=dict(file=name,url=url,kind=kind,received_at=datetime.now(timezone.utc).isoformat())
        try:
            response=requests.get(url,timeout=25);response.raise_for_status()
            raw=response.content
            if kind=='crypto':
                checksum=requests.get(url+'.CHECKSUM',timeout=15);checksum.raise_for_status()
                expected=checksum.text.split()[0]
                if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('checksum mismatch')
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    frame=pd.read_csv(archive.open(archive.namelist()[0]),header=None)
                frame.columns=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','unused']
                unit='us' if frame.open_time.iloc[0]>10**14 else 'ms'
                frame['bar_close_utc']=pd.to_datetime(frame.close_time,unit=unit,utc=True)
                frame.to_parquet(out/(name+'.parquet'),index=False)
            else:
                frame=pd.read_csv(io.BytesIO(raw))
                if not len(frame) or name[:-4] not in frame.columns:raise ValueError('invalid CSV')
            (out/name).write_bytes(raw)
            record.update(status='downloaded',rows=len(frame),sha256=hashlib.sha256(raw).hexdigest())
        except Exception as exc:record.update(status='error',error_type=type(exc).__name__)
        print(name,record['status'],record.get('rows',0),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,jobs))
    (out/'manifest.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
