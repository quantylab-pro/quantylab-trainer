"""Read-only weekday sampler. Stale quotes are retained, never called fresh."""
import fcntl
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/quantylab/quantylab/var/etf_intraday')
CODES = '091230 102110 138540 139220 139230 139260 143860 157490 157500 228790 228810 232080 261070 277630 292150 305540 307520 329200 364970 364980 396500 463250 466940 471760 494670 496080'.split()


def market_window(now):
    return now.weekday() < 5 and 540 <= now.hour * 60 + now.minute < 930


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'collector.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        now = datetime.now(ZoneInfo('Asia/Seoul'))
        status = dict(checked_at=now.isoformat(), status='outside_market_window')
        result = 0
        if market_window(now):
            out = ROOT / 'snapshots' / now.strftime('%Y%m%d') / now.strftime('%H%M%S_%f')
            try:
                proc = subprocess.run([sys.executable, '-m',
                    'quantylab.trainer.etf_single_intraday.collect_microstructure',
                    '--output', str(out), '--codes', *CODES], timeout=105)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
            records = [json.loads(p.read_text()) for p in out.glob('*.json')]
            captured = sum(r.get('status') == 'captured' for r in records)
            fresh = sum(r.get('quote_fresh', False) for r in records)
            result = int(returncode != 0 or captured != len(CODES))
            status.update(status='error' if result else ('ok' if fresh else 'no_fresh_quotes'),
                output=str(out), expected=len(CODES), captured=captured, fresh=fresh,
                note='Weekday gate only; holidays/stale quotes are not valid live features.')
        temp = ROOT / 'health.tmp'
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(ROOT / 'health.json')
        print(json.dumps(status), flush=True)
        if result:
            raise SystemExit(result)


if __name__ == '__main__':
    main()
