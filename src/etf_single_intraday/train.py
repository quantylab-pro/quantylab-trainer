"""분봉 단일 ETF 학습 CLI.

실행 예:

    python -m quantylab.trainer.etf_single_intraday.train \
        --dataset intraday_20260907 --trading-method swing \
        --episodes 10 --network-type mamba --device cuda

공통 PPO 학습 로직은 기존 단일 ETF 학습기에서 재사용하되, 환경 클래스는
이 패키지의 분봉 전용 ``IntradayTradingEnvironment``를 사용한다.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ..etf_single_swing import train as _legacy_train
from .environment import IntradayTradingEnvironment


# 기존 train.py의 main()은 이 전역 클래스를 참조하므로, CLI 진입 시에만
# 분봉 전용 환경으로 교체한다. 기존 etf_single_swing 실행에는 영향이 없다.
_legacy_train.SwingTradingEnvironment = IntradayTradingEnvironment


def main():
    """분봉 단일 ETF 학습을 실행한다."""
    # 기존 단일 ETF CLI를 재사용하되, 분봉에서 검증된 기본값을 자동 적용한다.
    # 사용자가 같은 옵션을 직접 주면 그 값을 우선한다.
    defaults = {
        '--rebalance-interval-bars': '15',
        '--max-position-change': '0.25',
        '--turnover-penalty-scale': '0.5',
        '--hold-threshold': '0.10',
        '--action-mix-prob': '0.0',
        '--action-mix-start': '0.0',
        '--action-mix-end': '0.0',
        '--rolling-sharpe-scale': '0.0',
        '--min-concentration': '2.0',
        '--drawdown-penalty-scale': '5.0',
        '--loss-aversion': '1.0',
        '--policy-dropout': '0.0',
        '--value-dropout': '0.0',
    }
    for option, value in defaults.items():
        if not any(arg.split('=', 1)[0] == option for arg in sys.argv):
            sys.argv.extend([option, value])
    if '--flat-at-session-end' not in sys.argv:
        sys.argv.append('--flat-at-session-end')

    # build_dataset이 기록한 날짜 split을 학습 CLI에 자동 연결한다.
    # 사용자가 명시한 옵션은 그대로 존중한다.
    if not any(arg.split('=', 1)[0] == '--train-end-date' for arg in sys.argv):
        base_path = '/home/quantylab/quantylab-trainer'
        dataset = 'intraday_20260907'
        for index, arg in enumerate(sys.argv):
            if arg == '--base-path' and index + 1 < len(sys.argv):
                base_path = sys.argv[index + 1]
            elif arg == '--dataset' and index + 1 < len(sys.argv):
                dataset = sys.argv[index + 1]
        meta_path = Path(base_path) / 'data' / dataset / 'dataset_meta.json'
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            split_dates = meta.get('split_dates', {})
            train_end = split_dates.get('train_end')
            val_end = split_dates.get('val_end')
            if train_end and val_end:
                val_start = (
                    datetime.strptime(str(train_end), '%Y%m%d') + timedelta(days=1)
                ).strftime('%Y%m%d')
                sys.argv.extend(['--train-end-date', str(train_end)])
                if not any(arg.split('=', 1)[0] == '--validation-start-date' for arg in sys.argv):
                    sys.argv.extend(['--validation-start-date', val_start])
                if not any(arg.split('=', 1)[0] == '--validation-end-date' for arg in sys.argv):
                    sys.argv.extend(['--validation-end-date', str(val_end)])
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            # 구형 데이터셋은 기존 CLI 동작을 유지한다.
            pass
    return _legacy_train.main()


if __name__ == '__main__':
    main()
