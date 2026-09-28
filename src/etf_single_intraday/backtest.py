"""분봉 단일 ETF 백테스트 CLI."""

from ..etf_single_swing import backtest as _legacy_backtest
from .environment import IntradayTradingEnvironment


_legacy_backtest.SwingTradingEnvironment = IntradayTradingEnvironment


def main():
    return _legacy_backtest.main()


if __name__ == '__main__':
    main()
