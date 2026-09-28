# etf_single_intraday

1분봉 데이터로 한 번에 하나의 ETF를 거래하는 PPO 학습 패키지다.

- 입력 데이터: `data/intraday_*/environment.csv`, `training_scaled.csv`
- 학습 진입점: `python -m quantylab.trainer.etf_single_intraday.train`
- 백테스트 진입점: `python -m quantylab.trainer.etf_single_intraday.backtest`
- 여러 ETF를 동시에 보유하고 비중을 배분하는 학습은 `etf_portfolio_swing`이 담당한다.

현재 학습은 64개 ETF를 청크별로 순차 처리하며 하나의 정책/가치 네트워크를
공유한다. 따라서 ETF마다 별도 모델 파일을 만드는 방식은 아니고, 단일 ETF
환경에서 학습한 공통 정책 모델이다.

2026-09-11 개선 연구에서는 별도의 gradient boosting 후보도 학습했다.
데이터·환경 수정과 비용 포함 평가 결과는 [개선 검토 보고서](REVIEW_20260911.md)를 참고한다.
현재 후보는 연구 전용이며 운용 승인된 모델은 없다.

2026-09-12 시점 재검토 및 공유 후보는 [최신 검토](REVIEW_20260912.md)를 참고한다.
매분 판단·장중 청산 및 96개 설정의 2026-09-14 결과는 [분봉 intraday 구현 보고서](REVIEW_20260914.md)를 참고한다.
이전 일별 top-k 성능은 시간 누수로 무효이며, AI 분석 날짜만 사용한 결과도 생성·수정 시각 검증이 필요하다.

- 연구 학습: `python -m quantylab.trainer.etf_single_intraday.research --horizon 15 --output output/new_experiment`
- 저장 연구 후보 평가: `python -m quantylab.trainer.etf_single_intraday.evaluate_candidate --help`
- 긴 이력·AI 시장 국면 비교: `python -m quantylab.trainer.etf_single_intraday.market_research --output output/long_experiment`
- 위의 기존 `backtest`는 연구 후보용 평가 경로가 아니다.
