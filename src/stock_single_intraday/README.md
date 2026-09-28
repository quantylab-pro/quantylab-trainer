# stock_single_intraday

표준 제품군 `stock-single-intraday`는 한 종목의 장중 신호와 당일 진입·청산을 평가한다.
초기 고정 유니버스는 `quantylab.acquisition.kiwoom.stock_intraday_universe.MAJOR_STOCK_CODES`의 30개 대형주다. 현재 장중 수집기는 이 유니버스의 1분봉을 5분마다 적재하고 있다.

## 데이터 현황

ETF와 주식 분봉이 같은 `stock_minute_candle` 테이블을 쓰므로 ETF 코드 조인으로 제외한다. 고정 종목별 일수와 분봉 완전성을 확인한다:

```bash
python -m quantylab.trainer.stock_single_intraday.readiness \
  --start-date 20260101 --end-date 20991231 \
  --output output/stock_single_intraday/readiness.json
```

주식 수집기는 원천 시각을 `HHMMSS`로 저장하고 ETF 수집기는 `HHMM`으로 저장한다. 종목 전용 로더가 모델 의사결정 분인 `HHMM`으로 정규화하고 ETF 코드를 제외한다.

2026-09-24 기준 장중 수집은 2026-09-15에 시작했다. 초기 점검에서 30종목 각각 2~6거래일, 일부 종목은 하루 중앙값 36~100여 분봉만 있어 세션이 불완전했다. 역사 분봉 백필을 시도 중이며, 완료 뒤 날짜·종목별 실제 커버리지를 다시 확인해야 한다. 충분한 연속 학습/검증/잠금 구간이 확인되기 전에는 후보를 학습하거나 정식 버전·배포 승인을 만들지 않는다.

## 연구 계약

- 종목별 출력 계약이며 포트폴리오 수익률로 합산하지 않는다.
- 신호는 분봉 종가까지의 정보만 쓰고, 다음 분봉 시가 이후에 체결한다.
- 학습, 검증, 잠금 평가는 거래일 단위로 순서대로 나눈다. 세션 누락과 중간 결측은 체결로 메우지 않는다.
- 당일 포지션은 세션 종료 전에 청산한다. 수수료, 매도세, 스프레드, 슬리피지와 체결 지연 가정은 실험 설정과 manifest에 남긴다.
- 종목별 성과, 횡단면 분포, 거래 수, 활성 거래일, Sharpe, 최대 낙폭, 회전율 및 비용 민감도를 함께 보고한다.
- 후보 선택에 사용한 날짜는 독립 잠금 평가라고 다시 부르지 않는다.

학습 실험은 위 데이터 현황과 고정된 날짜 분할을 검토한 뒤 수행한다. 현재 쌓이는 데이터의 날짜 범위와 주식 분봉 수집은 ETF 분봉에서 차용한 것으로 가정하지 않는다.

## 연구 실행

1분봉 신호를 5분봉으로 집계하는 별도 n분 모델 연구:

```bash
PYTHONPATH=src python -m stock_single_intraday.research_nminute \
  --output output/stock_single_intraday/run_YYYYMMDD
```

연구는 5분 OHLCV, 15/30/60분 고정 보유시간과 8개의 walk-forward 구간을 쓴다. 중단 후 이어서 실행할 때는 같은 output 경로와 `--resume`을 사용한다. 오래 걸리는 실행은 tmux 세션에서 직접 구동하지 말고 systemd 백그라운드 서비스와 메모리 상한, 로그 파일을 사용한다. 기준 통과 후보가 없으면 승자를 만들지 않는다.

5분 모델의 2026-09-25 연구 결과도 기준 통과 후보가 없었다. 전체 fold 지표 및 실행 판정은 [5분 모델 검토 보고서](REVIEW_20260925_NMINUTE_V2.md)를 참고한다. 이전 1분 연구 결과는 [1분 모델 검토 보고서](REVIEW_20260924.md)에 유지한다. 두 intraday 계열은 stock swing 모델과 별도이며, 기존 swing 모델의 산출물과 결과를 수정하지 않는다.

## 당일 종가 청산 모델

단기 보유시간을 고정하지 않고, 5분 신호 후 다음 봉 시가에 진입해 당일 마지막 완전한 5분봉 종가에 청산하는 별도 연구는 아래 스크립트로 시작한다.

```bash
bin/train_stock_single_intraday_eod.sh
```

이 스크립트는 3GiB 메모리 상한의 사용자 systemd 서비스에서 fold마다 별도 Python 프로세스를 실행하고, 현재 `stock-single-intraday-v1-eod-current-tax-exp01` tmux 세션에 로그와 서비스 상태를 붙여 보여준다. tmux 화면이 닫혀도 학습 서비스는 분리되어 계속 실행되며, fold·모델 조합마다 체크포인트를 저장해 같은 스크립트 재실행으로 이어서 계산한다. 현재 분봉 데이터에서 마지막 완전 5분봉은 15:15 시작, 15:19분봉 종가까지다. 이 청산 시각보다 30분 이상 앞선 진입만 검토한다. 비용·체결 가정과 후보 통과 기준은 run report에 기록하며, 검증 통과만으로 배포를 승인하지 않는다.

2026-09-25의 첫 당일 종가 청산 연구는 HGB/Ridge 6개 사양, VWAP·시초범위 돌파·상대강도 3개 규칙 기준을 비교했지만 통과 후보가 없었다. 상세 결과는 [EOD 청산 연구 보고서](REVIEW_20260925_EOD_CLOSE_V1.md)에 기록했다.

이후 비용 초과 분류, 피처 확장, 횡단면 rank, 진입 시간대와 현재 세율 적용을 포함한 개선 실험을 수행했지만 거래 수·거래당 비용 후 기대값·폴드 일관성 기준을 모두 만족한 모델은 아직 없다. [iteration 2 보고서](REVIEW_20260925_EOD_CLOSE_V2.md), [iteration 3 보고서](REVIEW_20260925_EOD_CLOSE_V3.md), [iterations 4–5 보고서](REVIEW_20260925_EOD_CLOSE_V4_V5.md), [현재 세율 실험 exp01](REVIEW_20260926_EOD_CURRENT_TAX_EXP01.md)를 참고한다. 정식 제품 버전은 v1로 유지하며 실험 ID를 버전 뒤에 붙인다. 기존 `stock-single-swing-v2`의 결과와 페이지는 이 작업에서 수정하지 않았다.

## 거래비용 가정

EOD 연구의 기준 왕복 비용은 왕복 증권사 수수료 3bp(매수·매도 각 1.5bp, 계좌 수수료가 다르면 교체 필요), 매수·매도 슬리피지 각 3bp, 매도 세금 합계다. 따라서 KOSPI 기준 2025년 24bp, 2026년 29bp다. 2026년 KOSPI 매도 세금 20bp는 증권거래세 5bp와 농어촌특별세 15bp를 합한 값이다([증권거래세법 시행령](https://law.go.kr/LSW/lumLsLinkPop.do?chrClsCd=010202&lspttninfSeq=64014), [국회예산정책처 2026 조세 분석](https://www.nabo.go.kr/board/file/down.do?fid=33319156)). 스트레스 평가는 수수료와 세금을 고정하고 슬리피지만 높인다. 실제 호가 스프레드와 체결 지연은 데이터가 없어 따로 추정되지 않았으며, 고정 슬리피지는 검증된 실측치가 아니다.

2026-09-26 비용 분해를 반영한 v6의 보조 비용 스트레스는 [v6 검토 보고서](REVIEW_20260926_EOD_CLOSE_V6.md)를 참고한다. 현재 세율을 전 기간에 적용한 공식 비교 실험은 [v1/eod-current-tax-exp01 결과](../../output/stock_single_intraday/stock-single-intraday-v1/eod-current-tax-exp01/report.json)에 보존했다.

## v1 추가 실험

당일 청산 방식을 비교한 `eod-exit-exp02`는 완료된 5분봉 종가로 손절·익절·추적청산 조건을 판단하고 다음 봉 시가에 청산한다. 실행은 `bin/train_stock_single_intraday_exit.sh`이며 systemd 작업자와 `stock-single-intraday-v1-eod-exit-exp02` tmux 화면을 사용한다. 12개 설정 모두 검증 기준을 통과하지 못했다. [exp02 검토 보고서](REVIEW_20260927_EOD_EXIT_EXP02.md)에 거래 결과와 내부 분봉 결측 진단을 기록했다.
