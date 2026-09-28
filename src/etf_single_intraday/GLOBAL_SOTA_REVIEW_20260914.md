# 미국·코인 데이터 확보 및 최신 아키텍처 검토

## 확보 완료

- Binance 공식 spot 공개 아카이브 BTCUSDT/ETHUSDT 1분봉: 2026-01-01~09-10, 각 364,320행, 총 728,640행. 36개 ZIP에 공식 CHECKSUM SHA256 검증. 원본과 Parquet, 출처/수신 시각 manifest 저장.
- FRED 일별: SP500 180행, NASDAQCOM 179행, VIXCLS 180행, DGS10 180행, DTWEXBGS 176행. 1/2부터 시작, 달러지수는 9/4까지, 나머지는 9/10까지. 행 수는 비결측 거래일 수와 동일하다고 단정하지 않음.
- `output/intraday_global_context_20260914/`, `output/intraday_crypto_sep_20260914/`, `output/intraday_us_context_20260914/`.
- 미국 첫 조회는 잘못된 CSV 경로로 실패했고 정식 fredgraph.csv 경로로 재실행해 위 수집 완료. 이전 실패 manifest는 보존.

## 실제 추가 피처 비교

BTC/ETH 각각 1/5/15분 수익률, 15분 변동성, taker 매수 거래대금 비율 생성. 완성된 봉 종료 시각+60초를 가용 시각으로 가정, backward as-of 최대 90초로 결합. 과거 실제 도착 시각을 확보한 것은 아니다. 주말/24시간 코인 수익을 국내 거래 시간과 혼동하지 않으며 USDT 표시 가격이지 USD 환율과 동일하다고 가정하지 않는다.

동일한 국내 기본 28피처·HGB·15분 보유, 고정 18bp 임계값, 3개 순차 검증 및 최근 회고 평가로 기본/코인 추가 비교. 미국 자료는 발표시점/빈티지 검증 전이므로 학습에서 제외.

| 최근 18거래일 | 기본 | 코인 추가 |
|---|---:|---:|
| 거래 수 | 109 | 110 |
| 비용 18bp 수익률 | +0.09580% | +0.05371% |
| 비용 36bp 수익률 | +0.02812% | -0.01616% |

ETF별 독립 계좌 평균이며 포트폴리오 수익률이 아니다. 새 기본 모델은 동일 정렬/동일 입력 조건으로 이번에 다시 학습한 대조군이므로 이전 다른 스크립트의 기본 모델과 동일 결과를 기대하지 않는다. 코인 추가가 이 설정에서는 악화됐다. 수집 성공을 유효 피처 검증 성공으로 간주하지 않는다. 반복 사용한 최근 구간은 새로운 독립 테스트가 아니며 실전 승인 없음.

## 아키텍처 검토 (공식 자료, 2026-09-14)

- **Chronos-2**: 120M encoder-only, 다변량/과거 외생변수/분위수 예측 지원, 모델 카드 Apache-2.0. 우선 검토 후보. 국내 ETF·코인·NAV는 past-only로 입력하고 미래에 알려진 시간 변수만 future covariate로 사용. 기초모델 사전학습 데이터와 평가 기간 중첩 및 라이선스 재확인 필요. [모델 카드](https://huggingface.co/amazon/chronos-2)
- **TimesFM 3.0**: 공식 저장소가 2026년 8월 공개 최신 버전으로 설명하며 native multivariate/covariates 지원. 공개 가중치는 non-commercial/non-production 제한이 있어 운영·상업 공유 기본 후보에서 제외. 2.5 가중치는 Apache-2.0, XReg/LoRA 지원이 대안. [공식 저장소](https://github.com/google-research/timesfm)
- **TimeXer**: 외생변수를 사용하는 시계열 Transformer의 공식 NeurIPS 2024 구현. 미국 전일 문맥·코인·장중 미시구조를 구분해 입력하는 구조 후보. 최신 종합 1위 모델이라고 주장하지 않음. [공식 코드](https://github.com/thuml/TimeXer)
- **PatchTST 및 작은 TCN**: PatchTST는 2023년 공식 구현을 가진 비교 기준이지 최신 SOTA라고 부르지 않는다. 60/120분 문맥, 세션 경계 마스크, 수익/손실 분위수 출력으로 작은 모델부터 검증하는 비용 효율적 비교군. [PatchTST](https://github.com/yuqinie98/PatchTST)

이 턴에서는 위 foundation model을 설치·학습하지 않았다. SOTA 벤치마크는 비용 후 금융 성과의 증거가 아니다. 데이터 추가 효과와 구조 변경 효과를 분리해 한 번에 하나씩 비교할 것.

## 다음 실험 설계

1. 코인은 전체 ETF에 강제 적용하지 않고 학습 구간에서 정의한 민감도/섹터별 적용 비교. 최근 성적으로 유니버스를 재선정하지 않는다.
2. 미국 현물 일별 값은 전일 상태 용도. 해당 소스의 실제 발표 시각·갱신/수정 이력이 없으면 엄밀한 point-in-time 검증을 주장하지 않는다. 한국 장중 위험선호 신호는 미국 선물/환율 실시간 자료가 더 직접적일 수 있지만 제공사 권한·비용·과거 보존 범위를 먼저 확인해야 한다.
3. Chronos-2 과거 외생변수 입력 vs 작은 TCN/TimeXer vs HGB를 동일 3구간에서 비교. 분봉 추론 주기와 다음 분 체결, 장 마감 청산을 유지.
4. 공유 가능 모델 판정은 다일 비용 후 안정성·날짜 집중도·지연 스트레스·피처 제거 대비 효과로 수행.

자동 미국/코인 수집 잡을 운영 Prefect에 추가한 것은 아니다. 이번은 과거 데이터 확보와 비교 실험, 아키텍처 검토까지 완료했다. 기존 NAV/호가 수집은 변경하지 않았다.

데이터 출처: [Binance 공식 공개 데이터](https://github.com/binance/binance-public-data), [FRED VIX](https://fred.stlouisfed.org/series/VIXCLS). 상세 비교는 `output/intraday_global_ablation_20260914/report.json`. 시점/결측/인과성 관련 테스트 7개 통과.
