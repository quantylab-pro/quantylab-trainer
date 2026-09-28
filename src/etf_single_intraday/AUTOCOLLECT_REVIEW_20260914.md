# 자동 수집 및 추가 모델 검토 — 2026-09-14

## 설치

- 원본 서비스: `/home/quantylab/quantylab/ops/systemd/etf-intraday-*`
- 사용자 systemd에 링크하고 두 타이머 enable 완료. Linger=yes로 로그아웃 이후에도 실행 가능.
- `etf-intraday-collect.timer`: 평일 매분, Python에서 KST 09:00 이상 15:30 미만 제한. 26 ETF 호가/체결강도 읽기 전용 조회. 주문 없음.
- `etf-intraday-bars.timer`: 평일 KST 16:10 최근 달력 5일 ETF 1분봉 DB upsert. 기존 전체 대상 ETF 사용. 아직 첫 예약 실행 전이므로 장마감 자동 실행 성공을 주장하지 않음.
- 스냅샷: `/home/quantylab/quantylab/var/etf_intraday/snapshots/YYYYMMDD/`
- 상태: 같은 디렉터리의 `health.json`, `bars-health.json`(분봉 갱신 실행 후).
- 호가 첫 실행 및 다음 예약 실행: 각각 26/26 captured, fresh 26/26, 서비스 종료 0 확인.
- lock/서비스 단일 실행, 105초 조회 timeout, 부분 실패 exit 1, 원본 보존. 자동 삭제 없음.
- 휴장 달력 미연동: 평일 휴장에도 조회할 수 있으며 오래된 호가는 fresh=false. 공급자 시각은 날짜 없는 값이라 freshness는 시간차 기반 보조 판정임.
- 장애 확인: `systemctl --user status etf-intraday-collect.service`; `journalctl --user -u etf-intraday-collect.service -n 40`.
- 중지: `systemctl --user disable --now etf-intraday-collect.timer etf-intraday-bars.timer` (이미 실행 중인 서비스는 별도 stop).

## 데이터·피처·전략 수정

학습 목표가 미래 종가 수익률이던 것을 실제 시뮬레이터와 동일한 다음 분 시가 진입→예정 청산 다음 분 시가 수익률로 수정했다. 캐시를 읽어도 목표를 재생성한다. 과거 결과와 새 결과는 목표가 달라 직접적인 구조 우열 비교가 아니다.

`microstructure_features.join_quotes`는 공급자 과거 시각이 아니라 실제 수신 시각 기준 backward as-of 결합, 최대 90초 제한, 결측 플래그를 제공한다. 수집 당일 데이터를 과거에 소급하지 않는다. 이 피처는 이력 부족으로 이번 학습에는 미포함이다.

기존 매분 추론, 5/15/30분 청산, 고정/동적 청산, 거래비용·지연·유동성 제한을 유지한다. AI 피처는 시점 정합성이 검증된 이력이 필요하며 이번 실험에 추가하지 않았다.

## 추가 72개 설정의 실험 결과

142만 분봉 표본, 과거 학습/6·7·8월 3개 순차 검증 구간으로 선별. 최근 8/18–9/10은 이미 관찰했던 회고 평가이지 새로운 독립 테스트가 아니다.

| 계열별 선택 후보 | 최근 거래 수 | 왕복 9bp | 18bp | 36bp |
|---|---:|---:|---:|---:|
| 선형 Ridge 30분 | 88 | -0.04828% | -0.06938% | -0.11157% |
| 최근 가중 HGB 30분 | 31 | +0.01606% | +0.00588% | -0.01444% |

수익은 ETF별 독립 1천만원 계좌 수익률 평균이며 포트폴리오 수익률이 아니다. Ridge/깊은 트리 36설정, 변동성 보정/최근 가중 36설정. 각 계열 후보는 최근 성적 아닌 검증 점수로 선택했으며 전체 검증 점수는 Ridge가 더 높지만 최근 손실이다. 최근 가중 후보는 검증 3구간 중 2구간 손실이다. 따라서 **안정적 개선 모델 확보 실패, 모두 deployment_approved=false**.

산출물:

- `output/intraday_minute_20260914_architecture/report.json`
- `output/intraday_minute_20260914_aligned/report.json`
- 각 디렉터리 `selected_candidate.joblib`, 비용별 거래·일간 CSV.

추가 수집의 목적은 실제 호가 비용과 체결강도를 충분히 축적해 기존 분봉 신호의 비용 후 예측력을 검증하는 것. 자동 수집은 설치했지만 자동 재학습/실거래 배포는 설치하지 않았다.
