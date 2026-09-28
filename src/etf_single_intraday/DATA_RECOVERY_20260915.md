# 장중 데이터 수집 복구 — 2026-09-15

## 변경

- 운영 배포 커밋: quantylab `8d705288`. 로컬 수정/테스트 → GitHub push →
  mison048 fast-forward pull → `prefect-intraday.yaml`의 두 deployment만 갱신.
- 호가 수집의 체결강도 전체 페이지 순회를 ka10046 첫 페이지 조회로 변경.
  API 원래 필드와 기존 연구용 체결시간/체결강도 별칭을 함께 저장한다.
  `intensity_first_page_only=true`로 이력 범위를 명시한다.
- 각 호가/체결강도 호출에 8초 제한, 초기 클라이언트 생성에 20초 제한.
  flow 전체 105초 수집 제한 도달 시에도 health에 timed_out/error를 기록한다.
- NAV/프로그램 deployment를 공용 default에서 realtime 풀로 이동.
  기존 realtime 동시 실행 제한 1은 유지한다.
- 로컬 분봉 systemd 서비스에 정상 SSH DB 터널 포트 15432를 명시.
  기존 오류는 PostgreSQL 기본 포트 5432 연결 거부였다.
- 5분 이상 밀린 미실행 예약 중 NAV 53건, 호가 2건을 취소 처리했다.
  저장 원본이나 실행 이력은 삭제하지 않았다. 실시간 스냅샷은 과거 예약
  시점으로 소급 복원할 수 없으므로 새 수신 시각을 기준으로 보존한다.

## 확인 결과

- snapshot/API 오류 테스트 3개 통과.
- 분봉 수집 64종목, 최근 5일 45,255행 upsert, 오류 0건.
- 모델 대상 26 ETF의 9/14 분봉 8,539행, 9/15 13:23까지 5,848행 확인.
- 9/14 code/date/time 중복 0건. ETF당 177~381행이며 15:15 봉은
  23 ETF에서만 존재한다. 모든 종목의 모든 분이 채워졌다는 뜻은 아니다.
- 13:26:21 호가 배치: captured=26, fresh=26, timed_out=false.
  체결강도 첫 페이지 표시도 26파일 모두 확인.
- 13:26:42 NAV/프로그램 배치: 52파일, NAV 26개 신선,
  프로그램 26개 빈 응답, 오류/시간초과 없음. 수동 확인 실행 COMPLETED.
- 후속 13:25 예약 NAV 실행도 13:27:14에 시작해 13:27:30 COMPLETED,
  NAV fresh=26, 프로그램 empty=26으로 완료했다. 복구 당시 대기열로
  예약 시각과 실제 수신 시각은 다르며 학습에는 실제 수신 시각을 쓴다.

## 지속 수집

- mison048 Prefect: 평일 장중 호가 매분, NAV/프로그램 5분 간격.
- mison026 systemd: 평일 KST 16:10 최근 분봉 갱신, timer enabled 확인.
- 원본: mison048 `/home/quantylab/quantylab/var/etf_intraday/`
  아래 `snapshots/YYYYMMDD/`, `context/YYYYMMDD/`.
- 상태: 운영 `health.json`, `context-health.json`; 로컬 `bars-health.json`.

장중 9/15 분봉은 미완료 세션이며 평가 잠금 데이터로 사용하지 않는다.
프로그램 빈 응답을 0으로 대체하지 않는다. NAV/호가의 다일 표본 확보는
이제 진행 중이며 기존 20거래일 축적 조건은 아직 충족하지 못했다.
자동 재학습/모델 승격/주문은 이번 작업에 포함하지 않았다.
