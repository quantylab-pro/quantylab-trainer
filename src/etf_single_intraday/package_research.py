"""Package a trusted local experiment for review; never deploys or orders."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args = p.parse_args()
    root = Path(args.run)
    report = json.loads((root / 'report.json').read_text())
    row = report['selected']
    if row is None:
        raise ValueError('No eligible candidate to package')
    bundle = joblib.load(root / (row['variant'] + '.joblib'))
    bundle.update(threshold=row['threshold'], variant=row['variant'],
                  research_only=True, deployment_approved=False)
    joblib.dump(bundle, root / 'selected_candidate.joblib')
    lines = ['# 분봉 연구 후보 모델 카드', '',
        f"선택 모델: {row['variant']}, 진입 임계값: {row['threshold']}", '',
        '학습은 2026-07-31까지이며, 최근 평가는 2026-08-18~09-10이다.',
        '26개 ETF 독립 1천만원 계좌 평균 수익률. 포트폴리오 수익률이 아니다.',
        '15분 보유, 다음 봉 시가 진입, 왕복 비용 18bp, 비중 25% 이하, 직전 거래량 주문 상한.',
        '최근 기간은 이전 실험에서 반복 확인했으므로 독립 최종 검증이 아니다.', '',
        '| 구간 | 수익률 % | 거래 수 |', '|---|---:|---:|']
    for i, m in enumerate(row['folds'], 1):
        lines.append(f"| 검증 {i} | {m['mean_return_pct']:.6f} | {m['trades']} |")
    m = row['recent']
    lines.append(f"| 최근 18일 | {m['mean_return_pct']:.6f} | {m['trades']} |")
    lines += ['', 'AI 처리: ' + report['ai_status'], '',
        '공유 목적은 연구 재현 및 검토이다. 수익성 확정 또는 실거래 배포 승인 모델이 아니다.',
        '표본은 samples.pkl, 전체 비교는 report.json, 선택 모델은 selected_candidate.joblib에 있다.',
        '후보는 검증 3구간 각각 10건 이상 거래를 요구한 뒤 평균−0.5×표준편차로 선택했다.']
    (root / 'MODEL_CARD.md').write_text('\n'.join(lines) + '\n')
    manifest = {}
    for name in ('report.json', 'samples.pkl', 'selected_candidate.joblib', 'MODEL_CARD.md'):
        with (root / name).open('rb') as handle:
            manifest[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
