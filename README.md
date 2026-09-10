# Daycon_AI_Hackathon
# 스트레스 지수 예측 AI 해커톤

[초격차] AI 헬스케어 6기 해커톤 — 신체 정보, 수면 패턴, 활동량 등 건강 데이터를 활용해 개인의 스트레스 점수를 예측하는 AI 모델 개발 프로젝트입니다.

## 대회 개요
- 주최: 데이콘
- 트랙: [초격차] AI 헬스케어 머신러닝 트랙
- 평가지표: MAE (회귀)
- 대회 링크: https://dacon.io/competitions/official/236764/overview/description

## 일정
| 날짜 | 내용 |
|---|---|
| 09.09 | 대회 시작 |
| 09.18 | 팀 병합 마감 |
| 09.18 | 대회 종료 |

## 팀원
| 이름 | GitHub |
|---|---|
| 김승인 | [@SeunginKimT](https://github.com/SeunginKimT) |
| 이형준 | [@neohj87](https://github.com/neohj87) |

## 프로젝트 구조
├── data/           # train, test 데이터 (git에는 미포함)
├── notebooks/      # EDA, 실험용 노트북
├── src/            # 전처리, 피처엔지니어링, 모델 코드
├── models/         # 학습된 모델 저장
├── submissions/    # 제출 파일
└── requirements.txt


## 개발 환경 설치
```bash
git clone <https://github.com/SeunginKimT/Daycon_AI_Hackathon>
cd Daycon_AI_Hackathon
uv venv
source venv/bin/activate
uv pip install -r requirements.txt
```
## 진행 상황
