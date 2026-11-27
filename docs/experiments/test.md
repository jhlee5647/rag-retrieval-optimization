# test 확인

dev로 구성을 정한 뒤, test 250문항으로 세 구성의 검색을 한 번씩 돌렸다. 규칙은 `docs/optimization.md`의 "평가 문항 분할"을 따른다. test 결과를 보고 구성을 고치지 않았다.

- 실행: `configs/exp/test-naive.yaml`(`r0-naive`와 같음), `test-hybrid-rerank.yaml`(`r1-hybrid-rerank`와 같음), `test-final.yaml`(`r4-expand-select-chunks`와 같음). `EVAL_FILE`을 test 파일로 두고 돌렸다.
- 이 보고서는 검색 지표만 다룬다. 지표는 근거 문서가 있는 235문항으로 잰다.

## 결과

| 지표 | `test-naive` | `test-hybrid-rerank` | `test-final` |
|---|---|---|---|
| Recall@5 | 0.462 | 0.697 | 0.798 |
| Precision@1 | 0.387 | 0.626 | 0.779 |
| MRR@5 | 0.437 | 0.681 | 0.812 |
| nDCG@5 | 0.414 | 0.657 | 0.786 |
| 근거 문서를 전부 가져온 문항 | 0.426 | 0.651 | 0.736 |
| 근거 문서를 하나도 못 가져온 문항 | 0.481 | 0.234 | 0.136 |
| semantic (63문항)의 Recall@5 | 0.333 | 0.571 | 0.714 |
| dev의 Recall@5 (참고) | 0.479 | 0.718 | 0.839 |

- 이웃한 구성끼리의 Recall@5 차이는 `test-naive`에서 `test-hybrid-rerank`가 +0.234 [+0.170, +0.298](오른 문항 77개, 내린 문항 15개), `test-hybrid-rerank`에서 `test-final`이 +0.102 [+0.058, +0.145](40개, 9개)다.
- `test-naive`에서 `test-final`까지는 +0.336 [+0.274, +0.397]이고, 오른 문항 92개, 내린 문항 6개, 그대로인 문항 137개다.
- 세 구성 모두 dev보다 0.02–0.04 낮다. 구성을 dev로 골랐으니 dev 수치가 낙관적인 것은 예상한 방향이다(`docs/experiments/round-4.md`의 "알아 둘 것").

## 알아 둘 것

- `test-naive`와 `test-final`은 2026-11-24에 돌렸다. `test-hybrid-rerank`는 단계마다 수치를 보고하려고 그 결과를 본 뒤 2026-11-25에 더 돌렸다. 이 결과로 구성을 고치지 않았다.
- 세 실행은 main에 남지 않은 작업 브랜치의 커밋(`4ba2340`, `5feeb7b`, `c6745b2`)에서 돌렸다. 그 커밋들과 main(`3061e82`)의 `src/`는 같다.
- `test-final.yaml`은 돌릴 때 답 프롬프트가 main에 없는 것이었다. 저장소에는 답 프롬프트를 `r4-expand-select-chunks`와 같게(`prompts/answer-complete.txt`) 바꿔 넣었다. 검색 부분은 돌린 것과 같다.
- 차이와 구간은 `pair`의 값이다. 전부 가져온 문항과 하나도 못 가져온 문항의 비율은 `scores.jsonl`의 문항별 Recall@5를 임시 스크립트로 센 값이다.
- Recall@k는 청크가 아니라 중복을 뺀 문서로 센다.
