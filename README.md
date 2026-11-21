# config 기반 RAG 검색·생성·채점 실행 환경

config 파일 하나에 RAG 파이프라인을 적고, 그대로 검색·생성·채점을 돌려 결과를 실행마다 폴더에 남기는 실행 환경이다. 데이터를 장비 밖으로 보내지 않도록 모델을 모두 로컬 서버로 띄우고, 외부 접속을 코드에서 막는다.

지금 들어 있는 구성은 naive RAG(임베딩 검색 → 답 생성) 하나다. 설계는 [docs/design.md](docs/design.md)에 있다.

## 환경 준비

- Ubuntu, GPU 48GB 한 장(RTX A6000에서 확인), docker와 nvidia-container-toolkit
- docker 권한. 실행 명령도 서버의 모델을 확인하려고 docker를 부른다: `sudo usermod -aG docker $USER` 뒤 다시 로그인
- uv (Python 3.12는 uv가 맞춘다)
- 문서 폴더(`*.txt`, 파일 하나가 문서 하나)와 평가 JSONL 파일

```
uv sync
cp .env.example .env
```

`.env`에서 경로 네 개를 장비에 맞게 고친다. 전송을 끄는 나머지 값은 바꾸지 않는다. 빠지거나 값이 다르면 패키지를 불러오지 못한다.

| 변수 | 뜻 |
|---|---|
| `DOCS_DIR` | 문서 폴더. 바로 아래의 `*.txt`만 읽는다 |
| `EVAL_FILE` | 평가 JSONL 파일 |
| `HF_HOME` | 모델 파일 폴더 |
| `QDRANT_STORAGE` | Qdrant 저장 폴더 |

모든 명령은 저장소 루트에서 `uv run --env-file .env python -m rag.<명령>`으로 실행한다.

## 서버

서버는 [configs/servers.yaml](configs/servers.yaml)의 정의대로 docker 컨테이너로 뜬다. 포트는 `127.0.0.1`에만 열린다.

| 역할 | 하는 일 | 포트 | 띄우는 데 |
|---|---|---|---|
| `qdrant` | 벡터 DB | 6333 | 약 1분 30초 |
| `embed` | 임베딩 모델 | 8003 | 약 2분 |
| `gen` | 생성 모델 | 8001 | 약 5분 |
| `judge` | 채점 모델 | 8002 | 약 3분 40초 |

```
uv run --env-file .env python -m rag.serve download <역할>  # 이미지와 고정 리비전의 모델 파일을 받는다. 외부 접속은 이 명령만 한다
uv run --env-file .env python -m rag.serve start <역할>  # 띄우고 요청을 받을 준비가 될 때까지 기다린다
uv run --env-file .env python -m rag.serve stop <역할>  # 내린다. 기동 실패 로그를 볼 수 있게 컨테이너는 남긴다
```

GPU 한 장에 `gen`과 `judge`는 함께 뜨지 않는다. `qdrant`와 `embed`는 띄워 둔 채, 생성할 때는 `gen`을, 채점할 때는 `judge`를 바꿔 띄운다. 한 번에 하나씩 띄운다.

## 실행 순서

시간은 평가 문항 500개 기준이다.

```
# 1. 벡터 DB와 임베딩 서버를 띄운다 (약 3분 30초)
uv run --env-file .env python -m rag.serve start qdrant && uv run --env-file .env python -m rag.serve start embed

# 2. 평가 문항마다 검색하고 검색 지표를 낸다. 결과는 runs/retrieve/naive/<시각>/ (약 2분)
uv run --env-file .env python -m rag.retrieve configs/naive.yaml

# 3. 생성 모델 서버를 띄운다 (약 5분)
uv run --env-file .env python -m rag.serve start gen

# 4. 2의 검색 결과로 문항마다 답을 만든다. 결과는 runs/generate/naive/<시각>/ (약 15분)
uv run --env-file .env python -m rag.generate configs/naive.yaml --contexts runs/retrieve/naive/<시각>

# 5. 생성 모델 서버를 내리고 채점 모델 서버를 띄운다. 둘은 GPU에 함께 올라가지 않는다 (약 4분)
uv run --env-file .env python -m rag.serve stop gen && uv run --env-file .env python -m rag.serve start judge

# 6. 4의 답을 채점해 그 생성 실행 폴더에 점수를 쓴다 (약 26분)
uv run --env-file .env python -m rag.score runs/generate/naive/<시각>

# 7. 지금까지의 실행들을 검색끼리, 생성끼리 표로 비교한다
uv run --env-file .env python -m rag.compare
```

- `retrieve`에 `--limit N`을 주면 평가 파일의 앞 N문항만 돌린다. 몇 문항으로 먼저 확인할 때 쓴다.
- 각 명령은 시작할 때 떠 있는 서버의 모델·리비전이 config와 같은지 확인하고, 다르면 띄울 명령을 알려 주며 멈춘다.
- `retrieve`는 config의 청크·임베딩 조건과 문서 수로 색인 이름을 계산한다. 같은 조건의 색인이 없으면 문서를 잘라 새로 만든다. 문서 1,000개에 약 2분이고, 문서 전체는 하루 가까이 걸린다.
- `generate`는 config의 검색 조건이 `--contexts`의 검색 실행과 다르면 거부한다. 검색 결과를 고정한 채 생성 구성만 바꿔 비교하는 용도다.
- `score`는 그 생성 실행 폴더에 점수를 쓴다. 이미 채점한 폴더는 덮어쓰지 않는다.
- `split`(`python -m rag.split configs/naive.yaml`)은 `EVAL_FILE`을 문항 유형별로 반씩 나눠 그 옆에 `<이름>-dev.jsonl`과 `<이름>-test.jsonl`을 쓴다. 구성을 고르는 문항과 확인하는 문항을 나눌 때 쓰고, 나눈 뒤 `.env`의 `EVAL_FILE`을 쓸 파일로 바꾼다.

## 결과 읽는 법

실행마다 `runs/<명령>/<config 이름>/<실행 시각>/`이 새로 생긴다.

| 파일 | 내용 |
|---|---|
| `config.yaml` | 실행에 쓴 config 사본. 생성 실행에는 프롬프트 틀 사본(`answer.txt`)도 있다 |
| `run.json` | 명령, git 커밋, 장비 이름, 문서 수, 평가 파일 해시, 색인 이름, 확인한 서버 모델, 소요 시간 |
| `contexts.jsonl` | (검색) 문항마다 가져온 청크와 문서 ID |
| `results.jsonl` | (생성) 문항마다 답, 청크, 문서 ID |
| `scores.jsonl` | 문항별 점수 |
| `metrics.json` | 평균과 나눠 낸 평균, 평균에 든 문항 수 |

검색 지표(`retrieve`)는 평가 파일의 근거 문서와 대조한 Recall, Precision, MRR, nDCG다. 근거 문서가 지정되지 않은 문항은 뺀다. 리트리버를 고를 때는 Recall@k를 주로 본다. 근거 문서가 1개인 문항이 많아 Precision@k는 상한이 낮다.

생성 지표(`score`)는 RAGAS의 세 지표다.

| 지표 | 보는 것 |
|---|---|
| `faithfulness` | 답의 주장이 가져온 청크에 근거하는가 |
| `response_relevancy` | 답이 질문에 맞는 내용인가. "모른다"는 답은 0점이다 |
| `factual_correctness` | 답이 정답과 사실 단위로 맞는가 (F1) |

- `by_hit`은 근거 문서를 전부(`all`)·일부(`some`) 가져왔는지, 못 가져왔는지(`none`), 지정되지 않았는지(`no_gold`)로 나눈 평균이다. 검색이 문제인지 생성이 문제인지 가를 때 본다.
- `by_type`은 문항 유형별 평균이다. 답이 없는 유형(`info_not_found`)의 낮은 `response_relevancy`는 올바르게 거절한 결과다.
- 채점 모델이 답을 내지 못한 점수는 0으로 채우지 않고 비운다. `empty`가 그 수다.
- 같은 실행을 다시 돌려도 점수가 조금 다를 수 있다. 임베딩과 채점 모델의 미세한 흔들림 때문이다. 작은 차이는 우연일 수 있다.

`compare`는 `runs/` 아래의 실행을 검색 실행끼리, 생성 실행끼리 표로 보여 준다.

## config

[configs/naive.yaml](configs/naive.yaml)이 예시다. 정의되지 않은 항목(오타 포함)이나, 만들 수 없는 조건(겹침 ≥ 청크 크기, 문맥이 생성 모델 한도를 넘음 등), 로컬이 아닌 서버 주소는 실행 전에 거부한다. `generation`, `generate`, `judge`는 함께 적거나 함께 뺀다. 빼면 검색만 하는 config다.

## 새 노드를 더하는 법

파이프라인의 단계 하나가 노드 하나다. config의 `retrieve`와 `generate`에 적힌 순서대로 노드가 이어진다.

1. `src/rag/nodes/`에 파일을 하나 둔다. 이 폴더의 파일은 자동으로 불러온다.
2. `@node("이름")`으로 등록한다. 노드는 config의 인자와 자원을 받아, State를 받아 바뀐 필드만 돌려주는 함수를 만든다.

```python
from typing import Any

from rag.config import Strict
from rag.pipeline import Resources, State, node


class TruncateParams(Strict):  # 인자 오타를 거부하려고 Strict를 쓴다
    keep: int


@node("truncate")
def truncate(params: dict[str, Any], resources: Resources):
    p = TruncateParams.model_validate(params)

    def run(state: State) -> dict[str, Any]:
        return {"contexts": state["contexts"][: p.keep], "doc_ids": state["doc_ids"][: p.keep]}

    return run
```

3. config에 적는다.

```yaml
retrieve:
  - search: {mode: dense, top_k: 10}
  - truncate: {keep: 5}
```

State의 필드는 `question`, `contexts`, `doc_ids`, `answer`다([src/rag/pipeline.py](src/rag/pipeline.py)). LangGraph는 여기 없는 필드를 오류 없이 버리므로, 새 필드가 필요하면 `State`에 먼저 더한다. 검색 단계의 노드는 `resources.vectorstore`를, 생성 단계의 노드는 `resources.llm`을 쓴다.

## 테스트

GPU와 서버 없이 돈다(가짜 임베딩, 가짜 생성 모델, 메모리 안의 Qdrant).

```
uv run --env-file .env.example pytest
uv run ruff check
uv run ruff format --check
```

## 유출 방지

- 사용 통계·추적을 끄는 환경변수가 빠지면 패키지를 불러오지 못한다.
- 이 프로세스가 `127.0.0.1` 밖으로 접속하려 하면 오류를 낸다. 내려받기는 docker가 한다.
- config의 서버 주소는 `127.0.0.1`과 `localhost`만 받는다.
- 컨테이너의 외부 송신은 코드로 막지 못한다. 확실히 막으려면 장비 방화벽에서 외부 송신을 차단한다.
