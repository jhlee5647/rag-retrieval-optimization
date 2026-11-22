# 설계

config 파일 하나에 RAG 파이프라인의 명세를 적고, 명령으로 그 명세대로 검색·생성·채점을 돌려 결과를 config와 함께 남기는 실행 환경이다.

지금 동작하는 구성은 naive RAG(임베딩 검색 뒤 답 생성) 하나다. 리랭킹이나 혼합 검색 같은 기법은 구현하지 않고, 노드 파일 하나와 config 한 줄로 끼워 넣을 수 있는 뼈대만 둔다.

작업 순서와 상태는 `docs/plan.md`, 작업 방식은 `CLAUDE.md`에 있다.

## 전제

- 문서 원본과 평가 데이터는 이미 있다. 그대로 쓰고 변환하거나 정제하지 않는다.
- 문서는 텍스트 파일이다. 파일 하나가 문서 하나이고, 확장자를 뺀 파일 이름이 문서 ID다.
- 문서 출처는 하나라고 가정한다. 언어는 영어를 기준으로 한다.
- 데이터는 장비 밖으로 내보낼 수 없다. 임베딩, 생성, 채점 모델을 모두 로컬에서 돌리고 외부 API를 부르지 않는다.
- 장비는 네트워크에 연결돼 있다. 그래서 유출은 코드와 설정으로 막는다.
- 파이프라인은 LangChain과 LangGraph로 만든다.
- 실행 장비는 Ubuntu, RTX A6000 48GB 한 장, docker다.

도구는 범용으로 쓰되, 첫 적용 대상은 EnterpriseRAG-Bench다.

| 항목 | 값 |
|---|---|
| 문서 | `dsid_<32자리 16진수>.txt` 511,957개. 한 폴더에 하위 폴더 없이 들어 있다 |
| 평가 파일 | JSONL 500문항. 필드는 `id`, `type`, `question`, `reference`, `answer_facts`, `expected_doc_ids`, `source_types` |
| 지정된 근거 문서 수 | 0개 30문항, 1개 377문항, 최대 10개 |

근거 문서는 평가 데이터가 문항마다 "이 문서를 가져와야 맞다"고 지정해 둔 문서다. 평가 파일의 `expected_doc_ids`에 적혀 있다.

근거 문서가 지정되지 않은 30문항은 두 유형이다. 관련 문서가 없다는 뜻이 아니라, 벤치마크가 정답 문서를 지정하지 않았다는 뜻이다.

| 유형 | 지정하지 않은 이유 |
|---|---|
| high-level | 여러 문서를 종합해야 답할 수 있는 질문이다. 답의 근거가 많은 문서에 흩어져 있고 어느 문서들을 가져와도 답에 이를 수 있어서, 벤치마크가 특정 문서를 정답으로 지정하지 않았다 |
| unanswerable | 문서에 답이 없는 질문이다. "답할 수 없다"가 정답이라 지정할 문서가 없다 |

## 실행 흐름

명령은 단계별로 나뉜다. 생성 모델과 채점 모델이 GPU 한 장에 함께 뜨지 않아서, 단계 사이에 서버를 바꿔야 하기 때문이다.

| 명령 | 하는 일 | 필요한 서버 |
|---|---|---|
| `retrieve <config>` | 색인 확인, 검색, 검색 지표 계산 | qdrant, embed |
| `generate <config> --contexts <검색 실행 폴더>` | 고정한 검색 결과로 답 생성 | gen |
| `score <생성 실행 폴더>` | 생성 지표 채점 | judge, embed |
| `compare` | 실행별 지표 표 출력 | 없음 |
| `serve download\|start\|stop <역할>` | 서버 내려받기, 띄우기, 내리기 | docker |

모든 명령은 저장소 루트에서 `uv run --env-file .env python -m rag.<명령>`으로 실행한다.

`retrieve`는 다음 순서로 동작한다.

1. `.env`에서 문서 폴더와 평가 파일 경로를 읽고 config를 검사한다.
2. 색인 키를 계산한다. 같은 조건으로 만들어 둔 색인이 Qdrant에 있으면 그대로 쓰고, 없으면 문서를 청킹·임베딩해 만든다.
3. 문항마다 검색 그래프를 돌린다.
4. 검색 지표를 계산해 실행 폴더에 저장한다.

이 구조는 리트리버를 먼저 최적화하고 생성자를 나중에 최적화하는 진행 방식에 맞춘 것이다. 검색 실험은 LLM 없이 검색 지표만 보고 반복한다. 고른 리트리버의 검색 결과를 고정한 채 생성 구성만 바꿔 생성 지표를 본다.

서버 교체는 사람이 한다. 실행 명령은 서버를 띄우거나 내리지 않고, 떠 있는 서버의 모델이 config와 같은지만 확인한다. 다르면 실행을 거부하고 띄울 명령을 알려 준다.

vLLM API는 모델 이름만 알려 주므로, config 주소의 포트를 연 컨테이너의 실행 인자에서 `--model`과 `--revision`을 읽어 대조한다. 그래서 실행 명령도 docker 권한이 필요하다. 대조한 서버는 `run.json`에, `score`는 `metrics.json`에 적는다.

## 디렉터리 구조

```
rag/
├─ CLAUDE.md, README.md, pyproject.toml, uv.lock
├─ .env.example          데이터 경로, 유출 방지 환경변수
├─ configs/
│  ├─ naive.yaml         파이프라인 명세
│  └─ servers.yaml       서버 정의: qdrant, embed, gen, judge
├─ prompts/answer.txt    답 생성 프롬프트 틀
├─ docs/design.md, docs/plan.md
├─ src/rag/
│  ├─ __init__.py        환경변수 검사, 외부 접속 차단
│  ├─ config.py          config 스키마, 실행 전 검사
│  ├─ data.py            문서 폴더와 평가 데이터 읽기 (필드 이름 매핑)
│  ├─ index.py           색인 키, 재사용 확인, 새로 만들기
│  ├─ pipeline.py        State, 노드 등록, 그래프 조립
│  ├─ nodes/             search.py, answer.py
│  ├─ metrics.py         검색 지표, 집계
│  ├─ runs.py            실행 폴더, run.json
│  ├─ retrieve.py, generate.py, score.py, compare.py
│  └─ serve.py           서버 download | start | stop
├─ tests/
└─ runs/                 실행 결과. git에서 제외한다
```

파일은 그 파일을 처음 쓰는 단위에서 만든다. 미리 만들어 두지 않는다.

## 레이어

| 레이어 | 파일 | 하는 일 |
|---|---|---|
| 명령 | `retrieve`, `generate`, `score`, `compare`, `serve` | 인자 해석, 실행 폴더 입출력, 아래 레이어 호출 |
| 파이프라인 | `pipeline.py`, `nodes/` | 노드 등록, config의 노드 목록을 일직선 그래프로 조립 |
| 색인 | `index.py` | 색인 키, 재사용 확인, 새로 만들기 |
| 평가 | `metrics.py`, `score.py`의 RAGAS 호출부 | 지표 계산과 집계 |
| 실행 기록 | `runs.py`, `data.py` | 실행 폴더, 데이터 읽기 |
| 설정과 가드 | `config.py`, `__init__.py` | 실행 전 검사, 유출 방지 |
| 저장소 밖 | Qdrant, vLLM 컨테이너 | `configs/servers.yaml`로 정의 |

명령만 다른 레이어를 부른다. 노드는 명령을 모르고, 평가는 파이프라인을 모른다.

## config

config 파일 하나가 검색, 생성, 채점의 전체 명세다.

```yaml
index: {url: http://127.0.0.1:6333, exact: true}
chunking: {size: 512, overlap: 64}
embedding:
  base_url: http://127.0.0.1:8003/v1
  model: Qwen/Qwen3-Embedding-4B
  revision: 5cf2132abc99cad020ac570b19d031efec650f2b
  max_input_tokens: 8192
  query_instruction: Given a web search query, retrieve relevant passages that answer the query
eval:
  fields: {id: id, question: question, answer: reference, doc_ids: expected_doc_ids, type: type}
  k: [1, 5]
retrieve:
  - search: {mode: dense, top_k: 5}
generation:
  base_url: http://127.0.0.1:8001/v1
  model: Qwen/Qwen3.8-27B-FP8
  revision: c3a462cc3d38cf351f3b26cbd763469eeb74ef81
  max_model_len: 32768
  max_tokens: 512
  temperature: 0
generate:
  - answer: {prompt: prompts/answer.txt}
judge:
  base_url: http://127.0.0.1:8002/v1
  model: RedHatAI/gemma-4-26B-A4B-it-NVFP4
  revision: 5557756b8dce33ac72f2bd702b11729fdba3b839
```

| 항목 | 뜻 |
|---|---|
| `index` | Qdrant 주소. `exact: true`면 근사(HNSW)가 아닌 정확 검색을 한다 |
| `chunking` | 청크 크기와 겹침. 단위는 임베딩 모델 토크나이저의 토큰 수다 |
| `embedding` | 임베딩 서버, 모델, 고정 리비전. `query_instruction`은 질문에만 붙이는 지시문이다. 기본값은 임베딩 모델 카드의 예시 문장이다. 문서 청크에는 붙이지 않으므로 바꿔도 색인을 다시 만들지 않는다 |
| `eval.fields` | 왼쪽은 이 도구가 쓰는 이름, 오른쪽은 평가 파일의 필드 이름이다 |
| `eval.k` | 검색 지표를 잴 순위 목록 |
| `retrieve` | 검색 단계의 노드 목록. 항목마다 `{노드 이름: 인자}` 하나다 |
| `generation` | 생성 서버, 모델, 고정 리비전, 토큰 한도, 온도 |
| `generate` | 생성 단계의 노드 목록 |
| `judge` | 채점 서버, 모델, 고정 리비전 |

`generation`, `generate`, `judge`는 함께 적거나 함께 뺀다. 빼면 검색만 하는 config가 된다.

`generate`는 config의 검색 부분(`index`, `chunking`, `embedding`, `eval`, `retrieve`)이 `--contexts`로 받은 검색 실행의 config 사본과 같은지 대조하고, 다르면 거부한다. 생성 실행에 남는 config 사본이 실제로 쓰인 검색 조건과 어긋나는 일을 막기 위해서다.

`score`는 대상 실행 폴더의 config 사본에서 `judge`와 `embedding`을 읽는다.

지표를 고르는 설정은 두지 않는다. 구현된 지표는 항상 전부 계산한다.

### 실행 전 검사

오래 걸리는 일을 시작하기 전에 다음을 거부한다.

- config 파일에 정의되지 않은 항목. 오타로 적은 항목이 조용히 무시되는 일을 막는다. 예를 들어 `overlap`을 `overlp`로 적으면 거부한다
- 등록되지 않은 노드 이름
- `chunking.overlap >= chunking.size`, `chunking.size > embedding.max_input_tokens`
- `top_k × chunking.size + max_tokens + 1024 > max_model_len`. 문맥이 생성 모델 한도를 넘는 경우다. 1024는 프롬프트 틀, 질문, 채팅 템플릿의 여유분이고, 청크 크기를 임베딩 토크나이저로 잰 데서 오는 차이도 여기에 포함한다
- `eval.k`의 최댓값이 검색 노드가 가져오는 문맥 수보다 큰 경우
- `127.0.0.1`이나 `localhost`가 아닌 서버 주소
- 서버에 떠 있는 모델이 config와 다른 경우

## 데이터 입력

데이터 경로는 장비마다 달라서 config가 아니라 `.env`에 둔다. `.env`는 git에서 제외하고 `.env.example`만 커밋한다.

| 변수 | 뜻 |
|---|---|
| `DOCS_DIR` | 문서 폴더. 바로 아래의 `*.txt`만 읽는다 |
| `EVAL_FILE` | 평가 JSONL 파일 |
| `HF_HOME` | 모델 파일 폴더. 모델 서버 컨테이너에 같은 경로로 붙인다 |
| `QDRANT_STORAGE` | Qdrant 저장 폴더 |

입력 형식은 "문서는 `.txt` 파일들, 평가는 JSONL"로 고정한다. 평가 파일의 필드 이름만 `eval.fields`로 맞춘다. 문서 ID 규칙을 바꾸는 설정은 두지 않는다.

`retrieve --limit N`은 평가 파일의 앞 N문항만 실행한다. `.env`는 전체 평가 파일을 가리키게 둔 채 일부만 돌려 볼 때 쓴다.

## 색인

### 색인 키

색인 이름은 `index-<해시 앞 16자>`다. 해시에는 다음이 들어간다.

- 색인 형식 상수: 형식 번호, 거리 함수, HNSW 설정, BM25 설정
- 청크 크기, 겹침, 청킹 방법(`RecursiveCharacterTextSplitter`)
- 임베딩 모델 이름과 리비전
- 문서 폴더의 `*.txt` 파일 수

문서 내용은 읽지 않는다. 51만 개 문서의 내용 지문을 실행마다 계산하면 약 6분이 걸리기 때문이다. 그 대신 문서 내용이 수정돼도 감지하지 못한다. 문서 파일 수는 다른 문서 집합을 같은 색인으로 잘못 쓰는 일을 막으려고 넣었다.

문서 파일이 늘거나 줄면 키가 달라져 새 색인을 만들기 시작한다. 문서 전체는 하루 가까이 걸리므로, 만들기 전에 문서 수와 함께 새로 만든다는 메시지를 출력한다.

### 재사용과 새로 만들기

Qdrant의 용어 세 가지를 쓴다.

| 용어 | 뜻 |
|---|---|
| 컬렉션 | Qdrant가 벡터를 담는 단위. 데이터베이스의 테이블에 해당한다 |
| 점 | 컬렉션에 저장되는 항목 하나. 여기서는 청크 하나다 |
| 별칭 | 컬렉션에 붙이는 또 하나의 이름. 별칭으로도 그 컬렉션을 검색할 수 있다 |

- 색인 이름은 컬렉션의 별칭으로 붙인다.
- 별칭은 색인을 끝까지 만든 뒤에만 붙인다. 그래서 별칭이 있으면 완성된 색인이고, 완성된 색인만 재사용한다.
- 새로 만드는 컬렉션의 이름은 `data-<같은 16자>`다.
- 그 이름의 컬렉션이 별칭 없이 남아 있으면 만들다 중단된 것이다. 지우고 처음부터 다시 만든다.
- 다 넣은 뒤 점 수가 청크 수와 같은지 확인하고, 같을 때만 별칭을 붙인다.

### 기존 색인

장비에 이미 만들어 둔 색인을 다시 만들지 않고 쓴다.

| 항목 | 값 |
|---|---|
| 컬렉션 | `data-d89aa11508031fc4` |
| 점 수 | 1,898,784 |
| 조건 | 청크 512, 겹침 64, `Qwen/Qwen3-Embedding-4B`의 위 리비전, 문서 511,957개 |

이 컬렉션에 새 키의 별칭을 한 번 붙인다. 개발자가 직접 실행한다. 별칭을 붙이기 전에 장비에서 `retrieve`를 돌리면 문서 전체의 색인을 새로 만들기 시작한다.

### 색인 형식

아래 값은 기존 색인을 만든 규칙이다. 새로 만드는 색인도 같은 규칙을 따라야 노드가 어느 색인에서나 똑같이 동작한다.

| 항목 | 값 |
|---|---|
| 밀집 벡터 | 이름 없는 기본 벡터. 크기는 임베딩 모델을 따르고(지금 모델은 2560), 거리는 Cosine |
| HNSW | `m: 16`, `ef_construct: 100` |
| 희소 벡터 | 이름 `bm25`, IDF 수정자 사용 |
| BM25 토큰 | 소문자로 바꾼 뒤 `\w+`로 자른 단어. 단어의 CRC32가 색인 번호다 |
| BM25 가중치 | 문서는 `f × (k1 + 1) / (f + k1)`, `k1 = 1.5`, 길이 보정 없음(`b = 0`). 질문은 단어마다 1 |
| 점 ID | `uuid5(NAMESPACE_URL, "<문서 ID>:<청크 번호>")` |
| payload | `{"page_content": <청크>, "metadata": {"doc_id": <문서 ID>, "chunk": <청크 번호>}}` |
| 청킹 | `RecursiveCharacterTextSplitter`. 길이는 임베딩 모델 토크나이저로 특수 토큰을 빼고 센 토큰 수 |
| 문서 읽기 | 파일 이름 순서, UTF-8 |

naive RAG는 BM25 벡터를 쓰지 않는다. 그래도 새 색인에 넣는 이유는, 나중에 더하려면 문서 전체를 다시 색인해야 하고 기존 색인에는 이미 들어 있기 때문이다. BM25 벡터는 직접 계산하므로 추가 의존성이 없다.

## 파이프라인과 노드

기법 하나가 노드 하나다. config의 `retrieve`와 `generate`에 노드 이름과 인자를 순서대로 적으면, 그 순서대로 이은 LangGraph의 일직선 그래프가 된다.

- 노드는 파이프라인의 한 단계를 맡는 함수다. 형태는 `(params, resources) → (State → 바뀐 필드만 담은 dict)`다.
- `params`는 config에 적은 인자다. `resources`는 config, 그리고 검색 단계의 벡터 저장소나 생성 단계의 생성 모델이다.
- 노드는 등록해서 쓴다. 등록은 노드에 config에서 부를 이름을 붙여 두는 일이다. `@node("이름")`로 등록하고, config에 그 이름을 적으면 해당 노드가 실행된다. `nodes/` 폴더에 파일을 두면 자동으로 불러온다.
- State는 노드들이 차례로 읽고 쓰는 공용 데이터다. 기본 필드는 `question`, `contexts`, `doc_ids`, `answer`다. LangGraph는 State에 없는 필드를 오류 없이 버리므로, 노드가 새 필드를 쓰려면 `pipeline.py`의 State에 먼저 더한다.
- 그래프 안의 노드 이름에는 순번을 붙인다(`0-search`). LangGraph가 State 필드와 같은 노드 이름을 거부하고, 같은 노드를 두 번 넣을 수도 있어야 해서다.

지금 만드는 노드는 둘이다.

| 노드 | 인자 | 하는 일 |
|---|---|---|
| `search` | `mode`(`dense`, `bm25`, `hybrid`), `top_k`, `candidates`(`hybrid`에서만) | 질문으로 청크를 찾아 `contexts`와 `doc_ids`를 채운다 |
| `answer` | `prompt`(틀 파일 경로) | 틀의 `{context}`와 `{question}`을 채워 생성 모델을 부르고 `answer`를 채운다 |

- `dense`는 질문을 임베딩해 가까운 청크를 찾는다. 질문은 `Instruct: <query_instruction>\nQuery:<질문>` 형식으로 임베딩한다. 문서 청크는 지시문 없이 임베딩돼 있다.
- `bm25`는 색인의 BM25 벡터로 질문의 단어가 든 청크를 찾는다. 임베딩 모델을 부르지 않는다.
- `hybrid`는 두 검색이 각각 `candidates`개를 가져오고, 두 순위를 RRF로 합쳐 `top_k`개를 고른다. 한 검색에서 n위인 청크가 `1 / (60 + n - 1)`점을 받는다.
- 문맥은 `\n\n---\n\n`으로 이어 붙인다.
- 프롬프트는 영어로 쓰고, 문맥에만 근거해 답하고 문맥에 답이 없으면 없다고 말하게 한다.
- 문항은 동시에 8개씩 처리한다. 설정으로 두지 않는다.

이 방식은 일직선만 표현한다. 검색 결과를 보고 다시 검색하는 식의 분기나 반복이 필요해지면 그때 설계를 다시 한다.

## 평가

### 검색 지표

근거 문서 ID와 대조해 계산하므로 LLM이 필요 없다. `retrieve`가 검색 직후에 계산한다.

가져온 청크를 문서 ID로 바꾸고, 처음 나온 것만 남긴 목록을 순위 목록 `R`이라 한다. 근거 문서 ID의 집합을 `G`라 한다. `eval.k`의 `k`마다 다음을 계산한다.

| 지표 | 정의 |
|---|---|
| Recall@k | `R`의 앞 k개 중 `G`에 속한 수 ÷ `G`의 크기 |
| Precision@k | `R`의 앞 k개 중 `G`에 속한 수 ÷ k |
| MRR@k | `R`의 앞 k개에서 `G`에 속한 첫 문서의 순위의 역수. 없으면 0 |
| nDCG@k | 관련 여부를 0과 1로 본 DCG@k ÷ 이상적인 순서의 DCG@k |

근거 문서가 지정되지 않은 문항은 검색 지표에서 뺀다. 대조할 정답 문서가 없어 점수를 계산할 수 없기 때문이다. 검색 자체는 모든 문항에 실행한다. 생성 단계가 그 문항의 문맥도 필요로 하기 때문이다. 평균에 들어간 문항 수와 빠진 문항 수를 함께 기록한다.

### 생성 지표

RAGAS의 세 지표를 쓴다. 채점 LLM은 `judge` 서버, 임베딩은 `embed` 서버다.

| 지표 | 보는 것 | 쓰는 입력 |
|---|---|---|
| Faithfulness | 답의 주장이 가져온 문맥에 근거하는가 | 답, 문맥 |
| Response Relevancy | 답이 질문에 맞는 내용인가 | 질문, 답 |
| Factual Correctness | 답이 정답과 사실 단위로 일치하는가 | 답, 정답 |

채점 모델이 출력 형식을 못 맞추면 그 문항의 점수가 빈다. 빈 점수의 수를 지표마다 기록한다.

- Factual Correctness는 F1 하나로 본다.
- Response Relevancy는 "모른다" 같은 회피형 답에 0점을 준다. 그래서 unanswerable 유형의 낮은 점수는 올바르게 답을 거절한 결과로 읽는다.

### 집계

전체 평균 하나로는 어디를 고칠지 알 수 없어서 나눠서도 평균을 낸다.

- **문항 유형별**: `eval.fields.type`으로 나눈다. 검색 지표와 생성 지표 모두에 적용한다.
- **검색 적중별**: 생성 지표에만 적용한다. 문항마다 근거 문서를 전부 가져왔는지, 일부만 가져왔는지, 못 가져왔는지, 근거 문서가 지정되지 않은 문항인지를 적고 그 구분대로 평균을 낸다.

| 보이는 현상 | 뜻하는 것 | 처방 후보 |
|---|---|---|
| 작은 k와 큰 k의 Recall이 모두 낮다 | 근거 문서를 후보로도 못 가져온다 | 혼합 검색, 질문 다시 쓰기, 임베딩 교체 |
| 작은 k의 Recall은 낮고 큰 k는 높다 | 가져왔지만 순위가 낮다 | 리랭킹 |
| 근거 문서를 가져왔는데 답이 틀리고 Faithfulness는 높다 | 맞는 문서의 엉뚱한 부분을 가져왔다 | 청크 크기, 문맥 수 |
| 근거 문서를 가져왔는데 답이 틀리고 Faithfulness도 낮다 | 문맥을 두고 지어낸다 | 프롬프트, 생성 모델 |
| 특정 유형만 낮다 | 유형별 약점 | 질문 분해, "모르면 모른다" 지시 |

큰 k의 Recall을 보려면 `top_k`를 그만큼 크게 둔 검색 실행이 따로 필요하다.

## 실행 기록

결과는 파일로 남긴다. 실험 추적 서버는 쓰지 않는다.

명령을 한 번 실행한 결과는 폴더 하나에 담긴다. 이 폴더를 실행 폴더라 한다.

```
runs/retrieve/<config 이름>/<실행 시각>/
  config.yaml  run.json  contexts.jsonl  scores.jsonl  metrics.json
runs/generate/<config 이름>/<실행 시각>/
  config.yaml  answer.txt  run.json  results.jsonl  scores.jsonl  metrics.json
```

- `<config 이름>`은 config 파일 이름에서 확장자를 뺀 것이다. `configs/naive.yaml`이면 `naive`다.
- `<실행 시각>`은 `YYYYMMDD-HHMMSS` 형식이다. 실행마다 새 폴더가 생기므로 앞선 결과를 덮어쓰지 않는다.
- 예를 들어 `configs/naive.yaml`로 검색하면 `runs/retrieve/naive/YYYYMMDD-HHMMSS/`가 생긴다.

| 파일 | 내용 |
|---|---|
| `config.yaml` | 실행에 쓴 config의 사본 |
| `answer.txt` | 실행에 쓴 프롬프트 틀의 사본 |
| `run.json` | 명령, git 커밋, 문서 수, 평가 파일 해시, 색인 이름, 서버 모델, 소요 시간, `--limit` 값. 생성 실행에는 사용한 검색 실행 폴더도 적는다 |
| `contexts.jsonl` | 문항마다 `id`, `question`, `contexts`, `doc_ids`와 노드가 더한 필드 |
| `results.jsonl` | 문항마다 `id`, `answer`, `contexts`, `doc_ids` |
| `scores.jsonl` | 문항별 점수. 검색 실행은 `retrieve`가, 생성 실행은 `score`가 쓴다 |
| `metrics.json` | 전체 평균, 나눠 낸 평균, 평균에 들어간 문항 수와 빠진 문항 수 |

`compare`는 `runs/` 아래의 `metrics.json`을 읽어 검색 실행끼리, 생성 실행끼리 표를 따로 낸다. 생성 실행 표에는 사용한 검색 실행을 함께 표시한다.

## 서버

서버는 모두 docker 컨테이너로 띄우고, 정의는 `configs/servers.yaml`에 둔다.

| 역할 | 모델 | 포트 | GPU 몫 |
|---|---|---|---|
| `qdrant` | 없음 | 6333 | 없음 |
| `embed` | `Qwen/Qwen3-Embedding-4B` | 8003 | 0.22 |
| `gen` | `Qwen/Qwen3.8-27B-FP8` | 8001 | 0.70 |
| `judge` | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 8002 | 0.66 |

- `gen`과 `judge`는 함께 뜨지 않는다. `embed`는 둘 중 하나와 함께 뜬다.
- 포트는 모두 `127.0.0.1`에만 연다.
- 모델 파일은 `HF_HOME` 아래에 고정 리비전으로 있어야 한다. 실행 중에는 내려받지 않는다.
- Qdrant 저장 폴더는 기존 색인이 들어 있는 폴더를 그대로 쓴다.

정의 전체는 `configs/servers.yaml`에 있다. `judge`는 이 장비에서 생성 서버로 띄워 채점에도 썼던 값에서 포트와 `--max-model-len`(65536 → 32768)만 바꿨다.

`serve start <역할>`은 컨테이너 `rag-<역할>`을 띄우고 상태 확인 주소가 200을 줄 때까지 기다린다. 멈춘 컨테이너는 로그를 볼 수 있게 남겨 두고, 다시 띄울 때 지운다.

## 유출 방지

코드에 네 겹을 둔다.

1. **주소 검사**: config의 서버 주소는 `127.0.0.1`과 `localhost`만 받는다.
2. **전송 스위치 검사**: 사용 통계와 추적 전송을 끄는 환경변수가 빠지거나 값이 다르면 패키지를 불러오지 못한다. 어떤 명령이든 패키지를 먼저 불러오므로 한 곳에서 검사한다.
3. **로컬 파일만 읽기**: 모델과 토크나이저는 로컬 파일만 읽는다. 내려받기는 `serve download`에서만 한다.
4. **외부 접속 차단**: 이 프로세스가 `127.0.0.1` 밖으로 접속하려 하면 오류를 낸다. 풀지 않는다. `serve download`는 이미지를 `docker pull`로, 모델을 이미지 안의 `hf download`로 받아, 외부와 통신하는 것은 docker뿐이다.

2번은 목록에 적은 스위치만 막는다. 의존성이 많아 목록에 없는 전송이 있을 수 있고, 4번이 그것을 막는다. 테스트도 같은 차단이 걸린 채로 돈다.

환경변수 목록은 다음과 같다. 설치된 의존성이 실제로 읽는 것만 적었다.

```
DO_NOT_TRACK=1
HF_HUB_DISABLE_TELEMETRY=1
LANGSMITH_TRACING_V2=false
LANGCHAIN_TRACING_V2=false
LANGSMITH_TRACING=false
LANGCHAIN_TRACING=false
RAGAS_DO_NOT_TRACK=true
```

- langsmith는 추적 여부를 `LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2`, `LANGSMITH_TRACING`, `LANGCHAIN_TRACING` 순서로 보고, 처음 값이 있는 것을 따른다. 그래서 넷 다 끈다.
- `LANGGRAPH_CLI_NO_ANALYTICS`는 langgraph-cli가 읽는데, langgraph-cli는 설치돼 있지 않아 뺐다.
- 4번(외부 접속 차단)은 감사 훅으로 `socket.connect`, `socket.sendto`, `socket.sendmsg`를 검사한다. 루프백 주소(`127.0.0.0/8`, `::1`)와 `localhost`만 통과시킨다. Python 소켓을 거치지 않는 C·Rust 확장의 접속(grpcio, hf-xet)은 막지 못한다.

컨테이너는 이 프로세스 밖이라 4번이 닿지 않는다. 컨테이너는 포트를 `127.0.0.1`에만 열고, 통계 전송을 끄고(`VLLM_NO_USAGE_STATS=1`, `QDRANT__TELEMETRY_DISABLED=true`), 오프라인(`HF_HUB_OFFLINE=1`)으로 띄운다. 컨테이너의 외부 송신까지 확실히 막으려면 장비 방화벽에서 차단해야 한다.

## 테스트와 코드 검사

- 자동 테스트는 GPU와 서버 없이 돈다. 결정적인 가짜 임베딩, 가짜 LLM, 메모리 안의 Qdrant를 쓴다.
- 테스트는 `.env.example`의 환경변수를 넣고 돈다. 사용자와 같은 값으로 도는지 확인하기 위해서다.
- RAGAS는 입력을 맞추고 점수를 집계하는 부분만 자동으로 테스트한다. 실제 채점 모델과의 동작은 장비에서 직접 확인한다.
- 개발을 실행 장비에서 하므로, 단위마다 사전 리뷰의 "직접 확인하기"에 실제 서버로 돌리는 명령을 넣는다.
- 린터와 포매터는 ruff다. 규칙은 ruff 기본값에 import 정렬을 더하고, 줄 길이는 100이다.
- 매 단위의 검증에서 `uv run --env-file .env.example pytest`, `uv run ruff check`, `uv run ruff format --check`를 실행한다.
- CI와 커밋 전 훅은 두지 않는다. 단위마다 사전 리뷰에서 위 명령의 출력을 확인한다.

## 주요 기술 선택

| 항목 | 선택 | 고려한 대안 | 선택 이유 |
|---|---|---|---|
| 도구 | Python 3.12, uv, pytest, ruff | pip, poetry | lockfile로 버전을 고정한다 |
| 파이프라인 | LangChain, LangGraph | 전제 | 노드 목록을 일직선 그래프로 조립한다 |
| 확장 방식 | 노드 목록 | 고정 파이프라인에 옵션 분기 | 기법을 더할 때 핵심 코드를 고치지 않는다 |
| 벡터 저장소 | Qdrant (docker) | Chroma, FAISS | 기존 색인을 재사용하고, 밀집·희소 벡터를 한 컬렉션에 둔다 |
| 모델 서빙 | vLLM (docker), OpenAI 호환 API | Ollama, 프로세스 안에서 모델 로딩 | 임베딩, 생성, 채점을 같은 방식으로 띄운다. 기존 색인을 만든 임베딩 경로와 같다 |
| 임베딩 모델 | `Qwen/Qwen3-Embedding-4B` | 없음 | 기존 색인을 만든 모델이다 |
| 채점 모델 | 생성 모델과 다른 모델 | 생성 모델로 채점 | 자기 답을 자기가 채점하는 편향을 피한다. 대신 서버를 바꿔야 한다 |
| 청킹 | 임베딩 토크나이저 토큰 기준 | 문자 수 기준 | 기존 색인과 같고, 모델 입력 한도와 단위가 같다 |
| 색인 키 | 색인 조건과 문서 파일 수 | 문서 내용 전체의 지문 | 실행마다 드는 약 6분을 없앤다 |
| config | YAML과 pydantic 엄격 검사 | Hydra | 오타와 한도 초과를 실행 전에 잡는다 |
| 데이터 경로 | `.env` | config | 장비마다 다른 값을 명세에서 분리한다 |
| 검색 지표 | 직접 구현 | 지표 라이브러리 | 지표마다 열 줄 안팎이라 의존성이 필요 없다 |
| 생성 지표 | RAGAS | 직접 쓴 채점 프롬프트 | 널리 쓰이는 정의를 그대로 쓴다 |
| 결과 저장 | 파일과 `compare` | 실험 추적 서버 | 서버가 없고 유출 경로가 늘지 않는다 |
| 명령 | `python -m rag.<명령>`, argparse | typer, click | 의존성이 필요 없다 |

의존성의 시작 목록은 `langchain-core`, `langchain-openai`, `langchain-qdrant`, `langchain-text-splitters`, `langgraph`, `qdrant-client`, `pydantic`, `pyyaml`, `httpx`, `huggingface-hub`, `tokenizers`, `ragas`이고, 개발용은 `pytest`, `ruff`다. 버전은 스캐폴딩 단위에서 고정한다.

## 장비에서 확인된 사실

구현할 때 그대로 쓰는 값과, 다시 재지 않아도 되는 측정값이다.

- `embed`의 GPU 몫 0.20은 기동에 실패한다. 0.22는 된다.
- 서버는 한 번에 하나씩 띄운다. 동시에 띄우면 GPU 메모리 측정이 겹쳐 실패한다.
- 기동 시간은 `qdrant` 약 1분 30초, `embed` 약 2분, `judge` 약 3분 40초, `gen` 약 5분이다.
- 문서 전체에 대한 정확 검색은 Qdrant 클라이언트의 기본 timeout 5초를 넘는다. 120초로 둔다.
- RAGAS의 채점 모델 출력 한도는 4096토큰으로 둔다. 기본값 1024로는 긴 답의 주장 목록이 잘려 점수가 빈다.
- `OpenAIEmbeddings`는 `check_embedding_ctx_length=False`로 만든다. 켜 두면 tiktoken으로 토큰화한 정수 배열을 보내 vLLM의 모델과 맞지 않는다.
- `QdrantVectorStore`는 `validate_collection_config=False`로 만든다. 기본 검사는 임베딩 서버를 한 번 더 부른다.
- 500문항에 검색 약 128초, 생성 약 902초, 채점 약 26분이 걸린다.
- 색인을 새로 만드는 속도는 초당 약 30청크다(문서 1,000개, 3,730청크에 약 2분). 문서 전체(약 190만 청크)로 환산하면 약 17시간이다.

## 알려진 제한

- **문서 내용 수정을 감지하지 못한다.** 색인 키에는 문서 파일 수만 들어간다.
- **검색 결과가 실행마다 조금 다를 수 있다.** 임베딩 서버가 같은 질문에 매번 미세하게 다른 벡터를 돌려줘서(1−cos ≈ 1e-4), 점수 차가 아주 작은 이웃의 순위가 뒤집힌다. 20문항 중 5문항에서 관찰됐다. 생성 실험은 검색 결과를 고정하므로 영향을 받지 않는다.
- **청크 단위의 검색 품질은 직접 재지 않는다.** 근거 문서 ID는 문서 단위라서, 맞는 문서의 엉뚱한 부분을 가져와도 맞았다고 친다. 집계 표의 셋째 줄 조합으로 간접 추정한다.
- **작은 점수 차이는 우연일 수 있다.** 500문항이면 평균의 표본 오차가 최대 2점쯤이고, LLM 채점은 판정 흔들림이 더해진다.
- **Precision@k는 상한이 낮다.** 근거 문서가 1개인 문항이 많아서다. 리트리버를 고를 때는 Recall@k를 주 지표로, nDCG@k를 보조로 본다.
- **컨테이너의 외부 송신은 코드로 막지 못한다.**
- **파이프라인은 일직선만 표현한다.**

## 아직 정하지 않은 것

| 항목 | 정하는 때 |
|---|---|
| 의존성 버전. RAGAS와 LangChain이 함께 설치되는지 | 스캐폴딩 단위. 안 되면 멈추고 묻는다 |
| 문서 51만 개의 파일 수를 세는 데 걸리는 시간 | retrieve 최소 실행 단위에서 장비에서 잰다 |
| 프롬프트 문구 | generate 단위의 사전 리뷰 |
