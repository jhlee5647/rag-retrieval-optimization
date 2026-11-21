"""문서 폴더와 평가 파일 읽기."""

import json
from pathlib import Path
from typing import Any

from rag.config import EvalFields


def count_docs(docs_dir: Path) -> int:
    """문서 폴더 바로 아래의 `*.txt` 파일 수를 센다.

    문서 51만 개 폴더에서 약 1.5초 걸린다 (장비에서 측정).

    Args:
        docs_dir: 문서 폴더.

    Returns:
        `*.txt` 파일 수. 하위 폴더와 다른 확장자의 파일은 세지 않는다.

    Raises:
        NotADirectoryError: 폴더가 없는 경우. 0개로 세어 엉뚱한 색인 이름을 만들지 않게 한다.
    """
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"문서 폴더가 없다: {docs_dir}")
    return sum(1 for _ in docs_dir.glob("*.txt"))


def read_questions(eval_file: Path, fields: EvalFields) -> list[dict[str, Any]]:
    """평가 JSONL 파일에서 문항마다 `eval.fields`에 적힌 필드를 읽는다.

    Args:
        eval_file: 평가 JSONL 파일. 빈 줄은 건너뛴다.
        fields: 이 도구가 쓰는 이름과 평가 파일의 필드 이름의 대응.

    Returns:
        파일 순서대로 `{"id", "question", "doc_ids", "type"}`의 목록. 키는 이 도구가 쓰는 이름이다.
        `fields.answer`를 적었으면 `"answer"`도 담는다.

    Raises:
        KeyError: 문항에 `fields`가 가리키는 필드가 없는 경우.
    """
    with eval_file.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    names = fields.model_dump(exclude_none=True)
    return [{ours: row[theirs] for ours, theirs in names.items()} for row in rows]
