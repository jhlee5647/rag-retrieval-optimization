"""문서 폴더와 평가 파일 읽기."""

import json
from pathlib import Path


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


def read_questions(eval_file: Path) -> list[dict[str, str]]:
    """평가 JSONL 파일에서 문항마다 `id`와 `question`을 읽는다.

    Args:
        eval_file: 평가 JSONL 파일. 빈 줄은 건너뛴다.

    Returns:
        파일 순서대로 `{"id": ..., "question": ...}`의 목록.
    """
    with eval_file.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"id": row["id"], "question": row["question"]} for row in rows]
