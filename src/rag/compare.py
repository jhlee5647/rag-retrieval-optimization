"""compare 명령: 실행 폴더들의 지표를 검색 실행끼리, 생성 실행끼리 표로 출력한다.

    uv run --env-file .env python -m rag.compare

`metrics.json`이 없는 실행(지표를 내기 전의 검색, 아직 채점하지 않은 생성)은 건너뛴다.
"""

import argparse
import json
from pathlib import Path
from typing import Any

RUNS_ROOT = Path("runs")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _number(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}"


def _runs(root: Path) -> list[tuple[str, dict[str, Any], Path]]:
    """`<root>/<config 이름>/<실행 시각>/metrics.json`을 실행 폴더 이름 순서로 읽는다."""
    found = []
    for path in sorted(root.glob("*/*/metrics.json")):
        name = path.parent.relative_to(root).as_posix()
        found.append((name, json.loads(path.read_text(encoding="utf-8")), path.parent))
    return found


def _keys(runs: list[tuple[str, dict[str, Any], Path]]) -> list[str]:
    # 실행마다 eval.k가 다를 수 있어 지표 이름을 합친다. 처음 나온 순서를 지킨다.
    return list(dict.fromkeys(key for _, metrics, _ in runs for key in metrics["mean"]))


def compare(runs_root: Path) -> str:
    """검색 실행 표와 생성 실행 표를 만든다.

    Args:
        runs_root: 실행 폴더들의 최상위 폴더. 보통 `runs`.

    Returns:
        마크다운 표 두 개. 지표가 없는 칸은 `-`다.
    """
    retrieval = _runs(runs_root / "retrieve")
    keys = _keys(retrieval)
    rows = [
        [name, str(m["n_scored"]), *(_number(m["mean"].get(k)) for k in keys)]
        for name, m, _ in retrieval
    ]
    parts = ["## 검색 실행", _table(["실행", "문항", *keys], rows) if rows else "(없음)"]

    generation = _runs(runs_root / "generate")
    keys = _keys(generation)
    rows = []
    for name, m, run_dir in generation:
        contexts = Path(json.loads((run_dir / "run.json").read_text())["contexts"])
        # 검색 실행은 검색 실행 표와 같은 이름(<config 이름>/<실행 시각>)으로 보인다.
        used = "/".join(contexts.parts[-2:])
        empty = sum(m["empty"].values())
        cells = [_number(m["mean"].get(k)) for k in keys]
        rows.append([name, used, str(m["n"]), str(empty), *cells])
    header = ["실행", "검색 실행", "문항", "빈 점수", *keys]
    parts += ["## 생성 실행", _table(header, rows) if rows else "(없음)"]
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    """`runs/` 아래 실행들의 지표 표를 출력한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    argparse.ArgumentParser(prog="python -m rag.compare", description=__doc__).parse_args(argv)
    print(compare(RUNS_ROOT))


if __name__ == "__main__":
    main()
