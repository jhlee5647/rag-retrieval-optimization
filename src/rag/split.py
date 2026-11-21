"""split 명령: 평가 파일을 문항 유형별로 반씩 나눠 dev 파일과 test 파일을 쓴다.

    uv run --env-file .env python -m rag.split configs/naive.yaml

`EVAL_FILE` 옆에 `<이름>-dev.jsonl`과 `<이름>-test.jsonl`이 생긴다. 구성은 dev로 고르고, test는 마지막
확인에만 쓴다(docs/optimization.md).
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from rag.config import load_config
from rag.runs import file_sha256

# 다시 나눠도 같은 분할이 나오게 고정한다. 바꾸면 이미 본 dev 문항이 test로 넘어간다.
SEED = 0


def _dev_positions(types: list[str]) -> set[int]:
    """유형마다 절반을 골라, dev에 넣을 문항의 위치를 돌려준다."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for position, name in enumerate(types):
        by_type[name].append(position)
    rng = random.Random(SEED)
    order: list[int] = []
    for positions in by_type.values():
        rng.shuffle(positions)
        order += positions
    # 유형별로 섞은 순서를 이어 붙여 하나 걸러 하나씩 고른다. 유형마다 반씩 갈리고, 문항 수가 홀수인
    # 유형의 남는 한 문항은 dev와 test에 번갈아 가서 두 파일의 문항 수가 같아진다.
    return set(order[::2])


def split(eval_file: Path, type_field: str) -> dict[Path, int]:
    """평가 파일을 유형별로 반씩 나눠 `<이름>-dev.jsonl`과 `<이름>-test.jsonl`을 쓴다.

    원본의 줄을 그대로, 원본 순서대로 쓴다. 유형별로 모아 쓰면 `retrieve --limit`이 쓰는 앞 N문항이
    한 유형으로 쏠린다.

    Args:
        eval_file: 평가 JSONL 파일. 빈 줄은 건너뛴다.
        type_field: 평가 파일에서 문항 유형이 든 필드 이름.

    Returns:
        쓴 파일과 그 문항 수. dev, test 순서다. 전체 문항 수가 홀수면 dev가 하나 많다.

    Raises:
        FileExistsError: 두 파일 중 하나라도 이미 있는 경우. 실험에 쓰고 있는 분할을 덮어쓰지 않는다.
        KeyError: 문항에 `type_field`가 없는 경우.
    """
    dev_file, test_file = (
        eval_file.with_name(f"{eval_file.stem}-{part}.jsonl") for part in ("dev", "test")
    )
    for path in (dev_file, test_file):
        if path.exists():
            raise FileExistsError(f"분할 파일이 이미 있다: {path}")
    with eval_file.open(encoding="utf-8") as f:
        lines = [line.rstrip("\n") + "\n" for line in f if line.strip()]
    dev = _dev_positions([json.loads(line)[type_field] for line in lines])

    written = {}
    for path, in_dev in ((dev_file, True), (test_file, False)):
        part = [line for position, line in enumerate(lines) if (position in dev) == in_dev]
        path.write_text("".join(part), encoding="utf-8")
        written[path] = len(part)
    return written


def main(argv: list[str] | None = None) -> None:
    """`EVAL_FILE`을 config의 유형 필드로 나누고, 쓴 파일마다 문항 수와 해시를 출력한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.split", description=__doc__)
    parser.add_argument("config", type=Path, help="config 파일 경로. eval.fields.type을 읽는다")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    written = split(Path(os.environ["EVAL_FILE"]), config.eval.fields.type)
    for path, count in written.items():
        # 해시는 실행 기록의 eval_file_sha256과 대조해, 어느 분할로 돌린 실행인지 가리는 데 쓴다.
        print(f"{path}  {count}문항  sha256 {file_sha256(path)}")


if __name__ == "__main__":
    main()
