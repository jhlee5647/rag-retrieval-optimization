"""pair 명령: 두 실행을 같은 문항끼리 짝지어, 지표마다 평균 차이와 그 부트스트랩 95% 구간을 표로 낸다.

    uv run --env-file .env python -m rag.pair <기준 실행 폴더> <후보 실행 폴더>

구간이 0을 포함하면 그 차이는 우연일 수 있다(docs/optimization.md의 "차이 판정"). 점수 파일의 지표에
더해 두 가지를 만들어 낸다. `all_gold@k`는 근거 문서를 전부 가져온 문항(`recall@k`가 1)이고,
`refusal`은 거절한 답(`response_relevancy`가 0)이다.
"""

import argparse
import json
import random
from collections.abc import Collection
from pathlib import Path
from typing import Any

from rag.compare import _table

# 같은 두 실행을 다시 비교해도 같은 구간이 나오게 고정한다.
SEED = 0
RESAMPLES = 10_000


def _read(run_dir: Path) -> dict[str, dict[str, Any]]:
    """`scores.jsonl`을 `{문항 id: {"type", "scores"}}`로 읽고, 만들어 내는 지표를 더한다."""
    rows = {}
    with (run_dir / "scores.jsonl").open(encoding="utf-8") as f:
        for row in map(json.loads, f):
            scores = dict(row["scores"])
            for key, value in row["scores"].items():
                if key.startswith("recall@"):
                    scores["all_gold@" + key.removeprefix("recall@")] = float(value == 1)
            if "response_relevancy" in scores:
                relevancy = scores["response_relevancy"]
                # 채점이 비어 거절인지 알 수 없는 답은 0으로 채우지 않고 비워 둔다.
                scores["refusal"] = None if relevancy is None else float(relevancy == 0)
            rows[row["id"]] = {"type": row["type"], "scores": scores}
    return rows


def _interval(diffs: list[float]) -> tuple[float, float]:
    """문항별 차이를 복원 추출해 낸 평균들의 가운데 95% 범위를 돌려준다."""
    # 지표마다 같은 시드로 시작해, 함께 보는 지표가 달라져도 한 지표의 구간이 바뀌지 않게 한다.
    rng = random.Random(SEED)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(RESAMPLES))
    return means[int(RESAMPLES * 0.025)], means[int(RESAMPLES * 0.975) - 1]


def pair(
    base_dir: Path,
    candidate_dir: Path,
    *,
    metrics: list[tuple[str, str]] | None = None,
    exclude_types: Collection[str] = (),
    answered: bool = False,
) -> str:
    """두 실행의 `scores.jsonl`을 문항 id로 짝지어 지표별 비교 표를 만든다.

    Args:
        base_dir: 기준 실행 폴더.
        candidate_dir: 후보 실행 폴더. 차이는 후보에서 기준을 뺀 값이다.
        metrics: 짝지을 `(기준의 지표, 후보의 지표)` 목록. 가져오는 청크 수가 다른 두 실행의
            `recall@5`와 `recall@10`처럼 이름이 다른 지표를 짝지을 때 쓴다. None이면 두 실행에
            모두 있는 지표를 전부 낸다.
        exclude_types: 뺄 문항 유형.
        answered: True면 두 실행이 모두 답한(거절하지 않은) 문항만 본다.

    Returns:
        마크다운 표. 지표마다 짝지은 문항 수, 한쪽이라도 점수가 비어 뺀 문항 수, 두 평균, 평균 차이,
        그 95% 구간, 점수가 오른 문항 수와 내린 문항 수다.

    Raises:
        ValueError: 두 실행의 문항이 다르거나, 걸러 낸 뒤 문항이 남지 않거나, 없는 지표를 적은 경우.
    """
    base, candidate = _read(base_dir), _read(candidate_dir)
    if base.keys() != candidate.keys():
        raise ValueError(
            f"두 실행의 문항이 다르다: 기준에만 {len(base.keys() - candidate.keys())}개, "
            f"후보에만 {len(candidate.keys() - base.keys())}개"
        )
    ids = [i for i in base if base[i]["type"] not in exclude_types]
    if answered:
        ids = [
            i for i in ids if all(run[i]["scores"].get("refusal") == 0 for run in (base, candidate))
        ]
    if not ids:
        raise ValueError("짝지을 문항이 없다")

    base_keys, candidate_keys = (list(run[ids[0]]["scores"]) for run in (base, candidate))
    if metrics is None:
        metrics = [(key, key) for key in base_keys if key in candidate_keys]
    rows = []
    for a, b in metrics:
        if a not in base_keys or b not in candidate_keys:
            raise ValueError(f"없는 지표: {a}={b}. 기준 {base_keys}, 후보 {candidate_keys}")
        pairs = [(base[i]["scores"][a], candidate[i]["scores"][b]) for i in ids]
        kept = [(x, y) for x, y in pairs if x is not None and y is not None]
        row = [a if a == b else f"{a} → {b}", str(len(kept)), str(len(pairs) - len(kept))]
        if kept:
            diffs = [y - x for x, y in kept]
            low, high = _interval(diffs)
            row += [
                f"{sum(x for x, _ in kept) / len(kept):.3f}",
                f"{sum(y for _, y in kept) / len(kept):.3f}",
                f"{sum(diffs) / len(diffs):+.3f}",
                f"[{low:+.3f}, {high:+.3f}]",
                str(sum(d > 0 for d in diffs)),
                str(sum(d < 0 for d in diffs)),
            ]
        else:
            row += ["-"] * 6
        rows.append(row)
    header = ["지표", "문항", "빈 점수", "기준", "후보", "차이", "95% 구간", "오름", "내림"]
    return _table(header, rows)


def _metric_pair(text: str) -> tuple[str, str]:
    """`A=B`는 기준의 A와 후보의 B를, `A`는 양쪽의 A를 짝짓는다."""
    a, _, b = text.partition("=")
    return a, b or a


def main(argv: list[str] | None = None) -> None:
    """두 실행 폴더의 비교 표를 출력한다.

    Args:
        argv: 명령줄 인자. None이면 `sys.argv`를 쓴다.
    """
    parser = argparse.ArgumentParser(prog="python -m rag.pair", description=__doc__)
    parser.add_argument("base", type=Path, help="기준 실행 폴더")
    parser.add_argument("candidate", type=Path, help="후보 실행 폴더")
    parser.add_argument(
        "--metric",
        action="append",
        type=_metric_pair,
        metavar="A[=B]",
        help="낼 지표. A=B는 기준의 A와 후보의 B를 짝짓는다. 여러 번 쓸 수 있고, 빼면 전부 낸다",
    )
    parser.add_argument(
        "--exclude-type", action="append", default=[], metavar="유형", help="뺄 문항 유형"
    )
    parser.add_argument(
        "--answered", action="store_true", help="두 실행이 모두 답한(거절하지 않은) 문항만 본다"
    )
    args = parser.parse_args(argv)
    table = pair(
        args.base,
        args.candidate,
        metrics=args.metric,
        exclude_types=args.exclude_type,
        answered=args.answered,
    )
    print(table)


if __name__ == "__main__":
    main()
