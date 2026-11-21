"""검색 지표와, 문항별 점수의 집계.

검색 지표의 정의는 docs/design.md의 "검색 지표"를 따른다. 관련 여부는 근거 문서인지 아닌지(0과 1)로만 본다.
"""

import math
from collections import defaultdict
from typing import Any


def ranked_docs(doc_ids: list[str]) -> list[str]:
    """청크의 문서 ID 목록에서 처음 나온 것만 남긴 순위 목록 R을 만든다.

    한 문서의 청크가 여러 개 잡혀도 그 문서를 한 번만 센다.
    """
    return list(dict.fromkeys(doc_ids))


def question_scores(doc_ids: list[str], gold: list[str], ks: list[int]) -> dict[str, float]:
    """한 문항의 Recall, Precision, MRR, nDCG를 `ks`의 순위마다 계산한다.

    Args:
        doc_ids: 가져온 청크의 문서 ID. 순위 순서다.
        gold: 근거 문서 ID. 비어 있으면 안 된다.
        ks: 지표를 잴 순위 목록.

    Returns:
        `{"recall@1": ..., "precision@1": ..., "mrr@1": ..., "ndcg@1": ..., ...}`.
    """
    ranked, relevant = ranked_docs(doc_ids), set(gold)
    scores = {}
    for k in ks:
        hits = [doc in relevant for doc in ranked[:k]]
        dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
        first = next((rank for rank, hit in enumerate(hits) if hit), None)
        scores[f"recall@{k}"] = sum(hits) / len(relevant)
        scores[f"precision@{k}"] = sum(hits) / k
        scores[f"mrr@{k}"] = 0.0 if first is None else 1 / (first + 1)
        scores[f"ndcg@{k}"] = dcg / ideal
    return scores


def _mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row["scores"][key] for row in rows) / len(rows) for key in rows[0]["scores"]}


def summarize(rows: list[dict[str, Any]], skipped: int) -> dict[str, Any]:
    """문항별 점수의 전체 평균과 유형별 평균을 낸다.

    Args:
        rows: 점수를 낸 문항마다 `{"id", "type", "scores"}`. 비어 있으면 평균도 비어 있다.
        skipped: 근거 문서가 지정되지 않아 평균에서 뺀 문항 수.

    Returns:
        `{"n_scored", "n_skipped", "mean", "by_type": {유형: {"n", "mean"}}}`.
    """
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    return {
        "n_scored": len(rows),
        "n_skipped": skipped,
        "mean": _mean(rows),
        "by_type": {t: {"n": len(rs), "mean": _mean(rs)} for t, rs in sorted(by_type.items())},
    }


def retrieval_hit(doc_ids: list[str], gold: list[str]) -> str:
    """근거 문서를 가져온 정도: `all`(전부), `some`(일부), `none`(못 가져옴), `no_gold`(지정 안 됨)."""
    if not gold:
        return "no_gold"
    found = set(gold) & set(doc_ids)
    if found == set(gold):
        return "all"
    return "some" if found else "none"


def _present_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["scores"][key] for row in rows if row["scores"][key] is not None]
    return sum(values) / len(values) if values else None


def summarize_answers(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    """생성 지표의 전체 평균, 유형별 평균, 검색 적중별 평균을 낸다.

    채점 모델이 출력 형식을 못 맞춰 빈 점수(None)는 0으로 채우지 않고 평균에서 빼며, 지표마다 그 수를 센다.

    Args:
        rows: 문항마다 `{"id", "type", "hit", "scores": {지표: 점수 또는 None}}`.
        keys: 지표 이름 목록.

    Returns:
        `{"n", "empty", "mean", "by_type", "by_hit"}`. 나눈 평균은 `{값: {"n", "mean"}}`이다.
    """

    def mean(rs: list[dict[str, Any]]) -> dict[str, float | None]:
        return {key: _present_mean(rs, key) for key in keys}

    def by(field: str) -> dict[str, Any]:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        return {v: {"n": len(rs), "mean": mean(rs)} for v, rs in sorted(groups.items())}

    return {
        "n": len(rows),
        "empty": {key: sum(row["scores"][key] is None for row in rows) for key in keys},
        "mean": mean(rows),
        "by_type": by("type"),
        "by_hit": by("hit"),
    }
