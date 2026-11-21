import math

import pytest

from rag.metrics import question_scores, ranked_docs, summarize


def test_ranked_docs_keeps_first_occurrence_of_each_doc():
    """한 문서의 청크가 여러 번 잡히면 처음 나온 순위에 한 번만 남긴다."""
    assert ranked_docs(["a", "x", "a", "b", "x"]) == ["a", "x", "b"]


def test_question_scores_match_hand_computed_values():
    """순위 목록 [a, x, b, y], 근거 문서 {a, b}에서 손으로 계산한 값과 같다."""
    # Arrange: 청크 순서는 a, a, x, b, y라 순위 목록은 [a, x, b, y]다. 두 번째 a는 세지 않는다
    doc_ids, gold = ["a", "a", "x", "b", "y"], ["b", "a"]
    ideal = 1 + 1 / math.log2(3)

    # Act
    s = question_scores(doc_ids, gold, ks=[1, 2, 3])

    # Assert: k=1은 a, k=2는 a와 x, k=3은 a(1위)와 b(3위)를 본다
    assert [s["recall@1"], s["precision@1"], s["mrr@1"], s["ndcg@1"]] == [0.5, 1.0, 1.0, 1.0]
    assert [s["recall@2"], s["precision@2"], s["mrr@2"]] == [0.5, 0.5, 1.0]
    assert s["ndcg@2"] == pytest.approx(1 / ideal)
    assert [s["recall@3"], s["mrr@3"]] == [1.0, 1.0]
    assert s["precision@3"] == pytest.approx(2 / 3)
    assert s["ndcg@3"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_precision_divides_by_k_when_fewer_docs_than_k():
    """중복을 뺀 문서가 k개보다 적어도 Precision@k는 k로 나눈다."""
    # Act: 청크 3개가 모두 문서 a에서 왔다
    s = question_scores(["a", "a", "a"], ["a"], ks=[3])

    # Assert
    assert s["precision@3"] == pytest.approx(1 / 3)
    assert s["recall@3"] == 1.0


def test_question_scores_rank_of_first_hit():
    """첫 근거 문서가 2위면 MRR@1은 0, MRR@2는 1/2이고, 못 가져오면 모두 0이다."""
    # Act
    hit_second = question_scores(["x", "a"], ["a"], ks=[1, 2])
    miss = question_scores(["x", "y"], ["a"], ks=[2])

    # Assert
    assert (hit_second["mrr@1"], hit_second["mrr@2"]) == (0.0, 0.5)
    assert hit_second["ndcg@2"] == pytest.approx(1 / math.log2(3))
    assert miss == {"recall@2": 0.0, "precision@2": 0.0, "mrr@2": 0.0, "ndcg@2": 0.0}


def test_summarize_averages_overall_and_by_type():
    """전체 평균과 유형별 평균을 내고, 평균에 든 문항 수와 뺀 문항 수를 적는다."""
    # Arrange
    rows = [
        {"id": "q1", "type": "basic", "scores": {"recall@1": 1.0}},
        {"id": "q2", "type": "basic", "scores": {"recall@1": 0.0}},
        {"id": "q3", "type": "multi", "scores": {"recall@1": 0.5}},
    ]

    # Act
    m = summarize(rows, skipped=2)

    # Assert
    assert m == {
        "n_scored": 3,
        "n_skipped": 2,
        "mean": {"recall@1": 0.5},
        "by_type": {
            "basic": {"n": 2, "mean": {"recall@1": 0.5}},
            "multi": {"n": 1, "mean": {"recall@1": 0.5}},
        },
    }


def test_summarize_without_scored_questions():
    """모든 문항에 근거 문서가 없으면 평균은 비우고 뺀 문항 수만 적는다."""
    assert summarize([], skipped=3) == {"n_scored": 0, "n_skipped": 3, "mean": {}, "by_type": {}}
