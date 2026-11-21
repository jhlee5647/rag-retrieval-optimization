import json

from rag.compare import compare


def write(path, data):
    path.mkdir(parents=True, exist_ok=True)
    for name, content in data.items():
        (path / name).write_text(json.dumps(content))


def test_prints_retrieval_and_generation_tables(tmp_path):
    """검색 실행끼리, 생성 실행끼리 표를 따로 내고, 생성 실행에는 사용한 검색 실행을 보여 준다."""
    # Arrange: eval.k가 서로 다른 검색 실행 둘(뒤의 실행에만 있는 지표도 칸이 생긴다), 지표 전의 검색 실행 하나, 채점한 생성 실행 하나와 안 한 것 하나
    runs = tmp_path / "runs"
    mean_a = {"recall@1": 0.5, "recall@5": 1.0}
    write(
        runs / "retrieve/naive/20260101-000000", {"metrics.json": {"n_scored": 2, "mean": mean_a}}
    )
    mean_b = {"recall@1": 0.25, "recall@10": 0.75}
    write(
        runs / "retrieve/other/20260102-000000", {"metrics.json": {"n_scored": 4, "mean": mean_b}}
    )
    write(runs / "retrieve/naive/20251231-000000", {"run.json": {}})
    write(
        runs / "generate/naive/20260103-000000",
        {
            "run.json": {"contexts": "runs/retrieve/naive/20260101-000000"},
            "metrics.json": {
                "n": 2,
                "empty": {"faithfulness": 1, "factual_correctness": 0},
                "mean": {"faithfulness": 0.8, "factual_correctness": None},
            },
        },
    )
    write(runs / "generate/naive/20260104-000000", {"run.json": {"contexts": "x/y/z"}})

    # Act
    out = compare(runs)

    # Assert
    assert out == (
        "## 검색 실행\n"
        "\n"
        "| 실행 | 문항 | recall@1 | recall@5 | recall@10 |\n"
        "|---|---|---|---|---|\n"
        "| naive/20260101-000000 | 2 | 0.500 | 1.000 | - |\n"
        "| other/20260102-000000 | 4 | 0.250 | - | 0.750 |\n"
        "\n"
        "## 생성 실행\n"
        "\n"
        "| 실행 | 검색 실행 | 문항 | 빈 점수 | faithfulness | factual_correctness |\n"
        "|---|---|---|---|---|---|\n"
        "| naive/20260103-000000 | naive/20260101-000000 | 2 | 1 | 0.800 | - |"
    )


def test_says_none_when_there_are_no_runs(tmp_path):
    """실행이 하나도 없으면 빈 표 대신 (없음)이라고 적는다."""
    assert compare(tmp_path / "runs") == "## 검색 실행\n\n(없음)\n\n## 생성 실행\n\n(없음)"
