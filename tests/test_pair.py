import json

import pytest

from rag.pair import main, pair


def write_run(path, scores_by_id, types=None):
    """문항 id → 점수로 실행 폴더의 scores.jsonl을 만든다. 적힌 순서대로 쓴다."""
    path.mkdir(parents=True)
    rows = [
        {"id": qid, "type": (types or {}).get(qid, "basic"), "scores": scores}
        for qid, scores in scores_by_id.items()
    ]
    (path / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def cells(table, metric):
    """표에서 그 지표 줄의 칸들을 돌려준다."""
    for line in table.splitlines():
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if row[0] == metric:
            return row
    raise AssertionError(f"{metric} 줄이 없다:\n{table}")


def test_pairs_scores_of_the_same_question(tmp_path):
    """파일에 적힌 순서가 달라도 같은 문항끼리 짝지어, 두 평균과 차이, 오른·내린 문항 수를 낸다."""
    # Arrange: 후보는 문항 순서가 반대다. 순서대로 짝지으면 오른 문항이 3개로 나온다
    base = write_run(
        tmp_path / "base",
        {"q1": {"fc": 0.2}, "q2": {"fc": 0.4}, "q3": {"fc": 0.6}, "q4": {"fc": 0.8}},
    )
    candidate = write_run(
        tmp_path / "candidate",
        {"q4": {"fc": 0.4}, "q3": {"fc": 0.6}, "q2": {"fc": 0.8}, "q1": {"fc": 0.6}},
    )

    # Act
    table = pair(base, candidate)

    # Assert
    assert table.splitlines()[:2] == [
        "| 지표 | 문항 | 빈 점수 | 기준 | 후보 | 차이 | 95% 구간 | 오름 | 내림 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    row = cells(table, "fc")
    assert row[1:6] == ["4", "0", "0.500", "0.600", "+0.100"]
    assert row[7:] == ["2", "1"]


def test_same_gain_on_every_question_gives_an_interval_without_width(tmp_path):
    """문항별 차이로 구간을 낸다. 문항마다 점수가 크게 달라도 모든 문항이 똑같이 올랐으면 구간에 폭이 없다."""
    # Arrange: 기준 점수는 0.0부터 0.75까지 퍼져 있고, 후보는 문항마다 0.25씩 높다
    scores = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
    base = write_run(tmp_path / "base", {f"q{i}": {"fc": s} for i, s in enumerate(scores)})
    candidate = write_run(
        tmp_path / "candidate", {f"q{i}": {"fc": s + 0.25} for i, s in enumerate(scores)}
    )

    # Act
    table = pair(base, candidate)

    # Assert
    assert cells(table, "fc")[5:] == ["+0.250", "[+0.250, +0.250]", "7", "0"]


def test_run_paired_with_itself_shows_no_difference(tmp_path):
    """같은 실행끼리 비교하면 차이도 구간도 0이고, 오르거나 내린 문항이 없다."""
    # Arrange
    run = write_run(tmp_path / "run", {"q1": {"fc": 0.3}, "q2": {"fc": 0.9}, "q3": {"fc": 0.0}})

    # Act
    table = pair(run, run)

    # Assert
    assert cells(table, "fc")[5:] == ["+0.000", "[+0.000, +0.000]", "0", "0"]


def test_interval_is_fixed_for_the_same_scores(tmp_path):
    """시드, 반복 횟수, 구간을 끊는 자리가 바뀌면 이미 보고한 구간과 달라진다. 같은 점수의 구간을 지킨다."""
    # Arrange: 오른 문항과 내린 문항이 섞인 20문항
    gains = [0.5, -0.25, 0.0, 0.25, 1.0, -0.5, 0.0, 0.25, 0.5, -0.25] * 2
    base = write_run(tmp_path / "base", {f"q{i}": {"fc": 0.5} for i in range(20)})
    candidate = write_run(
        tmp_path / "candidate", {f"q{i}": {"fc": 0.5 + g} for i, g in enumerate(gains)}
    )

    # Act
    table = pair(base, candidate)

    # Assert: 처음 낸 구간을 그대로 적어 둔 것이다
    assert cells(table, "fc")[5:7] == ["+0.150", "[-0.037, +0.338]"]


def test_stops_when_question_sets_differ(tmp_path):
    """문항이 다른 두 실행(평가 파일이나 --limit이 다름)은 짝지을 수 없으므로 멈춘다."""
    # Arrange
    base = write_run(tmp_path / "base", {"q1": {"fc": 0.5}, "q2": {"fc": 0.5}})
    candidate = write_run(tmp_path / "candidate", {"q1": {"fc": 0.5}, "q3": {"fc": 0.5}})

    # Act / Assert
    with pytest.raises(ValueError, match="기준에만 1개, 후보에만 1개"):
        pair(base, candidate)


def test_drops_pairs_with_an_empty_score_and_counts_them(tmp_path):
    """한쪽이라도 점수가 빈 문항은 그 지표에서만 양쪽 평균 모두에서 빼고, 뺀 수를 적는다."""
    # Arrange: 후보의 q2는 faithfulness만 비었다
    base = write_run(
        tmp_path / "base",
        {
            "q1": {"faithfulness": 1.0, "fc": 0.5},
            "q2": {"faithfulness": 0.0, "fc": 0.5},
            "q3": {"faithfulness": 1.0, "fc": 0.5},
        },
    )
    candidate = write_run(
        tmp_path / "candidate",
        {
            "q1": {"faithfulness": 1.0, "fc": 0.5},
            "q2": {"faithfulness": None, "fc": 1.0},
            "q3": {"faithfulness": 0.5, "fc": 0.5},
        },
    )

    # Act
    table = pair(base, candidate)

    # Assert: faithfulness의 기준 평균은 q2를 뺀 1.0이다
    row = cells(table, "faithfulness")
    assert row[1:6] + row[7:] == ["2", "1", "1.000", "0.750", "-0.250", "0", "1"]
    assert cells(table, "fc")[1:6] == ["3", "0", "0.500", "0.667", "+0.167"]


def test_derives_share_of_questions_with_all_gold_documents(tmp_path):
    """recall@k가 1인 문항(근거 문서를 전부 가져옴)의 비율을 all_gold@k로 낸다. 부분 점수는 세지 않는다."""
    # Arrange
    base = write_run(
        tmp_path / "base",
        {"q1": {"recall@5": 1.0}, "q2": {"recall@5": 0.5}, "q3": {"recall@5": 0.0}},
    )
    candidate = write_run(
        tmp_path / "candidate",
        {"q1": {"recall@5": 1.0}, "q2": {"recall@5": 1.0}, "q3": {"recall@5": 0.5}},
    )

    # Act
    table = pair(base, candidate)

    # Assert
    row = cells(table, "all_gold@5")
    assert row[1:6] + row[7:] == ["3", "0", "0.333", "0.667", "+0.333", "1", "0"]


def test_derives_refusal_from_zero_response_relevancy(tmp_path):
    """response_relevancy가 0인 답의 비율을 refusal로 낸다. 채점이 빈 답은 거절로 세지 않고 뺀다."""
    # Arrange
    base = write_run(
        tmp_path / "base",
        {
            "q1": {"response_relevancy": 0.0},
            "q2": {"response_relevancy": 0.8},
            "q3": {"response_relevancy": None},
        },
    )
    candidate = write_run(
        tmp_path / "candidate",
        {
            "q1": {"response_relevancy": 0.7},
            "q2": {"response_relevancy": 0.0},
            "q3": {"response_relevancy": 0.0},
        },
    )

    # Act
    table = pair(base, candidate)

    # Assert
    row = cells(table, "refusal")
    assert row[1:6] + row[7:] == ["2", "1", "0.500", "0.500", "+0.000", "1", "1"]


def test_pairs_metrics_with_different_names(tmp_path):
    """가져오는 청크 수가 다른 두 실행은 지표 이름이 다르다. 적어 준 짝대로 비교하고, 없는 지표는 거부한다."""
    # Arrange
    base = write_run(tmp_path / "base", {"q1": {"recall@5": 0.5}, "q2": {"recall@5": 0.0}})
    candidate = write_run(
        tmp_path / "candidate", {"q1": {"recall@10": 1.0}, "q2": {"recall@10": 0.0}}
    )

    # Act
    table = pair(
        base, candidate, metrics=[("recall@5", "recall@10"), ("all_gold@5", "all_gold@10")]
    )

    # Assert
    assert cells(table, "recall@5 → recall@10")[1:6] == ["2", "0", "0.250", "0.500", "+0.250"]
    assert cells(table, "all_gold@5 → all_gold@10")[3:6] == ["0.000", "0.500", "+0.500"]
    with pytest.raises(ValueError, match="없는 지표: recall@5=recall@5"):
        pair(base, candidate, metrics=[("recall@5", "recall@5")])


def test_default_metrics_are_those_in_both_runs(tmp_path):
    """지표를 적어 주지 않으면 두 실행에 모두 있는 지표만 낸다. 한쪽에만 있는 지표는 짝이 없다."""
    # Arrange: recall@1만 양쪽에 있다
    base = write_run(tmp_path / "base", {"q1": {"recall@1": 0.0, "recall@5": 1.0}})
    candidate = write_run(tmp_path / "candidate", {"q1": {"recall@1": 1.0, "recall@10": 1.0}})

    # Act
    table = pair(base, candidate)

    # Assert
    assert [line.split("|")[1].strip() for line in table.splitlines()[2:]] == [
        "recall@1",
        "all_gold@1",
    ]


def test_excludes_question_types(tmp_path):
    """뺄 유형의 문항은 평균에 넣지 않는다. 정답 칸이 채점 기준인 유형을 뺄 때 쓴다."""
    # Arrange: q3만 뺄 유형이다
    types = {"q3": "info_not_found"}
    base = write_run(
        tmp_path / "base", {"q1": {"fc": 0.5}, "q2": {"fc": 0.5}, "q3": {"fc": 0.0}}, types
    )
    candidate = write_run(
        tmp_path / "candidate", {"q1": {"fc": 1.0}, "q2": {"fc": 0.5}, "q3": {"fc": 0.0}}, types
    )

    # Act
    table = pair(base, candidate, exclude_types=["info_not_found"])

    # Assert
    assert cells(table, "fc")[1:6] == ["2", "0", "0.500", "0.750", "+0.250"]


def test_keeps_only_questions_both_runs_answered(tmp_path):
    """거절한 답의 점수가 섞이지 않게, 두 실행이 모두 답한 문항만 본다. 거절인지 모르는 답도 뺀다."""
    # Arrange: q1은 기준이, q2는 후보가 거절했고, q4는 후보의 채점이 비었다
    base = write_run(
        tmp_path / "base",
        {
            "q1": {"response_relevancy": 0.0, "faithfulness": 0.0},
            "q2": {"response_relevancy": 0.9, "faithfulness": 1.0},
            "q3": {"response_relevancy": 0.8, "faithfulness": 1.0},
            "q4": {"response_relevancy": 0.7, "faithfulness": 1.0},
        },
    )
    candidate = write_run(
        tmp_path / "candidate",
        {
            "q1": {"response_relevancy": 0.9, "faithfulness": 1.0},
            "q2": {"response_relevancy": 0.0, "faithfulness": 0.0},
            "q3": {"response_relevancy": 0.8, "faithfulness": 0.5},
            "q4": {"response_relevancy": None, "faithfulness": 1.0},
        },
    )

    # Act
    table = pair(base, candidate, answered=True)

    # Assert: 남는 것은 q3 하나다
    assert cells(table, "faithfulness")[1:6] == ["1", "0", "1.000", "0.500", "-0.500"]


def test_stops_when_no_question_is_left(tmp_path):
    """걸러 낸 뒤 문항이 남지 않으면 빈 표를 내지 않고 멈춘다."""
    # Arrange
    run = write_run(tmp_path / "run", {"q1": {"fc": 0.5}})

    # Act / Assert
    with pytest.raises(ValueError, match="짝지을 문항이 없다"):
        pair(run, run, exclude_types=["basic"])


def test_main_passes_options(tmp_path, capsys):
    """명령줄의 지표 짝, 뺄 유형, 답한 문항만 보기를 그대로 넘겨 표를 출력한다."""
    # Arrange: 옵션마다 표가 달라지는 두 실행
    types = {"q3": "info_not_found"}
    base = write_run(
        tmp_path / "base",
        {
            "q1": {"response_relevancy": 0.9, "faithfulness": 1.0, "fc": 0.5},
            "q2": {"response_relevancy": 0.0, "faithfulness": 0.0, "fc": 0.0},
            "q3": {"response_relevancy": 0.9, "faithfulness": 1.0, "fc": 0.0},
        },
        types,
    )
    candidate = write_run(
        tmp_path / "candidate",
        {
            "q1": {"response_relevancy": 0.9, "faithfulness": 0.5, "fc": 1.0},
            "q2": {"response_relevancy": 0.9, "faithfulness": 1.0, "fc": 1.0},
            "q3": {"response_relevancy": 0.9, "faithfulness": 0.0, "fc": 1.0},
        },
        types,
    )

    # Act
    options = ["--metric", "fc", "--metric", "faithfulness=fc"]
    main([str(base), str(candidate), *options, "--exclude-type", "info_not_found", "--answered"])

    # Assert: 남는 것은 q1 하나고, 적어 준 두 줄만 나온다
    table = capsys.readouterr().out
    assert len(table.splitlines()) == 4
    assert cells(table, "fc")[1:6] == ["1", "0", "0.500", "1.000", "+0.500"]
    assert cells(table, "faithfulness → fc")[1:6] == ["1", "0", "1.000", "1.000", "+0.000"]
