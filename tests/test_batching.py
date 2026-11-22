from lilytts.labeler.batching import cer, duration_batches, edit_distance


def test_batches_respect_padded_budget_and_cover_items():
    durs = [1.0, 2.5, 7.0, 3.0, 0.5, 12.0, 4.0, 30.0] * 50
    batches = duration_batches(durs, max_seconds=20.0, seed=1)
    seen = sorted(i for b in batches for i in b)
    assert seen == [i for i, d in enumerate(durs) if d <= 20.0]  # 30 s items are skipped
    for b in batches:
        assert max(durs[i] for i in b) * len(b) <= 20.0


def test_cer():
    assert edit_distance("キョーワ", "キョーワ") == 0
    assert edit_distance("キョーワ", "コンニチワ") == 4
    assert cer(["アイウ", "エオ"], ["アイウ", "エ"]) == 1 / 5
