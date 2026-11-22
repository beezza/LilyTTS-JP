from lilytts.text.ruby import RubySpan, normalize_ruby_field, parse_inline_ruby


def test_explicit_aozora():
    plain, spans, warns = parse_inline_ruby("今日は｜日本橋《にっぽんばし》で")
    assert plain == "今日は日本橋で"
    assert spans == [RubySpan(3, 6, "ニッポンバシ")]
    assert warns == []


def test_ascii_bar_and_implicit_base():
    plain, spans, _ = parse_inline_ruby("|安全《あんぜん》教室《きょうしつ》ですか")
    assert plain == "安全教室ですか"
    assert spans == [RubySpan(0, 2, "アンゼン"), RubySpan(2, 4, "キョウシツ")]


def test_implicit_base_takes_kanji_run_only():
    plain, spans, _ = parse_inline_ruby("これは人々《ひとびと》です")
    assert plain == "これは人々です"
    assert spans == [RubySpan(3, 5, "ヒトビト")]


def test_implicit_without_kanji_is_kept_as_text():
    plain, spans, warns = parse_inline_ruby("あいう《えお》")
    assert plain == "あいう《えお》"
    assert spans == []
    assert len(warns) == 1


def test_brace_html_and_kag():
    plain, spans, _ = parse_inline_ruby(
        "{今日|こんにち}は<ruby>上手<rp>(</rp><rt>うわて</rt><rp>)</rp></ruby>[ruby text=\"かた\"]方"
    )
    assert plain == "今日は上手方"
    assert spans == [RubySpan(0, 2, "コンニチ"), RubySpan(3, 5, "ウワテ"), RubySpan(5, 6, "カタ")]


def test_emphasis_dots_dropped_silently():
    plain, spans, warns = parse_inline_ruby("｜本当《・・》に")
    assert plain == "本当に"
    assert spans == [] and warns == []


def test_non_kana_reading_dropped_with_warning():
    plain, spans, warns = parse_inline_ruby("｜約束《promise》")
    assert plain == "約束"
    assert spans == [] and len(warns) == 1


def test_field_offsets():
    spans, warns = normalize_ruby_field([{"start": 0, "end": 2, "reading": "あんぜん"}], "安全教室ですか")
    assert spans == [RubySpan(0, 2, "アンゼン")] and warns == []


def test_field_base_located_left_to_right():
    spans, _ = normalize_ruby_field(
        [{"base": "方", "reading": "かた"}, {"base": "方", "reading": "ほう"}], "あの方とこの方"
    )
    assert spans == [RubySpan(2, 3, "カタ"), RubySpan(6, 7, "ホウ")]


def test_field_pairs_and_dict():
    assert normalize_ruby_field([["教室", "きょうしつ"]], "安全教室")[0] == [RubySpan(2, 4, "キョウシツ")]
    assert normalize_ruby_field({"安全": "あんぜん"}, "安全第一、安全")[0] == [
        RubySpan(0, 2, "アンゼン"),
        RubySpan(5, 7, "アンゼン"),
    ]


def test_field_mismatch_and_overlap():
    spans, warns = normalize_ruby_field(
        [{"start": 0, "end": 2, "base": "教室", "reading": "x"}], "安全教室"
    )
    assert spans == [] and len(warns) == 1
    spans, warns = normalize_ruby_field(
        [{"start": 0, "end": 3, "reading": "あんぜんきょう"}, {"start": 2, "end": 4, "reading": "きょうしつ"}],
        "安全教室",
    )
    assert spans == [RubySpan(0, 3, "アンゼンキョウ")] and len(warns) == 1
