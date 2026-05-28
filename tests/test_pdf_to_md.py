from src.pdfToMd import normalize_text_to_markdown


def test_normalize_text_removes_page_noise_but_keeps_chapter_heading() -> None:
    raw = "\n".join(
        [
            "CHAPTER I",
            "",
            "1",
            "Page 1",
            "It is a truth universally acknowledged.",
            "Project Gutenberg eBook of Pride and Prejudice",
            "",
            "CHAPTER II",
            "2",
            "Mr. Bennet replied that he had not.",
        ]
    )

    normalized = normalize_text_to_markdown(raw)

    assert "CHAPTER I" in normalized
    assert "CHAPTER II" in normalized
    assert "It is a truth universally acknowledged." in normalized
    assert "Mr. Bennet replied that he had not." in normalized
    assert "Page 1" not in normalized
    assert "Project Gutenberg" not in normalized
    assert "\n1\n" not in f"\n{normalized}\n"
    assert "\n2\n" not in f"\n{normalized}\n"
