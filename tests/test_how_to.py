from __future__ import annotations

from pathlib import Path

from minimal_nn_competition.leaderboard.render import BENCHMARK_GROUPS, render_html

ROOT = Path(__file__).resolve().parents[1]


def _guide_name(group: str) -> str:
    article = "an" if group[:1] in "aeiou" else "a"
    return f"how-to-make-{article}-{group}-submission.md"


def test_how_to_submission_guides_match_benchmark_groups() -> None:
    how_to = ROOT / "how-to"
    actual = {
        path.name
        for path in how_to.glob("how-to-make-a*-submission.md")
        if path.name != "how-to-make-a-submission.md"
    }
    expected = {_guide_name(group) for group in BENCHMARK_GROUPS}
    assert actual == expected


def test_general_submission_guide_exists_and_is_linked_from_the_site() -> None:
    guide = ROOT / "how-to" / "how-to-make-a-submission.md"
    assert guide.is_file()

    html = render_html([])
    assert (
        'href="https://github.com/wd7512/minimal-nn-competition/blob/main/'
        'how-to/how-to-make-a-submission.md"'
    ) in html
