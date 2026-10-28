"""Featured cards describe current passing leaders and all open benchmarks."""

from dataclasses import dataclass, field
from html.parser import HTMLParser

from minimal_nn_competition.leaderboard.render import render_html
from minimal_nn_competition.runner.entry import BenchmarkResult


@dataclass
class _Card:
    text: str = ""
    links: list[str] = field(default_factory=list)


class _FeaturedCards(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__()
        self.cards: dict[str, _Card] = {}
        self.current: _Card | None = None
        self.depth = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "data-featured" in attributes:
            self.current = _Card()
            self.cards[attributes["data-featured"]] = self.current
        if self.current is not None:
            if tag not in {"br", "hr", "img", "input", "wbr"}:
                self.depth += 1
            if tag == "a":
                self.current.links.append(attributes["href"])

    def handle_endtag(self, tag):
        if self.current is not None:
            self.depth -= 1
            if self.depth == 0:
                self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current.text += data


def _result(submission, benchmark, submitted_at, *, passed=True, parameters=10, name=None):
    return BenchmarkResult(
        submission,
        benchmark,
        1.0 if passed else 0.5,
        passed,
        parameters,
        {},
        name=name or submission,
        author="A <writer> & collaborator",
        submitted_at=submitted_at,
    )


def test_featured_cards_select_current_passing_leaders_and_escape_content() -> None:
    results = [
        _result("oldest-winner", "mnist", "2026-10-22"),
        _result("older-nonwinner", "mnist", "2026-10-01", parameters=20),
        _result("newer-nonwinner", "mnist", "2026-10-28", parameters=30),
        _result("newest-winner", "fashion-mnist", "2026-10-27", name="New <leader> & co"),
        _result("failed-model", "pendulum", "2026-10-29", passed=False),
        _result("undated-winner", "cartpole", None),
        _result("invalid-date-winner", "chess-eval-10k", "not-a-date"),
    ]
    explicit_link = "submissions/newest.html?x=1&y=2"
    html = render_html(results, {("fashion-mnist", "newest-winner"): explicit_link})
    cards = _FeaturedCards(html).cards

    assert "oldest-winner" in cards["oldest"].text
    assert "New <leader> & co" in cards["newest"].text
    assert "New <leader> & co" not in html
    assert "New &lt;leader&gt; &amp; co" in html
    assert explicit_link in cards["newest"].links
    assert "#panel-mnist" in cards["oldest"].links
    for excluded in (
        "older-nonwinner",
        "newer-nonwinner",
        "failed-model",
        "undated-winner",
        "invalid-date-winner",
    ):
        assert excluded not in cards["newest"].text + cards["oldest"].text


def test_featured_cards_break_date_ties_independently_of_result_order() -> None:
    results = [
        _result("mnist-winner", "mnist", "2026-10-23"),
        _result("cartpole-winner", "cartpole", "2026-10-23"),
    ]
    for ordered in (results, list(reversed(results))):
        cards = _FeaturedCards(render_html(ordered)).cards
        assert "mnist-winner" in cards["newest"].text
        assert "cartpole-winner" in cards["oldest"].text


def test_featured_cards_omit_models_without_a_valid_dated_passing_leader() -> None:
    results = [
        _result("invalid-winner", "mnist", "2026-10-32"),
        _result("dated-nonwinner", "mnist", "2026-10-22", parameters=20),
        _result("undated-winner", "cartpole", None),
        _result("failed-model", "pendulum", "2026-10-23", passed=False),
    ]
    assert set(_FeaturedCards(render_html(results)).cards) == {"challenges"}
    assert set(_FeaturedCards(render_html([])).cards) == {"challenges"}


def test_open_challenges_include_absent_benchmarks_with_working_destinations() -> None:
    results = [
        _result("mnist-winner", "mnist", None),
        _result("cartpole-winner", "cartpole", "invalid"),
        _result("snake-attempt", "snake-simple-8x8", "2026-10-22", passed=False),
        _result("custom-winner", "custom-benchmark", "2026-10-22"),
    ]
    html = render_html(results)
    card = _FeaturedCards(html).cards["challenges"]

    assert "10" in card.text
    assert "Find your next challenge" in card.text
    assert "Simple 8×8" in card.text
    assert "MNIST Shuffled" in card.text
    assert len(card.links) == 10
    assert "#panel-snake-simple-8x8" in card.links
    assert "#panel-mnist" not in card.links
    assert "#panel-cartpole" not in card.links
    assert "#panel-custom-benchmark" not in card.links
    for benchmark in ("fashion-mnist", "mnist-shuffled", "fashion-mnist-shuffled"):
        assert f"#panel-{benchmark}" in card.links
        assert f'id="panel-{benchmark}"' in html
    guide_prefix = "https://github.com/wd7512/minimal-nn-competition/blob/main/how-to/"
    assert guide_prefix + "how-to-make-a-snake-submission.md" in card.links
    assert guide_prefix + "how-to-make-a-chess-submission.md" in card.links
    assert guide_prefix + "how-to-make-a-control-submission.md" in card.links
    empty_card = _FeaturedCards(render_html([])).cards["challenges"]
    assert len(empty_card.links) == 12
    assert guide_prefix + "how-to-make-an-image-compression-submission.md" in empty_card.links
