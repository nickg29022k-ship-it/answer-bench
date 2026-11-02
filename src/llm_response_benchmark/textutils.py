"""Small text helpers shared by the data generator and the heuristic judge."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but of to in on at for with by from as is are was were be been being it its
    this that these those there their they them you your i we our he she his her so not no do does
    did can could should would will may might than then also about into over under more most such
    which who whom what when where why how all any each every some only very just""".split()
)

_TOKEN = re.compile(r"[a-z0-9$%][a-z0-9$%.,'-]*")


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        t = t.strip(".,'-")
        if t and t not in STOPWORDS:
            out.append(t)
    return out


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w and w not in {"-", "*"}])


def bullet_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", ln)]


def is_bulleted(text: str) -> bool:
    return len(bullet_lines(text)) >= 2


def overlap(needle: str, haystack_tokens: set[str]) -> float:
    """Fraction of content tokens in `needle` that appear in the haystack token set."""
    nt = set(tokens(needle))
    if not nt:
        return 0.0
    return len(nt & haystack_tokens) / len(nt)


def check_constraints(text: str, constraints) -> list[str]:
    """Return a list of violated constraints (empty list means all satisfied)."""
    violations = []
    if constraints.max_words is not None:
        wc = word_count(text)
        if wc > constraints.max_words:
            violations.append(f"over word limit ({wc} > {constraints.max_words})")
    bullets = bullet_lines(text)
    if constraints.format == "bullets" and len(bullets) < 2:
        violations.append("expected bullet points")
    if constraints.format == "paragraph" and len(bullets) >= 2:
        violations.append("expected a paragraph, got bullets")
    if constraints.num_bullets is not None and len(bullets) >= 2 and len(bullets) != constraints.num_bullets:
        violations.append(f"expected {constraints.num_bullets} bullets, got {len(bullets)}")
    return violations


def shorten(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."
