"""Module 2: Job Description Language Audit.

audit_jd(text) -> JDReport with:
* gender-coded words (masculine / feminine), using the word lists from Kat Matfield's Gender
  Decoder (MIT), which implement Gaucher, Friesen & Kay (2011), "Evidence that gendered wording
  in job advertisements exists and sustains gender inequality", JPSP 101(1).
* flags from jd_flags.yaml (age-coded, exclusionary, disability, credential, legal)
* pay-range check (Illinois requires pay scale + benefits in postings for employers with 15+
  employees since Jan 1, 2025; 820 ILCS 112/10(b-25))
* readability: Flesch-Kincaid grade and Flesch reading ease (built-in syllable counter, no downloads)
* a 0-100 JD Equity Score and a suggested rewrite with safe word-level substitutions

All of it is rule-based and explainable: every point lost is tied to a specific phrase.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Gender-coded word stems (Gender Decoder / Gaucher et al. 2011). A stem matches any word
# that starts with it, e.g. "compet" -> competitive, competition.
# ---------------------------------------------------------------------------

MASCULINE_STEMS = [
    "active", "adventurous", "aggress", "ambitio", "analy", "assert", "athlet", "autonom", "battle",
    "boast", "challeng", "champion", "compet", "confident", "courag", "decide", "decision", "decisive",
    "defend", "determin", "domina", "driven", "fearless", "fight", "force", "greedy", "head-strong",
    "headstrong", "hierarch", "hostil", "impulsive", "independen", "individual", "intellect", "lead",
    "logic", "objective", "opinion", "outspoken", "persist", "principle", "reckless", "self-confiden",
    "self-relian", "self-sufficien", "selfconfiden", "selfrelian", "selfsufficien", "stubborn",
    "superior", "unreasonab",
]
FEMININE_STEMS = [
    "agree", "affectionate", "child", "cheer", "collab", "commit", "communal", "compassion", "connect",
    "considerate", "cooperat", "co-operat", "depend", "emotiona", "empath", "feel", "flatterable",
    "gentle", "honest", "interpersonal", "interdependen", "interpersona", "inter-personal",
    "inter-dependen", "inter-persona", "kind", "kinship", "loyal", "modesty", "nag", "nurtur",
    "pleasant", "polite", "quiet", "respon", "sensitiv", "submissive", "support", "sympath", "tender",
    "together", "trust", "understand", "warm", "whin", "enthusias", "inclusive", "yield", "share",
    "sharin",
]
# Words that start with a stem but are a different word entirely (not a judgment call about
# context; the published lists are otherwise used as-is so results match the Gender Decoder).
STEM_EXCEPTIONS = {
    "kindergarten", "kindergartens", "committee", "committees", "connecticut",
    # benefits / EEO boilerplate where the word means something else:
    "dependents",   # "coverage for you and your dependents" (not "dependable")
    "depending",    # "pay depending on experience"
    "individuals",  # "individuals with disabilities" = people (not "individualistic")
}

PAY_PATTERNS = [
    r"\$\s?\d[\d,]*(\.\d+)?\s*(k|K)?\s*(-|–|—|to)\s*\$?\s?\d[\d,]*",  # $50,000 - $60,000 / $25-$30
    r"\$\s?\d[\d,]*(\.\d+)?\s*(k|K)?\s*(per|/|an|a)\s*(hour|hr|year|yr|annum|month)",
    r"(salary|pay|compensation|wage|hourly rate)\s*(range|scale)?\s*[:\-]?\s*\$",
]
BENEFIT_WORDS = r"benefits?|health insurance|medical|dental|401\(?k\)?|paid time off|pto|retirement"


@lru_cache(maxsize=1)
def load_flags(path: str | None = None) -> list[dict]:
    p = Path(path) if path else Path(__file__).with_name("jd_flags.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))["flags"]


# ---------------------------------------------------------------------------
# Readability (Flesch 1948; Kincaid et al. 1975)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def count_syllables(word: str) -> int:
    """Heuristic English syllable count (vowel groups, silent final e, -le endings)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|[^laeiouy]ed|[^laeiouy]e)$", "", w)  # silent e / -es / -ed
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(groups))


def sentences(text: str) -> list[str]:
    # bullet lines count as their own sentences (job ads are mostly bullets)
    parts = re.split(r"(?<=[.!?])\s+|\n+\s*(?:[-•*·▪●]|\d+[.)])?\s*", text)
    return [p for p in (s.strip() for s in parts) if _WORD.search(p or "")]


def readability(text: str) -> dict[str, float]:
    words = _WORD.findall(text)
    sents = sentences(text)
    if not words or not sents:
        return {"words": 0, "sentences": 0, "fk_grade": 0.0, "reading_ease": 100.0}
    syl = sum(count_syllables(w) for w in words)
    wps, spw = len(words) / len(sents), syl / len(words)
    return {
        "words": len(words),
        "sentences": len(sents),
        "fk_grade": round(0.39 * wps + 11.8 * spw - 15.59, 1),
        "reading_ease": round(206.835 - 1.015 * wps - 84.6 * spw, 1),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    start: int
    end: int
    text: str
    kind: str          # "masculine" | "feminine" | flag id
    category: str      # "gender-coded" or flag category
    severity: str      # high | medium | low | info
    why: str
    suggestion: str


@dataclass
class JDReport:
    text: str
    hits: list[Hit] = field(default_factory=list)
    masculine: list[str] = field(default_factory=list)
    feminine: list[str] = field(default_factory=list)
    readability: dict = field(default_factory=dict)
    has_pay_range: bool = False
    has_benefits: bool = False
    score: int = 100
    deductions: list[tuple[str, int]] = field(default_factory=list)
    rewrite: str = ""

    @property
    def gender_coding(self) -> str:
        m, f = len(self.masculine), len(self.feminine)
        d = m - f
        if m == 0 and f == 0:
            return "neutral"
        if d >= 3:
            return "strongly masculine-coded"
        if d >= 1:
            return "masculine-coded"
        if d <= -3:
            return "strongly feminine-coded"
        if d <= -1:
            return "feminine-coded"
        return "neutral"

    @property
    def flags(self) -> list[Hit]:
        return [h for h in self.hits if h.category != "gender-coded"]


def _gender_hits(text: str) -> tuple[list[Hit], list[str], list[str]]:
    hits, masc, fem = [], [], []
    for m in re.finditer(r"[A-Za-z][A-Za-z-]*", text):
        w = m.group(0).lower()
        if w in STEM_EXCEPTIONS:
            continue
        for stems, kind, out in ((MASCULINE_STEMS, "masculine", masc), (FEMININE_STEMS, "feminine", fem)):
            if any(w.startswith(s) for s in stems):
                out.append(w)
                hits.append(Hit(m.start(), m.end(), m.group(0), kind, "gender-coded", "info",
                                f"{kind.capitalize()}-coded word (Gaucher et al. 2011).",
                                "Balance masculine- and feminine-coded wording; it's the overall mix that matters."))
                break
    return hits, masc, fem


def _flag_hits(text: str) -> list[Hit]:
    hits = []
    for f in load_flags():
        if f.get("unless") and re.search(f["unless"], text, re.I):
            continue
        for pat in f["patterns"]:
            rx = pat if pat.startswith("\\b") else rf"\b(?:{pat})\b"
            for m in re.finditer(rx, text, re.I):
                hits.append(Hit(m.start(), m.end(), m.group(0), f["id"], f["category"], f["severity"],
                                f["why"], f["suggestion"]))
    # drop overlaps (keep the first / longest)
    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    kept, last_end = [], -1
    for h in hits:
        if h.start >= last_end:
            kept.append(h)
            last_end = h.end
    return kept


SEVERITY_POINTS = {"high": 12, "medium": 6, "low": 3}


def audit_jd(text: str) -> JDReport:
    text = text.replace("\r\n", "\n")
    r = JDReport(text=text)
    g_hits, r.masculine, r.feminine = _gender_hits(text)
    f_hits = _flag_hits(text)
    r.hits = sorted(g_hits + f_hits, key=lambda h: h.start)
    r.readability = readability(text)
    r.has_pay_range = any(re.search(p, text, re.I) for p in PAY_PATTERNS)
    r.has_benefits = bool(re.search(BENEFIT_WORDS, text, re.I))

    # --- JD Equity Score: start at 100, every deduction is itemized -------------------------
    ded: list[tuple[str, int]] = []
    flag_pts = 0
    for h in f_hits:
        flag_pts += SEVERITY_POINTS[h.severity]
    if flag_pts:
        ded.append((f"{len(f_hits)} flagged phrase(s)", min(flag_pts, 45)))
    imbalance = len(r.masculine) - len(r.feminine)
    if imbalance >= 3:
        ded.append((f"Masculine-coded wording outweighs feminine-coded by {imbalance}", min(3 * imbalance, 15)))
    grade = r.readability["fk_grade"]
    if grade > 10:
        ded.append((f"Reading level grade {grade} (aim for 10 or below)", min(round(2 * (grade - 10)), 15)))
    if not r.has_pay_range:
        ded.append(("No pay range (required in Illinois for employers with 15+ employees)", 10))
    elif not r.has_benefits:
        ded.append(("Pay listed but no benefits description (Illinois requires both)", 3))
    if r.readability["words"] > 900:
        ded.append((f"Long posting ({r.readability['words']} words)", 5))
    r.deductions = ded
    r.score = max(0, 100 - sum(p for _, p in ded))
    r.rewrite = suggested_rewrite(text, f_hits)
    return r


def suggested_rewrite(text: str, flag_hits: list[Hit]) -> str:
    """Apply only the safe word-level replacements defined in the YAML (case-preserving)."""
    repl = {f["id"]: f.get("replace") for f in load_flags()}
    out, pos = [], 0
    for h in sorted(flag_hits, key=lambda h: h.start):
        new = repl.get(h.kind)
        if not new:
            continue
        out.append(text[pos:h.start])
        out.append(new[0].upper() + new[1:] if h.text[:1].isupper() else new)
        pos = h.end
    out.append(text[pos:])
    rewritten = "".join(out)
    if repl and re.search(r"degree required", rewritten, re.I) and not re.search(r"equivalent", rewritten, re.I):
        rewritten = re.sub(r"(degree)( is)? required", r"\1 or equivalent experience\2 required", rewritten,
                           count=1, flags=re.I)
    return rewritten


# ---------------------------------------------------------------------------
# Presentation helpers (used by the app; plain functions so they're testable)
# ---------------------------------------------------------------------------

HIGHLIGHT = {
    "masculine": "#cfe1f7", "feminine": "#f9d9e6",
    "high": "#f6c9c6", "medium": "#fbe3b6", "low": "#e7e6df",
}


def highlighted_html(report: JDReport) -> str:
    """Original text with hits wrapped in <mark title=...>; safe-escaped."""
    parts, pos = [], 0
    for h in report.hits:
        if h.start < pos:
            continue
        parts.append(html.escape(report.text[pos:h.start]))
        color = HIGHLIGHT[h.kind] if h.category == "gender-coded" else HIGHLIGHT[h.severity]
        tip = html.escape(f"{h.category}: {h.why} Suggestion: {h.suggestion}")
        parts.append(f'<mark style="background:{color};color:#0b0b0b;border-radius:3px;padding:0 2px" '
                     f'title="{tip}">{html.escape(h.text)}</mark>')
        pos = h.end
    parts.append(html.escape(report.text[pos:]))
    return "<div style='white-space:pre-wrap;line-height:1.6'>" + "".join(parts) + "</div>"


def report_markdown(report: JDReport, title: str = "Job description") -> str:
    lines = [f"# FairHire AI · JD Language Audit: {title}", "",
             f"**JD Equity Score: {report.score}/100**", "",
             f"- Gender coding: {report.gender_coding} "
             f"({len(report.masculine)} masculine-coded, {len(report.feminine)} feminine-coded words)",
             f"- Reading level: grade {report.readability['fk_grade']} "
             f"(Flesch reading ease {report.readability['reading_ease']})",
             f"- Pay range listed: {'yes' if report.has_pay_range else 'no'}", ""]
    if report.deductions:
        lines += ["## Score deductions", *[f"- {d} (−{p})" for d, p in report.deductions], ""]
    if report.flags:
        lines += ["## Flagged phrases", "| Phrase | Category | Severity | Why | Suggestion |", "|---|---|---|---|---|"]
        for h in report.flags:
            lines.append(f"| {h.text} | {h.category} | {h.severity} | {h.why} | {h.suggestion} |")
        lines.append("")
    lines += ["## Gender-coded words", f"- Masculine-coded: {', '.join(sorted(set(report.masculine))) or 'none'}",
              f"- Feminine-coded: {', '.join(sorted(set(report.feminine))) or 'none'}", "",
              "_Statistical and language analysis only, not legal advice._"]
    return "\n".join(lines)


def read_docx(data: bytes) -> str:
    import io

    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)
