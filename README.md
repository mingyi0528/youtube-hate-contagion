# Codebook

This folder contains the coding materials used for LLM-based classification
of **incivility** and **intolerance** in Korean-language YouTube comments
from far-right channels during the December 2024 martial law declaration
and impeachment crisis.

## Contents

| File | Description |
|---|---|
| `codebook_v3.py` | The exact system-instruction text (prompt) passed to the LLM classifier, plus the shared slur and trigger-word lexicons it references. This is the operational definition of both constructs as actually implemented, not a paraphrase of it. |

## What's in `codebook_v3.py`

- **`INCIVILITY_CODEBOOK`** — operational definition and coding instructions
  for incivility, organized around 5 indicators (profanity/slur use,
  personal attack, dishonesty attack, ridicule, rude command/punitive
  language), with a worked example table and an explicit list of
  boundary cases that do **not** count as incivility (factual statements,
  policy criticism, unaggravated emotional expression, neutral political
  labels, etc.).
- **`INTOLERANCE_CODEBOOK`** — operational definition and coding
  instructions for intolerance, organized around 4 indicators (slur,
  stereotype, dehumanization, exclusion/violence), again with a worked
  example table and boundary cases (individual-targeted insults vs.
  group-targeted language, simple party-dissolution statements vs.
  dissolution combined with punitive language, etc.).
- **`DUAL_CODEBOOK`** — the combined instruction used when both
  constructs are coded jointly in a single pass, including a joint
  example table showing how the two labels can diverge on the same
  comment (incivility without intolerance, intolerance without
  incivility, both, neither).
- **`SLUR_LEXICON`** — the group-slur vocabulary underlying incivility
  indicator 1 and intolerance indicator 1, organized by target category
  (political affiliation, martial-law/impeachment-specific terms,
  regional origin, gender, national origin/ethnicity, religion, age,
  disability, occupation).
- **`TRIGGER_LEXICON`** — the full escalation-trigger vocabulary (slurs +
  dehumanizing terms + exclusion/violence phrases), approximately 118
  items in total. This is the complete list summarized at a higher level
  in the manuscript's appendix.

## Version history

- **v2 → v3 (2026-05-05):** Added group-slur use as a qualifying
  condition under incivility indicator 1. Under v1/v2, a comment
  containing only a group slur (no separate profanity, personal attack,
  etc.) was coded incivility = 0; manual validation review found this
  systematically too restrictive, since a group slur is itself an
  insulting form of address. Intolerance coding rules are unchanged —
  slurs already counted as intolerance = 1, and now also count as
  incivility = 1 on the same comment.

## Correspondence to the manuscript

- Operational definitions and coding criteria summarized in the methods
  section correspond to `INCIVILITY_CODEBOOK` / `INTOLERANCE_CODEBOOK`
  above.
- The condensed trigger-word rules and three-tier summary table in the
  appendix correspond to `TRIGGER_LEXICON`; this file contains the full,
  unabridged list.

## What is *not* included here

Raw YouTube comment text, author-identifying information, comment IDs,
and the full comment-level labeled dataset are not included or
redistributed here, in accordance with research ethics and privacy
requirements described in the manuscript.
