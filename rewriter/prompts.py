"""
Prompt templates for the rewriting LLM.
Encodes the "smart constraints" - preserve citations, formulas, and domain terms
while substantially changing structure and wording.
"""

SYSTEM_PROMPT = """You are a precise academic rewriter. Your job is to rewrite a paragraph \
to reduce textual similarity to its source while preserving exact meaning.

Rules you MUST follow:
1. Preserve ALL citations exactly as written (e.g. "(Smith, 2020)", "[12]").
2. Preserve ALL mathematical formulas, equations, and symbols exactly as written.
3. Preserve ALL domain-specific technical terms and proper nouns - do not paraphrase jargon.
4. Substantially change sentence structure, word choice, and phrasing elsewhere.
5. Do not change the factual meaning of the paragraph.
6. Do not add, remove, or fabricate any claims, numbers, or data.
7. Return ONLY the rewritten paragraph. No preamble, no explanation, no quotation marks.
"""

REWRITE_USER_TEMPLATE = """Original paragraph (flagged as {similarity_pct}% similar to a source):

\"\"\"{original_text}\"\"\"

Rewrite it now, following all rules above."""

RETRY_USER_TEMPLATE = """Your previous rewrite was still too similar to the source \
({previous_score_pct}% similarity, target is below {target_pct}%).

Original paragraph:
\"\"\"{original_text}\"\"\"

Your previous attempt:
\"\"\"{previous_attempt}\"\"\"

Rewrite it again with MORE structural and lexical change this time, while still \
following all the preservation rules (citations, formulas, technical terms, meaning)."""


def build_rewrite_prompt(original_text: str, similarity_score: float) -> str:
    return REWRITE_USER_TEMPLATE.format(
        original_text=original_text,
        similarity_pct=round(similarity_score * 100, 1),
    )


def build_retry_prompt(original_text: str, previous_attempt: str, previous_score: float, target: float) -> str:
    return RETRY_USER_TEMPLATE.format(
        original_text=original_text,
        previous_attempt=previous_attempt,
        previous_score_pct=round(previous_score * 100, 1),
        target_pct=round(target * 100, 1),
    )
