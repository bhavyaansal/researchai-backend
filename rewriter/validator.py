"""
Validator: re-scores a rewritten paragraph against its matched source
to check whether it has dropped below the target similarity threshold.
"""
from scoring.similarity import lexical_score, combined_score
from search.embedder import compute_semantic_similarity


def validate_rewrite(rewritten_text: str, matched_source_text: str) -> dict:
    """
    Re-score the rewritten paragraph against the source it was originally
    flagged against. Returns the new scores so the caller can decide whether
    to accept this rewrite or retry.
    """
    if not matched_source_text:
        return {
            "lexical_score": 0.0,
            "semantic_score": 0.0,
            "combined_score": 0.0,
        }

    lex_s = lexical_score(rewritten_text, matched_source_text)
    sem_s = compute_semantic_similarity(rewritten_text, matched_source_text)
    comb_s = combined_score(lex_s, sem_s)

    return {
        "lexical_score": lex_s,
        "semantic_score": sem_s,
        "combined_score": comb_s,
    }

