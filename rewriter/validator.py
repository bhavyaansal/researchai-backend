"""
Validator: re-scores a rewritten paragraph against its matched source
to check whether it has dropped below the target similarity threshold.
"""
from scoring.similarity import lexical_score, semantic_score, combined_score
from search.embedder import embed_text


def validate_rewrite(rewritten_text: str, matched_source_text: str) -> dict:
    """
    Re-score the rewritten paragraph against the source it was originally
    flagged against. Returns the new scores so the caller can decide whether
    to accept this rewrite or retry.
    """
    lex_s = lexical_score(rewritten_text, matched_source_text)

    emb_rewritten = embed_text(rewritten_text)
    emb_source = embed_text(matched_source_text)
    sem_s = semantic_score(emb_rewritten, emb_source)

    comb_s = combined_score(lex_s, sem_s)

    return {
        "lexical_score": lex_s,
        "semantic_score": sem_s,
        "combined_score": comb_s,
    }
