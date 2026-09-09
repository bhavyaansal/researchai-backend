"""
Coordinate mapping: given the paragraphs already carry start_char/end_char
(from parser.split_into_paragraphs), this module's job is just to package
that span info together with its hybrid search match into a clean record
ready to be written to the database.
"""


def map_span_to_match(paragraph: dict, match: dict) -> dict:
    """
    Combine a paragraph's character coordinates with its best search match
    into a single record describing one flagged span.

    paragraph: {"text":..., "start_char":..., "end_char":...}
    match:     {"source_title":..., "source_text":..., "source_url":...,
                 "lexical_score":..., "semantic_score":..., "combined_score":...}
    """
    return {
        "original_text": paragraph["text"],
        "start_char": paragraph["start_char"],
        "end_char": paragraph["end_char"],
        "matched_source_title": match["source_title"],
        "matched_source_text": match["source_text"],
        "source_url": match.get("source_url"),
        "lexical_score": match["lexical_score"],
        "semantic_score": match["semantic_score"],
        "combined_score": match["combined_score"],
    }