"""Debug script to trace pipeline scoring."""
from ingestion.parser import extract_text, split_into_paragraphs
from search.lexical import build_bm25_index, search_lexical
from search.hybrid import hybrid_search
from search.semantic import _build_index
from scoring.similarity import global_document_score
from db.models import SessionLocal
from config import settings

db = SessionLocal()
_build_index()
bm25_index, sources = build_bm25_index(db)

# Simulate a climate change doc
full_text = """Climate Change and Global Ecosystems

Abstract

Climate change represents one of the most critical challenges facing humanity in the twenty-first century. Rising global temperatures, driven by increased concentrations of greenhouse gases, are fundamentally altering ecosystems, weather patterns, and biodiversity across the planet.

Introduction

The Intergovernmental Panel on Climate Change (IPCC) has documented that global surface temperatures have increased by approximately 1.1 degrees Celsius above pre-industrial levels. This warming trend is accelerating due to continued fossil fuel combustion, deforestation, and industrial emissions.

Conclusion

Based on the evidence reviewed, immediate and sustained action to reduce greenhouse gas emissions is essential to prevent catastrophic environmental changes and protect the wellbeing of future generations."""

paragraphs = split_into_paragraphs(full_text)
print("Paragraphs found:", len(paragraphs))
print("SIMILARITY_FLAG_THRESHOLD:", settings.SIMILARITY_FLAG_THRESHOLD)
print()

flagged = []
for para in paragraphs:
    text = para["text"].strip()
    if len(text) < 30:
        print("SKIP (too short):", text[:40])
        continue

    match = hybrid_search(text, bm25_index, sources, top_k=5)
    print("Para:", text[:70])
    if match:
        cs = match["combined_score"]
        ls = match["lexical_score"]
        ss = match["semantic_score"]
        print("  combined={:.4f}, lex={:.4f}, sem={:.4f}".format(cs, ls, ss))
        print("  Source:", match["source_title"])
        print("  Flagged:", cs >= settings.SIMILARITY_FLAG_THRESHOLD)
        if cs >= settings.SIMILARITY_FLAG_THRESHOLD:
            flagged.append((para, match))
    else:
        print("  No match (None)")
    print()

print("Total flagged:", len(flagged))

if flagged:
    all_scores = [m["combined_score"] for _, m in flagged]
    all_lengths = [len(p["text"]) for p, _ in flagged]
    g_score = global_document_score(all_scores, all_lengths)
    print("Global document score:", g_score)

db.close()
