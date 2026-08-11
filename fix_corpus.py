# Run from backend folder with venv activated:
# python -c "exec(open('fix_corpus.py').read())"

import sys
sys.path.insert(0, '.')
from db.models import SessionLocal, SourceDocument, init_db

init_db()
db = SessionLocal()

# Show what's in corpus
all_sources = db.query(SourceDocument).all()
print(f"Total corpus entries: {len(all_sources)}")
for s in all_sources:
    print(f"  [{s.id[:8]}] {s.source_title} | {s.sentence_text[:60]}...")

# Delete anything that looks like a user upload
# (titles that are filenames like .pdf .docx .txt)
import re
bad = db.query(SourceDocument).filter(
    SourceDocument.source_title.op('REGEXP')(r'\.(pdf|docx|txt)$')
).all()

# SQLite doesn't support REGEXP — use Python filter instead
bad = [s for s in all_sources if re.search(r'\.(pdf|docx|txt|PDF|DOCX|TXT)$', s.source_title)]
print(f"\nFound {len(bad)} polluted entries to remove:")
for b in bad:
    print(f"  Removing: {b.source_title}")
    db.delete(b)

db.commit()
print("Done. Corpus cleaned.")
db.close()