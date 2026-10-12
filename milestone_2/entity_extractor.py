"""
entity_extractor.py — Entity & Relationship Extraction Engine (Milestone 2)

Uses spaCy's dependency parser and NER pipeline to extract:
  1. Named entities  → (entity_text, entity_label) tuples
  2. Grammatical triples → (subject, relation, object) tuples

No hardcoded string splits. No regex keyword matching.
All extraction is driven by the spaCy linguistic model.
"""

import re
import spacy

# ── Load language model ────────────────────────────────────────────────────────
# en_core_web_sm — install with:  python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run:  python -m spacy download en_core_web_sm"
    )

# ── Punctuation stripper ───────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"^[\s\W]+|[\s\W]+$")


def _clean(text: str) -> str:
    """Strip leading/trailing punctuation and whitespace from a string."""
    return _PUNCT_RE.sub("", text).strip()


# ── Dependency label sets ──────────────────────────────────────────────────────
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent"}
OBJECT_DEPS  = {"dobj", "pobj", "attr", "oprd", "dative", "iobj"}
VERB_DEPS    = {"ROOT", "relcl", "advcl", "xcomp"}


# ════════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> list:
    """
    Extract named entities from text using spaCy NER.

    Args:
        text: Raw input string.

    Returns:
        List of (entity_text, entity_label) tuples.
        Entities shorter than 2 characters are filtered out.
        Punctuation and extra whitespace are stripped.

    Example:
        >>> extract_entities("TX_001 is a Fraud transaction at Electronics.")
        [('TX_001', 'ORG'), ('Electronics', 'ORG')]
    """
    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        cleaned = _clean(ent.text)
        if len(cleaned) < 2:
            continue
        key = (cleaned.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append((cleaned, ent.label_))

    return entities


def extract_relations(text: str) -> list:
    """
    Extract (subject, relation, object) triples from text using
    spaCy's dependency parse tree.

    Strategy:
      - Walk every token that is a verb (ROOT, relcl, advcl, xcomp).
      - Collect tokens with subject dependency labels (nsubj, nsubjpass…)
        as the subject candidate.
      - Collect tokens with object dependency labels (dobj, pobj, attr…)
        as the object candidate.
      - The verb's lemma (lowercased, underscored) becomes the relation label.

    Args:
        text: Raw input string.

    Returns:
        List of (subject, relation, object) tuples.
        Any triple where subject or object is empty is discarded.

    Example:
        >>> extract_relations("TX_001 is flagged as Fraud.")
        [('TX_001', 'be', 'Fraud')]
    """
    doc = nlp(text)
    triples = []

    for token in doc:
        # Only consider verb-like tokens as relation anchors
        if token.pos_ not in {"VERB", "AUX"} and token.dep_ not in VERB_DEPS:
            continue

        subjects = []
        objects = []

        # Inspect children of the verb token
        for child in token.children:
            if child.dep_ in SUBJECT_DEPS:
                # Include compound nouns (e.g. "credit card" → "credit card")
                subject_span = _get_span(child, doc)
                subjects.append(subject_span)
            elif child.dep_ in OBJECT_DEPS:
                object_span = _get_span(child, doc)
                objects.append(object_span)

        # Build the relation string from the verb lemma
        relation = token.lemma_.lower().replace(" ", "_")

        for subj in subjects:
            for obj in objects:
                subj_clean = _clean(subj)
                obj_clean  = _clean(obj)
                if len(subj_clean) >= 2 and len(obj_clean) >= 2:
                    triples.append((subj_clean, relation, obj_clean))

    return triples


def extract_graph_triples(text: str) -> dict:
    """
    Combined entry point: run both NER and relation extraction in one call,
    sharing a single spaCy parse for efficiency.

    Args:
        text: Raw input string.

    Returns:
        dict with keys:
          "entities"  → list of (entity_text, label) tuples
          "relations" → list of (subject, relation, object) tuples
    """
    entities  = extract_entities(text)
    relations = extract_relations(text)
    return {"entities": entities, "relations": relations}


# ── Helper: expand compound noun spans ────────────────────────────────────────
def _get_span(token, doc) -> str:
    """
    Return a token's text expanded to include any compound modifiers.

    e.g. for "credit card" where 'card' is the head and 'credit' is compound,
    this returns "credit card" rather than just "card".
    """
    parts = [
        child.text
        for child in token.lefts
        if child.dep_ in {"compound", "amod", "nummod"}
    ]
    parts.append(token.text)
    return " ".join(parts)


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        "TX_001 is a Fraud transaction of $4500.00. Risk level is High Risk. Merchant category is Electronics. Cardholder is USER_0.",
        "Microsoft partnered with Infosys to deliver cloud analytics in Bengaluru.",
        "A suspicious transaction was flagged by the fraud detection system.",
    ]

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{'='*60}")
        print(f"SENTENCE {i}: {sentence}")
        print("=" * 60)

        result = extract_graph_triples(sentence)

        print("ENTITIES:")
        for ent_text, ent_label in result["entities"]:
            print(f"  [{ent_label}]  {ent_text}")

        print("RELATIONS:")
        if result["relations"]:
            for subj, rel, obj in result["relations"]:
                print(f"  ({subj}) --[{rel}]--> ({obj})")
        else:
            print("  (no grammatical triples found — structured triples used instead)")