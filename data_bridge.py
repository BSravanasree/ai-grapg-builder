"""
data_bridge.py — M1 → M2 Pipeline Connector

Reads the processed credit card dataset produced by Milestone 1 and feeds each
row through the entity/relation extraction engine, then loads all resulting
triples into Neo4j.

Usage:
    python data_bridge.py              # runs with default limit=100
    from data_bridge import run_pipeline
    run_pipeline(limit=500)            # process 500 rows
    run_pipeline(limit=None)           # process entire dataset
"""

import os
import sys
import pandas as pd

# ── Path setup so this file can be run from the project root OR milestone_2/ ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "milestone_2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "milestone_3"))

from entity_extractor import extract_graph_triples
from neo4j_loader import (
    test_connection,
    init_schema,
    create_graph,
    get_graph_stats,
    close_driver,
)

# ── Locate the processed CSV ───────────────────────────────────────────────────
CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "processed_creditcard.csv"),
    os.path.join(PROJECT_ROOT, "milestone_1", "processed_creditcard.csv"),
    os.path.join(PROJECT_ROOT, "processed_creditcard.csv"),
]


def _find_csv() -> str:
    """Return the path of the first existing processed CSV candidate."""
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "processed_creditcard.csv not found. "
        "Run milestone_1/notebook.ipynb first to generate it.\n"
        f"Searched in:\n" + "\n".join(f"  {p}" for p in CSV_CANDIDATES)
    )


# ── Row → Sentence ─────────────────────────────────────────────────────────────
def row_to_sentence(row: pd.Series) -> str:
    """
    Convert a single dataframe row into a human-readable sentence that the
    NLP pipeline can process.

    Handles both the raw creditcard.csv schema and the enriched schema
    produced by the M1 notebook (which adds fraud_label, risk_flag, etc.).
    """
    # ── Determine fraud label ────────────────────────────────────────────────
    if "fraud_label" in row.index:
        fraud_str = row["fraud_label"]                     # "Fraud" or "Normal"
    elif "is_fraud" in row.index:
        fraud_str = "Fraud" if int(row["is_fraud"]) == 1 else "Normal"
    elif "Class" in row.index:
        fraud_str = "Fraud" if int(row["Class"]) == 1 else "Normal"
    else:
        fraud_str = "Unknown"

    # ── Determine risk level ─────────────────────────────────────────────────
    if "risk_flag" in row.index:
        risk_str = "High Risk" if int(row["risk_flag"]) > 0 else "Low Risk"
    else:
        risk_str = "Unknown Risk"

    # ── Determine amount ─────────────────────────────────────────────────────
    amount = row.get("amount", row.get("Amount", 0.0))
    try:
        amount = float(amount)
    except (ValueError, TypeError):
        amount = 0.0

    # ── Determine transaction ID / node ─────────────────────────────────────
    tx_node = row.get("transaction_node", "")
    if not tx_node:
        tx_id = row.get("transaction_id", row.get("id", ""))
        tx_node = f"TX_{tx_id}" if tx_id else "TX_UNKNOWN"

    # ── Determine merchant ───────────────────────────────────────────────────
    merchant = row.get("merchant_node", row.get("merchant_category", "Unknown Merchant"))

    # ── Determine cardholder ─────────────────────────────────────────────────
    cardholder = row.get("cardholder_node", "Unknown Cardholder")

    # ── Build sentence ───────────────────────────────────────────────────────
    sentence = (
        f"{tx_node} is a {fraud_str} transaction of ${amount:.2f}. "
        f"Risk level is {risk_str}. "
        f"Merchant category is {merchant}. "
        f"Cardholder is {cardholder}."
    )
    return sentence


# ── Additional domain-specific triples from structured fields ──────────────────
def row_to_structured_triples(row: pd.Series) -> list:
    """
    Generate deterministic, structured (subject, relation, object) triples
    directly from column values — bypassing NLP for known fields.

    This guarantees at least one valid triple per row even when NLP
    does not find a relationship in the sentence.
    """
    triples = []

    tx_node = row.get("transaction_node", "")
    if not tx_node:
        tx_id = row.get("transaction_id", row.get("id", ""))
        tx_node = f"TX_{tx_id}" if tx_id else None

    merchant = row.get("merchant_node", row.get("merchant_category", ""))
    cardholder = row.get("cardholder_node", "")

    fraud_label = ""
    if "fraud_label" in row.index:
        fraud_label = str(row["fraud_label"])
    elif "is_fraud" in row.index:
        fraud_label = "Fraud" if int(row["is_fraud"]) == 1 else "Normal"
    elif "Class" in row.index:
        fraud_label = "Fraud" if int(row["Class"]) == 1 else "Normal"

    if tx_node and merchant:
        triples.append((tx_node, "TRANSACTED_AT", str(merchant)))
    if tx_node and cardholder:
        triples.append((str(cardholder), "BELONGS_TO", tx_node))
    if tx_node and fraud_label:
        triples.append((tx_node, "FLAGGED_AS", fraud_label))

    return triples


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(limit: int = 100):
    """
    Full M1 → M2 pipeline:
      1. Load processed CSV
      2. Convert each row to a sentence
      3. Extract entities & relations via spaCy
      4. Build additional structured triples from column values
      5. Write all triples to Neo4j in batches of 50

    Args:
        limit: Maximum number of rows to process.
               Set to None to process the entire dataset (~10,000 rows).
    """
    print("\n" + "=" * 60)
    print("  data_bridge.py — M1 → M2 Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    csv_path = _find_csv()
    print(f"\n[pipeline] 📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)
    total = len(df)
    print(f"[pipeline] ✅ Loaded {total:,} rows.")

    # ── Step 2: Connect to Neo4j ─────────────────────────────────────────────
    print("\n[pipeline] 🔌 Connecting to Neo4j...")
    if not test_connection():
        print("[pipeline] ❌ Cannot reach Neo4j. Aborting.")
        return
    init_schema()

    # ── Step 3: Process in batches ───────────────────────────────────────────
    BATCH_SIZE = 50
    all_triples = []

    for i, (_, row) in enumerate(df.iterrows()):
        # Natural language sentence → NLP triples
        sentence = row_to_sentence(row)
        nlp_result = extract_graph_triples(sentence)
        nlp_triples = nlp_result.get("relations", [])

        # Structured triples from column values (always valid)
        structured_triples = row_to_structured_triples(row)

        all_triples.extend(nlp_triples)
        all_triples.extend(structured_triples)

        # Flush every BATCH_SIZE rows
        if len(all_triples) >= BATCH_SIZE or (i + 1) == total:
            create_graph(all_triples)
            all_triples = []

        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            print(f"[pipeline] ⏳ Processed {i + 1:,} / {total:,} rows...")

    # ── Step 4: Print summary ────────────────────────────────────────────────
    stats = get_graph_stats()
    print("\n" + "=" * 60)
    print("  Pipeline Complete — Summary")
    print("=" * 60)
    print(f"  Rows processed   : {total:,}")
    print(f"  Nodes in graph   : {stats['node_count']:,}")
    print(f"  Relationships    : {stats['relationship_count']:,}")
    print(f"  Fraud nodes      : {stats['fraud_node_count']:,}")
    print("=" * 60 + "\n")

    close_driver()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(limit=100)
