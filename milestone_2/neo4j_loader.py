"""
neo4j_loader.py — Graph Construction & Storage Hub
Milestone 2: Loads entity-relation triples into Neo4j.

Security fixes applied:
  - Credentials loaded from .env via python-dotenv (no hardcoded passwords)
  - Cypher injection fixed: relationship types validated against ALLOWED_REL_TYPES
    allowlist before being embedded in queries. Node names are fully parameterized.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ── Allowlist of permitted relationship types ─────────────────────────────────
# Only types in this set can ever be written to the graph.
# This completely eliminates Cypher injection via the rel parameter.
ALLOWED_REL_TYPES = {
    "PARTNERED_WITH",
    "LOCATED_IN",
    "WORKS_FOR",
    "OWNS",
    "RELATED_TO",
    "FLAGGED_AS",
    "PROCESSED_BY",
    "BELONGS_TO",
    "TRANSACTED_AT",
    "LINKED_TO",
}

# ── Driver (lazy singleton) ───────────────────────────────────────────────────
_driver = None


def get_driver():
    """Return a cached Neo4j driver, creating it on first call."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver


def close_driver():
    """Close the driver connection cleanly."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ── Connection health check ───────────────────────────────────────────────────
def test_connection():
    """
    Attempt a lightweight ping query to verify Neo4j is reachable.
    Prints a success or failure message and returns True/False.
    """
    try:
        with get_driver().session() as session:
            result = session.run("RETURN 1 AS ping")
            value = result.single()["ping"]
            if value == 1:
                print("[neo4j_loader] ✅ Neo4j connection successful.")
                return True
    except Exception as e:
        print(f"[neo4j_loader] ❌ Neo4j connection failed: {e}")
        return False


# ── Schema initialisation (run once on startup) ───────────────────────────────
def init_schema():
    """
    Create uniqueness constraints so MERGE operations are safe and fast.
    Safe to call multiple times — IF NOT EXISTS prevents duplicate errors.
    """
    with get_driver().session() as session:
        session.run(
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
    print("[neo4j_loader] ✅ Schema initialised.")


# ── Core graph writer ─────────────────────────────────────────────────────────
def _write_triple(tx, src: str, rel: str, tgt: str):
    """
    Internal transaction function.

    Relationship type injection is prevented by two layers:
      1. `rel` is validated against ALLOWED_REL_TYPES before this function
         is ever called (see create_graph below).
      2. Node names (src, tgt) are fully parameterised — they are passed as
         $src / $tgt and never interpolated into the query string.

    APOC is NOT required; instead we use a pre-validated rel string embedded
    only after allowlist confirmation, which is the recommended Neo4j pattern
    when APOC is not available.
    """
    query = (
        f"MERGE (a:Entity {{name: $src}}) "
        f"MERGE (b:Entity {{name: $tgt}}) "
        f"MERGE (a)-[:{rel}]->(b)"
    )
    tx.run(query, src=src, tgt=tgt)


def create_graph(relations: list):
    """
    Write a list of (subject, relation, object) triples to Neo4j.

    Args:
        relations: list of (src, rel, tgt) tuples.
                   `rel` must be in ALLOWED_REL_TYPES or it is skipped.
    """
    if not relations:
        return

    skipped = 0
    written = 0

    with get_driver().session() as session:
        for src, rel, tgt in relations:
            # ── Validate inputs ────────────────────────────────────────────
            src = src.strip()
            tgt = tgt.strip()
            rel = rel.strip().upper().replace(" ", "_")

            if not src or not tgt:
                skipped += 1
                continue

            # ── Allowlist check (prevents injection) ───────────────────────
            if rel not in ALLOWED_REL_TYPES:
                print(
                    f"[neo4j_loader] ⚠️  Skipping unknown relationship type "
                    f"'{rel}' — not in allowlist."
                )
                skipped += 1
                continue

            try:
                session.execute_write(_write_triple, src, rel, tgt)
                written += 1
            except Exception as e:
                print(f"[neo4j_loader] ❌ Failed to write ({src})-[{rel}]->({tgt}): {e}")
                skipped += 1

    print(
        f"[neo4j_loader] Written: {written} triples | Skipped: {skipped} triples."
    )


# ── Graph statistics query ────────────────────────────────────────────────────
def get_graph_stats() -> dict:
    """
    Return a dict with node_count, relationship_count, and fraud_node_count.
    Used by the dashboard stats panel.
    """
    try:
        with get_driver().session() as session:
            nodes = session.run(
                "MATCH (n:Entity) RETURN count(n) AS cnt"
            ).single()["cnt"]

            rels = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS cnt"
            ).single()["cnt"]

            fraud = session.run(
                "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS 'fraud' "
                "RETURN count(n) AS cnt"
            ).single()["cnt"]

        return {"node_count": nodes, "relationship_count": rels, "fraud_node_count": fraud}
    except Exception as e:
        print(f"[neo4j_loader] ❌ Stats query failed: {e}")
        return {"node_count": 0, "relationship_count": 0, "fraud_node_count": 0}


# ── Fetch all edges for visualisation ────────────────────────────────────────
def fetch_graph(search_term: str = "") -> list:
    """
    Return all (source, relationship, target) triples from Neo4j.
    If search_term is provided, filter to nodes whose name contains that string
    (case-insensitive) plus their immediate neighbours.
    """
    try:
        with get_driver().session() as session:
            if search_term.strip():
                result = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    WHERE toLower(a.name) CONTAINS toLower($term)
                       OR toLower(b.name) CONTAINS toLower($term)
                    RETURN a.name AS source, type(r) AS rel, b.name AS target
                    """,
                    term=search_term.strip(),
                )
            else:
                result = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    RETURN a.name AS source, type(r) AS rel, b.name AS target
                    """
                )
            return result.data()
    except Exception as e:
        print(f"[neo4j_loader] ❌ fetch_graph failed: {e}")
        return []


# ── Entry-point test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== neo4j_loader.py self-test ===")
    if test_connection():
        init_schema()

        # Test with valid triples
        sample = [
            ("Infosys", "PARTNERED_WITH", "Microsoft"),
            ("Microsoft", "LOCATED_IN", "Bengaluru"),
            ("Transaction_001", "FLAGGED_AS", "Fraud"),
        ]
        create_graph(sample)

        # Test injection attempt — should be skipped, not executed
        malicious = [
            ("NodeA", "INJECTED_TYPE; DROP DATABASE", "NodeB"),
        ]
        create_graph(malicious)

        stats = get_graph_stats()
        print(f"Graph stats: {stats}")
    else:
        print("Please start Neo4j before running this test.")