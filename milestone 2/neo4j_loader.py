from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j123"

driver = GraphDatabase.driver(uri, auth=(username, password))


def create_graph(relations):
    with driver.session() as session:
        for src, rel, tgt in relations:
            query = """
            MERGE (a:Entity {name: $src})
            MERGE (b:Entity {name: $tgt})
            MERGE (a)-[:%s]->(b)
            """ % rel

            session.run(query, src=src, tgt=tgt)