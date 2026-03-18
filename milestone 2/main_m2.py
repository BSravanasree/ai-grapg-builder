from entity_extractor import extract_entities, extract_relations
from neo4j_loader import create_graph

text = "Infosys partnered with Microsoft in Bengaluru."

entities = extract_entities(text)
relations = extract_relations(text)

print("Entities:", entities)
print("Relations:", relations)

create_graph(relations)