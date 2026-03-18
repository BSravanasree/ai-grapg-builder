import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return entities


def extract_relations(text):
    relations = []

    if "partnered with" in text:
        parts = text.split("partnered with")
        relations.append((parts[0].strip(), "PARTNERED_WITH", parts[1].strip()))

    if "in" in text:
        parts = text.split("in")
        relations.append((parts[0].strip(), "LOCATED_IN", parts[1].strip()))

    return relations