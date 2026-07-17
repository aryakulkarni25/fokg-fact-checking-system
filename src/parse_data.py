from rdflib import Graph, RDF, URIRef
import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "trainData.txt"
TEST_FILE = BASE_DIR / "testData.txt"

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

HAS_TRUTH = URIRef("http://swc2017.aksw.org/hasTruthValue")


# ==========================================================
# Parse RDF File
# ==========================================================

def parse_rdf(file_path, is_train=True):

    graph = Graph()

    graph.parse(file_path, format="nt")

    rows = []

    for stmt in graph.subjects(RDF.type, RDF.Statement):

        head = graph.value(stmt, RDF.subject)
        relation = graph.value(stmt, RDF.predicate)
        tail = graph.value(stmt, RDF.object)

        truth = None

        if is_train:
            value = graph.value(stmt, HAS_TRUTH)

            if value is not None:
                truth = int(float(value))

        rows.append({

            "fact_uri": str(stmt),

            "head": str(head),

            "relation": str(relation),

            "tail": str(tail),

            "truth": truth

        })

    df = pd.DataFrame(rows)

    return df


# ==========================================================
# Main
# ==========================================================

print("=" * 60)
print("Parsing training data...")
print("=" * 60)

train_df = parse_rdf(TRAIN_FILE, True)

print(train_df.head())
print()

print("Training Facts :", len(train_df))
print("True Facts     :", (train_df["truth"] == 1).sum())
print("False Facts    :", (train_df["truth"] == 0).sum())

train_df.to_csv(DATA_DIR / "train.csv", index=False)

print("\nSaved -> data/train.csv")

print("\n")

print("=" * 60)
print("Parsing testing data...")
print("=" * 60)

test_df = parse_rdf(TEST_FILE, False)

print(test_df.head())
print()

print("Testing Facts :", len(test_df))

test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("\nSaved -> data/test.csv")

print("\nDone.")