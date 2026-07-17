import os
import pandas as pd
import networkx as nx

# =====================================================
# Paths
# =====================================================

BASE = os.path.dirname(os.path.dirname(__file__))

TRAIN_FILE = os.path.join(BASE, "data", "train.csv")
TEST_FILE = os.path.join(BASE, "data", "test.csv")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("=" * 60)
print("Building Graph...")
print("=" * 60)

# =====================================================
# Build Graph
# =====================================================

G = nx.MultiDiGraph()

all_data = pd.concat([train, test], ignore_index=True)

for _, row in all_data.iterrows():
    G.add_edge(
        row["head"],
        row["tail"],
        relation=row["relation"]
    )

print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())

# =====================================================
# Graph Statistics
# =====================================================

degree = dict(G.degree())
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

relation_freq = train.groupby("relation").size().to_dict()
head_freq = train.groupby("head").size().to_dict()
tail_freq = train.groupby("tail").size().to_dict()

pair_freq = train.groupby(
    ["head", "tail"]
).size().to_dict()

triple_freq = train.groupby(
    ["head", "relation", "tail"]
).size().to_dict()

# =====================================================
# Helper Functions
# =====================================================

def shortest_path(head, tail):
    try:
        return nx.shortest_path_length(G, head, tail)
    except:
        return 10


def common_neighbors(head, tail):
    try:
        h = set(G.successors(head))
        t = set(G.predecessors(tail))
        return len(h & t)
    except:
        return 0


def jaccard_score(head, tail):
    try:
        h = set(G.successors(head))
        t = set(G.predecessors(tail))

        union = h | t

        if len(union) == 0:
            return 0

        return len(h & t) / len(union)

    except:
        return 0


# =====================================================
# Feature Extraction
# =====================================================

def add_features(df):

    df = df.copy()

    print("Adding graph degree features...")

    df["head_degree"] = df["head"].map(degree).fillna(0)
    df["tail_degree"] = df["tail"].map(degree).fillna(0)

    df["head_in_degree"] = df["head"].map(in_degree).fillna(0)
    df["head_out_degree"] = df["head"].map(out_degree).fillna(0)

    df["tail_in_degree"] = df["tail"].map(in_degree).fillna(0)
    df["tail_out_degree"] = df["tail"].map(out_degree).fillna(0)

    print("Adding frequency features...")

    df["relation_freq"] = df["relation"].map(relation_freq).fillna(0)
    df["head_freq"] = df["head"].map(head_freq).fillna(0)
    df["tail_freq"] = df["tail"].map(tail_freq).fillna(0)

    df["pair_freq"] = [
        pair_freq.get((h, t), 0)
        for h, t in zip(df["head"], df["tail"])
    ]

    df["triple_freq"] = [
        triple_freq.get((h, r, t), 0)
        for h, r, t in zip(
            df["head"],
            df["relation"],
            df["tail"]
        )
    ]

    print("Computing shortest paths...")

    df["shortest_path"] = [
        shortest_path(h, t)
        for h, t in zip(df["head"], df["tail"])
    ]

    print("Computing common neighbors...")

    df["common_neighbors"] = [
        common_neighbors(h, t)
        for h, t in zip(df["head"], df["tail"])
    ]

    print("Computing Jaccard scores...")

    df["jaccard"] = [
        jaccard_score(h, t)
        for h, t in zip(df["head"], df["tail"])
    ]

    return df


# =====================================================
# Train Features
# =====================================================

print()
print("=" * 60)
print("Generating TRAIN features")
print("=" * 60)

train_features = add_features(train)

# =====================================================
# Test Features
# =====================================================

print()
print("=" * 60)
print("Generating TEST features")
print("=" * 60)

test_features = add_features(test)

# =====================================================
# Save
# =====================================================

train_out = os.path.join(BASE, "data", "train_features.csv")
test_out = os.path.join(BASE, "data", "test_features.csv")

train_features.to_csv(train_out, index=False)
test_features.to_csv(test_out, index=False)

print()
print("=" * 60)
print("Feature Engineering Complete")
print("=" * 60)
print("Train Shape :", train_features.shape)
print("Test Shape  :", test_features.shape)
print()
print("Saved:")
print(train_out)
print(test_out)