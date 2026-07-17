import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))

PRED = os.path.join(BASE, "predictions.csv")

RESULT = os.path.join(BASE, "result.ttl")

PROP = "<http://swc2017.aksw.org/hasTruthValue>"
TYPE = "^^<http://www.w3.org/2001/XMLSchema#double>"

df = pd.read_csv(PRED)

with open(RESULT, "w", encoding="utf8") as f:

    for _, row in df.iterrows():

        score = float(row["truth_score"])

        line = (
            f"<{row['fact_uri']}> "
            f"{PROP} "
            f"\"{score:.6f}\"{TYPE} .\n"
        )

        f.write(line)

print("=" * 60)
print("Result.ttl Generated")
print("=" * 60)

print(RESULT)