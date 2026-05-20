import json
import sys

with open("metrics.json") as f:
    m = json.load(f)

print(f"Accuracy: {m['accuracy']}")

if m["accuracy"] >= 0.80:
    print("Accuracy check passed!")
else:
    sys.exit(1)