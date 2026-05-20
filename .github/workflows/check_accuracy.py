import json

with open("metrics.json") as f:
    m = json.load(f)

print(f"Accuracy: {m['accuracy']}")

if m["accuracy"] >= 0.80:
    print("Accuracy check passed!")
else:
    raise Exception(f"Accuracy too low: {m['accuracy']}")
