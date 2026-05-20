import subprocess
import sys

result = subprocess.run(
    [sys.executable, "pipeline.py"],
    capture_output=True,
    text=True
)

print(result.stdout)

if "success" in result.stdout:
    print("Accuracy check passed!")
else:
    print("Output:", result.stdout)
    raise Exception("Pipeline did not complete successfully")
