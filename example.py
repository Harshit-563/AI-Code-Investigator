import subprocess

result = subprocess.run(
    ["git", "--version"],
    capture_output=True,
    text=True
)

print(result.stdout)