from pathlib import Path
p = Path(".env")
raw = p.read_text(encoding="utf-8-sig")  # handles BOM
env = dict(line.split("=",1) for line in raw.splitlines() if line and "=" in line)
k = env.get("OPENAI_API_KEY","")
print("len:", len(k))
print("has PASTE/HERE:", "PASTE" in k, "HERE" in k)
print("head/tail:", k[:8], "...", k[-6:])
