from collections import Counter
import re
from pathlib import Path

text = Path("inputs/sentencas.txt").read_text(encoding="utf-8").lower()
tokens = re.findall(r"[a-zà-ú]+", text)
freq = Counter(tokens).most_common(10)

print("Top 10 palavras mais frequentes:")
for w, c in freq:
    print(f"- {w}: {c}")

print("\nInsight rápido:")
print("- Texto menciona temperatura/quente/frio, sugerindo variável explicativa para demanda.")
