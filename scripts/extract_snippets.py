"""Copy the marked code regions (BEGIN/END snippet:<name>) into snippets/*.py so the report quotes real code."""
import re
import textwrap
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "snippets"
out.mkdir(exist_ok=True)
for f in list((root / "src").glob("*.py")) + list((root / "scripts").glob("*.py")):
    txt = f.read_text()
    for m in re.finditer(r"# BEGIN snippet:(\w+)\n(.*?)\n\s*# END snippet:\1", txt, re.S):
        (out / f"{m.group(1)}.py").write_text(textwrap.dedent(m.group(2)) + "\n")
        print("snippet", m.group(1), "<-", f.name)
