"""Render preprint.md to preprint.pdf (needs `markdown` and Google Chrome)."""
import subprocess
from pathlib import Path

import re

import markdown

here = Path(__file__).parent
text = (here / "preprint.md").read_text()
# Python-Markdown needs a blank line before a list that follows a paragraph line (GitHub doesn't)
text = re.sub(r"(?m)^(?!\s*(?:[-*]|\d+\.)\s)(\S.*)\n(?=(?:[-*]|\d+\.)\s)", r"\1\n\n", text)
body = markdown.markdown(text, extensions=["tables"])
css = """
@page { size: A4; margin: 22mm 20mm; }
body { font-family: 'Times New Roman', Georgia, serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 8pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; width: 100%; }
th, td { border-top: 1px solid #999; border-bottom: 1px solid #999; padding: 3pt 5pt; text-align: left; }
th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; }
hr { border: 0; border-top: 1px solid #bbb; }
p, li { orphans: 3; widows: 3; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
(here / "preprint.html").write_text(html)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                f"--print-to-pdf={here / 'preprint.pdf'}", (here / "preprint.html").as_uri()], check=True,
               capture_output=True)
(here / "preprint.html").unlink()
print("wrote", here / "preprint.pdf")
