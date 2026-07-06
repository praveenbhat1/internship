"""Convert a Markdown file to a clean PDF (pure-python, no system deps).

Usage: python md_to_pdf.py [in.md] [out.pdf] [diagram.png]
If a diagram image is given, it is appended on its own landscape page.
"""
import os
import sys
import markdown
from xhtml2pdf import pisa

src = sys.argv[1] if len(sys.argv) > 1 else "project_brief.md"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".pdf"
diagram = sys.argv[3] if len(sys.argv) > 3 else None

with open(src, encoding="utf-8") as f:
    body = markdown.markdown(f.read(), extensions=["tables", "fenced_code", "sane_lists"])

fig = ""
if diagram and os.path.exists(diagram):
    fig = (f'<div class="figpage"><h2>Methodology Diagram</h2>'
           f'<img src="{os.path.abspath(diagram)}" style="width:16cm;" />'
           f'<div class="figcap">End-to-end pipeline: Input &rarr; Preprocessing &rarr; '
           f'frozen SigLIP-2 (image + cached text) + Orientation Head &rarr; CMAA &rarr; '
           f'OCFR &rarr; Hybrid DACG &rarr; Classifier &rarr; 26 attributes '
           f'(dashed = training-only loss).</div></div>')

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@page {{ size: A4 portrait; margin: 2cm; }}
@page landscapefig {{ size: A4 landscape; margin: 1.2cm; }}
.figpage {{ page: landscapefig; page-break-before: always; text-align: center; }}
.figcap {{ font-size: 9pt; color: #555; margin-top: 10px; text-align: left; }}
body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #222; line-height: 1.45; }}
h1 {{ font-size: 19pt; color: #1a3d6d; border-bottom: 2px solid #1a3d6d; padding-bottom: 4px; }}
h2 {{ font-size: 13pt; color: #1a73e8; margin-top: 16px; }}
h3 {{ font-size: 11.5pt; color: #333; }}
em {{ color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
th, td {{ border: 1px solid #bbb; padding: 5px 7px; text-align: left; font-size: 9.5pt; }}
th {{ background: #eef3fb; }}
blockquote {{ border-left: 4px solid #1a73e8; margin: 8px 0; padding: 4px 12px; background: #f6f9ff; color: #333; }}
code, pre {{ font-family: Menlo, Consolas, monospace; font-size: 9pt; background: #f4f4f4; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 14px 0; }}
ul {{ margin: 4px 0 8px 0; }}
</style></head><body>{body}{fig}</body></html>"""

with open(out, "w+b") as f:
    status = pisa.CreatePDF(html, dest=f)
print("ERROR" if status.err else f"OK -> {out}")
