"""Generate PDF investment briefing from markdown."""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path

# Read markdown
md_path = Path(__file__).parent / "INVESTMENT_BRIEFING.md"
md_content = md_path.read_text(encoding='utf-8')

# Convert markdown to HTML
html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Create full HTML document with styling
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            margin: 1in;
            size: letter;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 100%;
        }}
        h1 {{
            color: #0066cc;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
            page-break-after: avoid;
        }}
        h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #555;
            page-break-after: avoid;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #0066cc;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f5f9ff;
            font-style: italic;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 9pt;
        }}
        strong {{
            color: #0066cc;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 30px 0;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Generate PDF
output_path = Path(__file__).parent / "INVESTMENT_BRIEFING.pdf"
base_url = str(Path(__file__).parent)

print("Generating PDF...")
HTML(string=full_html, base_url=base_url).write_pdf(output_path)
print(f"PDF saved to: {output_path}")
