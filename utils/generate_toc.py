# generate_toc.py
import re

# Path to your README.md
readme_path = "README.md"

# Read the README file
with open(readme_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

toc_lines = []
for line in lines:
    # Match headings starting with #
    match = re.match(r'^(#{2,6})\s+(.*)', line)
    if match:
        level = len(match.group(1)) - 1  # level 2 (#) = 1 indent, etc.
        title = match.group(2).strip()
        # GitHub anchor format: lowercase, spaces -> -, remove special chars
        anchor = re.sub(r'[^\w\s-]', '', title).lower().replace(' ', '-')
        toc_lines.append(f"{'  ' * (level-1)}- [{title}](#{anchor})")

# Print or save the ToC
toc_text = "\n".join(toc_lines)
print(toc_text)

# Optionally, write to file
with open("README_TOC.md", "w", encoding="utf-8") as f:
    f.write(toc_text)
