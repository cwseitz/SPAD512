import re
import sys

def parse_tex(tex_file):
    keys = set()
    cites = [
        r'\\cite\{([^}]*)\}',
        r'\\citep\{([^}]*)\}',
        r'\\citet\{([^}]*)\}',
        r'\\citealt\{([^}]*)\}',
        r'\\citeauthor\{([^}]*)\}',
        r'\\Cite\{([^}]*)\}',
        r'\\footcite\{([^}]*)\}',
        r'\\textcite\{([^}]*)\}',
    ]

    with open(tex_file, 'r', encoding='utf-8') as file:
        content = file.read()
        for pattern in cites:
            matches = re.findall(pattern, content)
            for match in matches:
                keys = match.split(',')
                keys.update(key.strip() for key in keys)

    return keys

def clean_bib(bib_file, citation_keys):
    """
    Remove unused entries from the .bib file.
    """
    with open(bib_file, 'r', encoding='utf-8') as file:
        bib_content = file.read()

    # Regular expression to match each bib entry by its key
    bib_entries = re.findall(r'@.*?\{([^,]*),.*?\n\}', bib_content, re.DOTALL)

    # Create a dictionary of all bib entries
    entry_dict = {}
    for entry in bib_entries:
        entry_key = entry.split(",")[0].strip()
        entry_dict[entry_key] = entry

    # Keep only the entries that are cited in the .tex file
    cited_entries = {key: entry_dict[key] for key in citation_keys if key in entry_dict}

    # Write the cleaned .bib file
    with open(bib_file, 'w', encoding='utf-8') as file:
        for entry in cited_entries.values():
            file.write(f"@{entry}\n\n")

    print(f"Cleaned .bib file saved: {bib_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bibcleaner.py <main.tex> <references.bib>")
        sys.exit(1)

    tex_file = sys.argv[1]
    bib_file = sys.argv[2]

    citation_keys = parse_tex(tex_file)
    clean_bib(bib_file, citation_keys)
