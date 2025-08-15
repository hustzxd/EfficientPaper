#!/usr/bin/env python3
"""
Fix markdown-toc generated links to be compatible with MkDocs
"""
import re
import sys

def fix_markdown_links(content):
    """
    Convert markdown-toc style links to MkDocs compatible format
    Changes #Some-Title-With-Mixed-Case to #some-title-with-mixed-case
    """
    def link_replacer(match):
        link_text = match.group(1)
        anchor = match.group(2)
        # Convert anchor to lowercase
        fixed_anchor = anchor.lower()
        return f"[{link_text}](#{fixed_anchor})"
    
    # Pattern to match [text](#Anchor-With-Mixed-Case)
    pattern = r'\[([^\]]+)\]\(#([A-Za-z0-9\-]+)\)'
    
    return re.sub(pattern, link_replacer, content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_mkdocs_links.py <markdown_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_markdown_links(content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed markdown links in {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()