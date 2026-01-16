"""
Batch update script to add OpenAI fallback to all agentic workflow files
"""
import os
import re

# Files to update
files_to_update = [
    "agentic/solution_workflow.py",
    "agentic/router_agent.py",
    "agentic/comparison_workflow.py",
    "agentic/grounded_chat_workflow.py",
    "agentic/chat_agents.py",
    "agentic/potential_product_index.py",
    "agentic/agents.py",
    "agentic/rag_components.py",
]

# Also update tools files
tools_files = [
    "tools/analysis_tools.py",
    "tools/instrument_tools.py",
    "tools/ranking_tools.py",
    "tools/intent_tools.py",
    "tools/schema_tools.py",
]

def add_import_if_missing(content):
    """Add llm_fallback import if not present"""
    if "from llm_fallback import" in content:
        return content

    # Find the last import statement
    lines = content.split('\n')
    last_import_idx = -1

    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            last_import_idx = i

    if last_import_idx >= 0:
        # Insert after the last import
        lines.insert(last_import_idx + 1, "from llm_fallback import create_llm_with_fallback")
        return '\n'.join(lines)

    return content

def replace_chatgoogleai(content):
    """Replace ChatGoogleGenerativeAI instantiations with create_llm_with_fallback"""
    # Pattern to match ChatGoogleGenerativeAI(...) calls
    pattern = r'ChatGoogleGenerativeAI\s*\('
    replacement = 'create_llm_with_fallback('

    return re.sub(pattern, replacement, content)

def update_file(filepath):
    """Update a single file"""
    if not os.path.exists(filepath):
        print(f"[SKIP] File not found: {filepath}")
        return False

    print(f"[UPDATE] Processing: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Add import
    content = add_import_if_missing(content)

    # Replace instantiations
    content = replace_chatgoogleai(content)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[SUCCESS] Updated: {filepath}")
        return True
    else:
        print(f"[NO_CHANGE] No changes needed: {filepath}")
        return False

def main():
    """Main update function"""
    print("="*60)
    print("OpenAI Fallback Batch Update Script")
    print("="*60)

    all_files = files_to_update + tools_files
    updated_count = 0

    for filepath in all_files:
        if update_file(filepath):
            updated_count += 1

    print("\n" + "="*60)
    print(f"Update Complete: {updated_count}/{len(all_files)} files updated")
    print("="*60)

if __name__ == "__main__":
    main()
