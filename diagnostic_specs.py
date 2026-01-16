#!/usr/bin/env python
"""Quick diagnostic to show specification breakdown from 3 sources."""

import os
import sys
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

from agentic.deep_agent.parallel_specs_enrichment import run_parallel_3_source_enrichment

test_items = [
    {'number': 1, 'type': 'instrument', 'name': 'Pressure Transmitter', 'category': 'Pressure', 'sample_input': 'Pressure Transmitter 0-100 bar'},
]

user_input = 'I need a pressure transmitter with 4-20mA output and ATEX Zone 1 and IP66 rating'

print("Running parallel 3-source enrichment...")
result = run_parallel_3_source_enrichment(items=test_items, user_input=user_input, session_id='diagnostic')

if result.get('success'):
    print("\n" + "="*60)
    print("SPECIFICATION BREAKDOWN BY SOURCE")
    print("="*60)
    
    for item in result.get('items', []):
        user = item.get('user_specified_specs', {})
        llm = item.get('llm_generated_specs', {})
        std = item.get('standards_specifications', {})
        
        print(f"\nProduct: {item.get('name')}")
        print("-"*40)
        
        print(f"\n1. USER-SPECIFIED (MANDATORY): {len(user)} specs")
        for k,v in user.items():
            print(f"   - {k}: {v}")
        
        print(f"\n2. LLM-GENERATED: {len(llm)} specs")
        for k,v in list(llm.items())[:8]:
            val = v.get('value', v) if isinstance(v, dict) else v
            print(f"   - {k}: {val}")
        if len(llm) > 8:
            print(f"   ... and {len(llm)-8} more")
        
        print(f"\n3. STANDARDS-BASED: {len(std)} specs")
        for k,v in list(std.items())[:8]:
            val = v.get('value', v) if isinstance(v, dict) else v
            print(f"   - {k}: {val}")
        if len(std) > 8:
            print(f"   ... and {len(std)-8} more")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"  User-specified specs:  {len(user)}")
        print(f"  LLM-generated specs:   {len(llm)}")
        print(f"  Standards specs:       {len(std)}")
        combined = item.get('combined_specifications', {})
        print(f"  Total combined:        {len(combined)}")
        print()
        
        # Source breakdown in combined
        user_count = sum(1 for v in combined.values() if isinstance(v, dict) and v.get('source') == 'user_specified')
        llm_count = sum(1 for v in combined.values() if isinstance(v, dict) and v.get('source') == 'llm_generated')
        std_count = sum(1 for v in combined.values() if isinstance(v, dict) and v.get('source') == 'standards')
        print("After deduplication (in combined_specifications):")
        print(f"  From user input:   {user_count} (MANDATORY)")
        print(f"  From LLM:          {llm_count}")
        print(f"  From standards:    {std_count}")
else:
    print(f"Failed: {result.get('error')}")
