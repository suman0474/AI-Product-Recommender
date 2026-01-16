import os
from pinecone import Pinecone

# Load environment variables
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

pc = Pinecone(api_key=api_key)
index = pc.Index("agentic-quickstart-test")

print("=" * 80)
print("PINECONE IMPLEMENTATION VERIFICATION")
print("=" * 80)

# 1. Verify Index Stats
print("\n1. INDEX STATISTICS")
print("-" * 80)
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
print(f"Namespaces: {list(stats.namespaces.keys())}")
for ns, ns_stats in stats.namespaces.items():
    print(f"  - {ns}: {ns_stats.vector_count} vectors")

# 2. Verify Data Integrity - Fetch Sample Records
print("\n2. DATA INTEGRITY CHECK")
print("-" * 80)
expected_ids = ["rec1", "rec2", "rec5", "rec7", "rec15", "rec17", "rec21", "rec26", "rec33", "rec38", "rec48", "rec50"]
fetch_result = index.fetch(namespace="example-namespace", ids=expected_ids[:3])

print(f"Expected records: {len(expected_ids)}")
# Check the actual structure of fetch_result
if hasattr(fetch_result, 'vectors'):
    fetched_records = fetch_result.vectors
elif hasattr(fetch_result, 'records'):
    fetched_records = fetch_result.records
else:
    fetched_records = {}
    print(f"WARNING: Unexpected fetch_result structure: {type(fetch_result)}")

print(f"Fetched sample (first 3): {len(fetched_records)} records")
for record_id, record in fetched_records.items():
    content = record.get('metadata', {}).get('content') if isinstance(record, dict) else getattr(record, 'metadata', {}).get('content')
    if not content:
        # Try fields attribute
        content = record.fields.get('content', 'N/A') if hasattr(record, 'fields') else 'N/A'
    category = record.get('metadata', {}).get('category') if isinstance(record, dict) else getattr(record, 'metadata', {}).get('category')
    if not category:
        category = record.fields.get('category', 'N/A') if hasattr(record, 'fields') else 'N/A'
    print(f"  - {record_id}: {str(content)[:50]}...")
    print(f"    Category: {category}")

# 3. List All Record IDs
print("\n3. LISTING ALL RECORDS")
print("-" * 80)
try:
    list_result = index.list(namespace="example-namespace", limit=100)
    # Handle different possible response structures
    if hasattr(list_result, 'vectors'):
        all_ids = [r.id for r in list_result.vectors]
    elif hasattr(list_result, 'records'):
        all_ids = [r.id if hasattr(r, 'id') else r.get('id') for r in list_result.records]
    else:
        all_ids = []
        print(f"WARNING: Unexpected list_result structure: {type(list_result)}")

    print(f"Total records found via list: {len(all_ids)}")
    print(f"Record IDs: {sorted(all_ids)}")

    # Check for missing records
    missing = set(expected_ids) - set(all_ids)
    if missing:
        print(f"WARNING: Missing records: {missing}")
    else:
        print("SUCCESS: All expected records present")
except Exception as e:
    print(f"Error listing records: {e}")

# 4. Test Search Functionality
print("\n4. SEARCH FUNCTIONALITY TEST")
print("-" * 80)
try:
    # Simple search without filters
    search_result = index.search(
        namespace="example-namespace",
        query={
            "top_k": 3,
            "inputs": {"text": "historical monuments"}
        }
    )
    print(f"Search returned {len(search_result['result']['hits'])} results")
    for i, hit in enumerate(search_result['result']['hits'], 1):
        print(f"  {i}. {hit['_id']}: {hit['fields']['content'][:50]}... (score: {hit['_score']:.3f})")
except Exception as e:
    print(f"ERROR: Search failed - {e}")

# 5. Test Reranking
print("\n5. RERANKING TEST")
print("-" * 80)
try:
    rerank_result = index.search(
        namespace="example-namespace",
        query={
            "top_k": 5,
            "inputs": {"text": "historical monuments"}
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 3,
            "rank_fields": ["content"]
        }
    )
    print(f"Reranked search returned {len(rerank_result['result']['hits'])} results")
    for i, hit in enumerate(rerank_result['result']['hits'], 1):
        print(f"  {i}. {hit['_id']}: {hit['fields']['content'][:50]}... (score: {hit['_score']:.3f})")
except Exception as e:
    print(f"ERROR: Reranking failed - {e}")

# 6. Test Metadata Filtering
print("\n6. METADATA FILTERING TEST")
print("-" * 80)
try:
    filter_result = index.search(
        namespace="example-namespace",
        query={
            "top_k": 5,
            "inputs": {"text": "famous works"},
            "filter": {"category": {"$eq": "history"}}
        }
    )
    print(f"Filtered search (category=history) returned {len(filter_result['result']['hits'])} results")
    for i, hit in enumerate(filter_result['result']['hits'], 1):
        print(f"  {i}. {hit['_id']}: Category={hit['fields']['category']}, Content: {hit['fields']['content'][:40]}...")

    # Verify all results are history category
    categories = [hit['fields']['category'] for hit in filter_result['result']['hits']]
    if all(cat == 'history' for cat in categories):
        print("SUCCESS: All filtered results have category='history'")
    else:
        print(f"WARNING: Some results don't match filter: {set(categories)}")
except Exception as e:
    print(f"ERROR: Filtering failed - {e}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
