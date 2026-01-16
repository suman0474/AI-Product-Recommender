import os
import time
from pinecone import Pinecone

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable not set")

pc = Pinecone(api_key=api_key)

# Sample dataset with factual statements from different domains
records = [
    {"_id": "rec1", "content": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history"},
    {"_id": "rec2", "content": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science"},
    {"_id": "rec5", "content": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature"},
    {"_id": "rec7", "content": "The Great Wall of China was built to protect against invasions.", "category": "history"},
    {"_id": "rec15", "content": "Leonardo da Vinci painted the Mona Lisa.", "category": "art"},
    {"_id": "rec17", "content": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history"},
    {"_id": "rec21", "content": "The Statue of Liberty was a gift from France to the United States.", "category": "history"},
    {"_id": "rec26", "content": "Rome was once the center of a vast empire.", "category": "history"},
    {"_id": "rec33", "content": "The violin is a string instrument commonly used in orchestras.", "category": "music"},
    {"_id": "rec38", "content": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history"},
    {"_id": "rec48", "content": "Vincent van Gogh painted Starry Night.", "category": "art"},
    {"_id": "rec50", "content": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy"}
]

# Target the index
print("Connecting to index...")
dense_index = pc.Index("agentic-quickstart-test")

# Wait for index to be ready
print("Waiting for index to be ready...")
for i in range(30):  # Wait up to 30 seconds
    try:
        stats = dense_index.describe_index_stats()
        print(f"Index is ready! Current vector count: {stats.total_vector_count}")
        break
    except Exception as e:
        if i < 29:
            print(f"Index not ready yet, waiting... ({i+1}/30)")
            time.sleep(1)
        else:
            print(f"Warning: Index might not be fully ready: {e}")

# Upsert the records into a namespace
print("\nUpserting records to namespace 'example-namespace'...")
dense_index.upsert_records("example-namespace", records)
print(f"Successfully upserted {len(records)} records")

# Wait for vectors to be indexed
print("\nWaiting for vectors to be indexed...")
time.sleep(10)

# View stats for the index
stats = dense_index.describe_index_stats()
print(f"\n--- Index Stats ---")
print(f"Total vectors: {stats.total_vector_count}")
print(f"Namespaces: {list(stats.namespaces.keys())}")
if stats.namespaces:
    for ns, ns_stats in stats.namespaces.items():
        print(f"  {ns}: {ns_stats.vector_count} vectors")

# Define the query
query = "Famous historical structures and monuments"
print(f"\n--- Search Query ---")
print(f"Query: '{query}'")

# Search without reranking
print("\n--- Search WITHOUT Reranking ---")
results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    }
)

# Print the results
for i, hit in enumerate(results['result']['hits'], 1):
    print(f"{i}. ID: {hit['_id']:<5} | Score: {round(hit['_score'], 2):<5} | Category: {hit['fields']['category']:<10} | Text: {hit['fields']['content'][:70]}")

# Search with reranking
print("\n--- Search WITH Reranking ---")
reranked_results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,
        "inputs": {
            'text': query
        }
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 10,
        "rank_fields": ["content"]
    }
)

# Print the reranked results
for i, hit in enumerate(reranked_results['result']['hits'], 1):
    print(f"{i}. ID: {hit['_id']:<5} | Score: {round(hit['_score'], 2):<5} | Category: {hit['fields']['category']:<10} | Text: {hit['fields']['content'][:70]}")

print("\n✅ Quickstart completed successfully!")
print("\nKey observations:")
print("- Without reranking: Some results might include less relevant items (like Shakespeare)")
print("- With reranking: Results are much more focused on historical structures and monuments")
