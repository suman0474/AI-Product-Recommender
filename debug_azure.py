
import sys
import traceback

print("Python executable:", sys.executable)
print("Attempting to import azure.storage.blob...")

try:
    from azure.storage.blob import BlobServiceClient
    print("SUCCESS: azure.storage.blob imported correctly.")
except ImportError:
    print("FAILURE: Could not import azure.storage.blob")
    traceback.print_exc()
except Exception:
    print("FAILURE: Unexpected error during import")
    traceback.print_exc()
