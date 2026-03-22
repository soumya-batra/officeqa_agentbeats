"""
One-time script to upload documents from a folder to an OpenAI Vector Store.
Run: python setup_vector_store.py
Then set VECTOR_STORE_ID=<printed id> in your environment.
"""
import os
import glob
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FOLDER = os.environ.get("DOCS_FOLDER", "./treasury_bulletins_transformed")
VS_NAME = os.environ.get("VECTOR_STORE_NAME", "treasury-bulletins")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Create vector store
vs = client.vector_stores.create(name=VS_NAME)
print(f"Created vector store: {vs.id}")

# Gather files
file_paths = [p for p in glob.glob(f"{FOLDER}/**/*", recursive=True) if os.path.isfile(p)]
if not file_paths:
    print(f"No files found in {FOLDER}", file=sys.stderr)
    sys.exit(1)

print(f"Uploading {len(file_paths)} file(s) from {FOLDER} ...")
file_streams = [open(p, "rb") for p in file_paths]

try:
    batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id,
        files=file_streams,
    )
finally:
    for f in file_streams:
        f.close()

print(f"Status: {batch.status}")
print(f"Files — completed: {batch.file_counts.completed}, failed: {batch.file_counts.failed}")
print()
print(f"Add to your environment:")
print(f"  VECTOR_STORE_ID={vs.id}")
print(f"  ENABLE_FILE_SEARCH=true")
