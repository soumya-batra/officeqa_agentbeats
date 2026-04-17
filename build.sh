#!/bin/bash
set -e

IMAGE="ghcr.io/soumya-batra/officeqa-agent:latest"

# --- Stage only the required files into the build context ---

# Corpus (transformed .txt files)
rm -rf .docker-corpus
cp -r treasury_bulletins_transformed .docker-corpus

# FAISS index (txt_meta4 variant — best performing):
#   nebius/index.faiss, nebius/chunks.json  (semantic search)
#   bm25s/                                  (stemmed keyword search)
rm -rf .docker-faiss
mkdir -p .docker-faiss/nebius
cp faiss_index_txt_meta4/nebius/index.faiss  .docker-faiss/nebius/
cp faiss_index_txt_meta4/nebius/chunks.json  .docker-faiss/nebius/
cp -r faiss_index_txt_meta4/bm25s            .docker-faiss/bm25s

echo "Staged corpus (~367 MB) + FAISS index (~2.4 GB) into build context"

# --- Build ---
docker build --platform linux/amd64 -f Dockerfile.officeqa-agent -t "$IMAGE" .

# --- Clean up staging dirs ---
rm -rf .docker-corpus .docker-faiss

echo ""
echo "Built: $IMAGE"
echo "Next:  docker push $IMAGE"
