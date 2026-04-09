#!/bin/bash
set -e

IMAGE="ghcr.io/soumya-batra/officeqa-agent:latest"

# --- Stage only the required files into the build context ---

# Corpus (transformed .txt files)
rm -rf .docker-corpus
cp -r treasury_bulletins_transformed .docker-corpus

# FAISS index — only the files the server actually needs:
#   gemini/index.faiss, gemini/chunks.json  (semantic search)
#   bm25.pkl, year_index.json               (keyword + year filtering)
rm -rf .docker-faiss
mkdir -p .docker-faiss/gemini
cp faiss_index/gemini/index.faiss  .docker-faiss/gemini/
cp faiss_index/gemini/chunks.json  .docker-faiss/gemini/
cp faiss_index/bm25.pkl            .docker-faiss/
cp faiss_index/year_index.json     .docker-faiss/

echo "Staged corpus (~367 MB) + FAISS index (~2.4 GB) into build context"

# --- Build ---
docker build --platform linux/amd64 -f Dockerfile.officeqa-agent -t "$IMAGE" .

# --- Clean up staging dirs ---
rm -rf .docker-corpus .docker-faiss

echo ""
echo "Built: $IMAGE"
echo "Next:  docker push $IMAGE"
