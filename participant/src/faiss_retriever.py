import json
import logging
import re
import time
from pathlib import Path

import numpy as np

from chunker import enrich, iter_chunks, tok
from corpus_loader import load_corpus
from models import RetrievedContext

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_BATCH_TOKENS = 250_000
RRF_K = 60            # standard RRF constant
GLOBAL_CANDIDATE_K = 50   # how many each of FAISS global and BM25 global contribute
ERA_CANDIDATE_K = 200     # how many year-era FAISS contributes (already filtered to era)
RRF_TOP_N = 30            # keep this many after RRF, feed to cross-encoder
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


class FaissRetriever:
    def __init__(self, corpus_dir: Path, index_dir: Path, top_k: int = 5):
        self._top_k  = top_k
        self._index  = None
        self._chunks: list[dict] = []
        self._bm25   = None
        self._bm25_corpus: list[list[str]] = []
        self._cross_encoder = None
        self._cross_encoder_loaded = False
        # Offline indices built after chunks are loaded
        self._chunk_bulletin_year: list[int | None] = []   # bulletin year per chunk
        self._year_content_index: dict[int, list[int]] = {}  # year -> chunk indices containing that year

        index_path    = index_dir / "index.faiss"
        meta_path     = index_dir / "chunks.json"
        bm25_path     = index_dir / "bm25.pkl"
        year_idx_path = index_dir / "year_index.json"

        if index_path.exists() and meta_path.exists():
            self._load(index_path, meta_path)
        else:
            self._build(corpus_dir, index_dir, index_path, meta_path)

        self._build_bm25(bm25_path)
        self._build_year_index(year_idx_path)

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_bm25(self, bm25_path: Path) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed; BM25 disabled")
            return
        import pickle
        if bm25_path.exists():
            logger.info("Loading BM25 index from %s", bm25_path)
            with bm25_path.open("rb") as f:
                self._bm25 = pickle.load(f)
            logger.info("Loaded BM25 index")
            return
        logger.info("Building BM25 index over %d chunks", len(self._chunks))
        self._bm25_corpus = [self._tokenize(c["content"]) for c in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus)
        with bm25_path.open("wb") as f:
            pickle.dump(self._bm25, f)
        logger.info("Built and saved BM25 index to %s", bm25_path)

    # ── load ──────────────────────────────────────────────────────────────────

    def _load(self, index_path: Path, meta_path: Path) -> None:
        import faiss
        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_path))
        with meta_path.open("r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        logger.info("Loaded FAISS index with %d chunks", len(self._chunks))

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(
        self,
        corpus_dir: Path,
        index_dir: Path,
        index_path: Path,
        meta_path: Path,
    ) -> None:
        import faiss
        from openai import OpenAI

        logger.info("Building FAISS index from corpus at %s", corpus_dir)
        documents = load_corpus(corpus_dir)

        all_chunks: list[dict] = []
        for doc in documents:
            for chunk in iter_chunks(doc.content):
                all_chunks.append({
                    "source":        str(doc.path),
                    "content":       enrich(chunk),
                    "table_summary": chunk.get("table_summary", ""),
                    "type":          chunk.get("type", "text"),
                    "section":       chunk.get("section", ""),
                })

        logger.info("Chunked corpus into %d chunks, embedding...", len(all_chunks))

        client     = OpenAI()
        embeddings: list[list[float]] = []
        batches    = self._build_batches(all_chunks)

        logger.info("Split into %d token-safe batches", len(batches))

        for i, batch in enumerate(batches):
            self._embed_batch_with_retry(client, batch, embeddings)
            logger.info("Embedded batch %d / %d  (%d chunks so far)",
                        i + 1, len(batches),
                        sum(len(b) for b in batches[:i+1]))

        dim    = len(embeddings[0])
        matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(all_chunks, f)

        self._index  = index
        self._chunks = all_chunks
        logger.info("Built and saved FAISS index with %d chunks", len(all_chunks))

    # ── batch builder ─────────────────────────────────────────────────────────

    def _build_batches(self, all_chunks: list[dict]) -> list[list[str]]:
        """Split chunks into token-safe batches for the embedding API.
        Respects both the 300k token limit and the 2048 input limit per request.
        """
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for chunk in all_chunks:
            content       = chunk["content"]
            chunk_tokens  = tok(content)

            # flush if adding this chunk would exceed either limit
            if current_batch and (
                current_tokens + chunk_tokens > MAX_BATCH_TOKENS
                or len(current_batch) >= 2048
            ):
                batches.append(current_batch)
                current_batch  = []
                current_tokens = 0

            current_batch.append(content)
            current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    # ── embed helpers ─────────────────────────────────────────────────────────

    def _embed_batch_with_retry(
        self,
        client,
        batch: list[str],
        embeddings: list,
    ) -> None:
        for attempt in range(10):
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                embeddings.extend([e.embedding for e in resp.data])
                return
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = 2 ** attempt * 5
                    logger.warning(
                        "Rate limit hit, retrying in %ds (attempt %d)",
                        wait, attempt + 1,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Embedding failed after 10 attempts")

    def _embed_query(self, question: str) -> np.ndarray:
        import faiss
        from openai import OpenAI

        client = OpenAI()
        resp   = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[question],
        )
        vec = np.array([resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    # ── year-aware helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_years(question: str) -> list[int]:
        # Use (?<!\d) instead of \b so "FY2013" and "FY2023" are matched correctly
        return [int(y) for y in re.findall(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", question)]

    def _build_year_index(self, year_idx_path: Path) -> None:
        """Precompute bulletin year per chunk and an inverted index: year -> chunk indices."""
        year_pat = re.compile(r"treasury_bulletin_(\d{4})")

        # _chunk_bulletin_year is cheap to rebuild from source paths — always recompute
        self._chunk_bulletin_year = []
        for chunk in self._chunks:
            m = year_pat.search(chunk.get("source", ""))
            self._chunk_bulletin_year.append(int(m.group(1)) if m else None)

        if year_idx_path.exists():
            logger.info("Loading year content index from %s", year_idx_path)
            with year_idx_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            self._year_content_index = {int(k): v for k, v in raw.items()}
            logger.info("Loaded year index: %d distinct years", len(self._year_content_index))
            return

        logger.info("Building year content index over %d chunks", len(self._chunks))
        content_year_pat = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
        self._year_content_index = {}
        for i, chunk in enumerate(self._chunks):
            for year_str in set(content_year_pat.findall(chunk["content"])):
                self._year_content_index.setdefault(int(year_str), []).append(i)

        with year_idx_path.open("w", encoding="utf-8") as f:
            json.dump(self._year_content_index, f)
        logger.info(
            "Built and saved year index: %d distinct years to %s",
            len(self._year_content_index), year_idx_path,
        )

    def _year_era_indices(self, years: list[int], before: int = 1, after: int = 8) -> list[int]:
        """Indices of chunks from bulletins published within [year-before, year+after] of any detected year."""
        if not years:
            return []
        return [
            i for i, by in enumerate(self._chunk_bulletin_year)
            if by is not None and any(year - before <= by <= year + after for year in years)
        ]

    def _year_content_indices(self, years: list[int], era_set: set[int]) -> list[int]:
        """Chunk indices that both mention a target year in their text AND are in the era set."""
        result: set[int] = set()
        for year in years:
            result.update(i for i in self._year_content_index.get(year, []) if i in era_set)
        return list(result)

    # ── cross-encoder rerank ──────────────────────────────────────────────────

    def _load_cross_encoder(self) -> None:
        if self._cross_encoder_loaded:
            return
        self._cross_encoder_loaded = True
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info("Loaded cross-encoder: %s", CROSS_ENCODER_MODEL)
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s); skipping rerank", exc)

    def _rerank(self, question: str, candidates: list[RetrievedContext]) -> list[RetrievedContext]:
        """Score (question, chunk) pairs with the cross-encoder and return top_k."""
        self._load_cross_encoder()
        if self._cross_encoder is None or len(candidates) <= self._top_k:
            return candidates[:self._top_k]
        pairs = [(question, c.content[:8000]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        logger.info(
            "Cross-encoder reranked %d → %d  (top score=%.3f)",
            len(candidates), self._top_k, ranked[0][0] if ranked else 0,
        )
        return [c for _, c in ranked[:self._top_k]]

    # ── retrieve ──────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        faiss_ranks: dict[int, int],
        bm25_ranks: dict[int, int],
        faiss_w: float = 0.5,
        bm25_w: float = 0.5,
    ) -> list[tuple[int, float]]:
        """Combine FAISS and BM25 ranks via weighted RRF. Returns (idx, score) sorted descending."""
        all_idx = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores: dict[int, float] = {}
        for idx in all_idx:
            faiss_rrf = faiss_w / (RRF_K + faiss_ranks[idx] + 1) if idx in faiss_ranks else 0.0
            bm25_rrf  = bm25_w  / (RRF_K + bm25_ranks[idx] + 1) if idx in bm25_ranks else 0.0
            rrf_scores[idx] = faiss_rrf + bm25_rrf
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve_by_source_files(
        self,
        source_files: list[str],
        question: str,
        query_vec: "np.ndarray | None" = None,
    ) -> list[RetrievedContext]:
        if self._index is None or not self._chunks:
            return []
        names = {Path(f).name for f in source_files}
        candidate_indices = [i for i, c in enumerate(self._chunks) if Path(c["source"]).name in names]
        if not candidate_indices:
            return []

        candidate_k = min(len(candidate_indices), max(GLOBAL_CANDIDATE_K, ERA_CANDIDATE_K))

        if query_vec is None:
            query_vec = self._embed_query(question)

        # FAISS ranks within candidate set
        candidate_matrix = np.array(
            [self._index.reconstruct(i) for i in candidate_indices], dtype=np.float32
        )
        faiss_scores = (candidate_matrix @ query_vec.T).flatten()
        faiss_order = np.argsort(faiss_scores)[::-1][:GLOBAL_CANDIDATE_K]
        faiss_ranks = {candidate_indices[int(i)]: rank for rank, i in enumerate(faiss_order)}

        # BM25 ranks within candidate set (scoped — BM25 weighted higher)
        bm25_ranks: dict[int, int] = {}
        if self._bm25 is not None:
            tokens = self._tokenize(question)
            all_bm25_scores = self._bm25.get_scores(tokens)
            candidate_bm25 = sorted(
                ((i, all_bm25_scores[i]) for i in candidate_indices),
                key=lambda x: x[1],
                reverse=True,
            )
            bm25_ranks = {i: rank for rank, (i, _) in enumerate(candidate_bm25[:GLOBAL_CANDIDATE_K])}

        top_rrf = self._rrf_fuse(faiss_ranks, bm25_ranks, faiss_w=0.4, bm25_w=0.6)[:RRF_TOP_N]
        candidates = [self._make_result(self._chunks[idx], score) for idx, score in top_rrf]
        return self._rerank(question, candidates)

    def retrieve(
        self,
        question: str,
        query_vec: "np.ndarray | None" = None,
    ) -> list[RetrievedContext]:
        if self._index is None or not self._chunks:
            return []

        if query_vec is None:
            query_vec = self._embed_query(question)

        # FAISS global (semantic) — top GLOBAL_CANDIDATE_K
        _, faiss_indices = self._index.search(query_vec, GLOBAL_CANDIDATE_K)
        faiss_ranks: dict[int, int] = {
            int(idx): rank
            for rank, idx in enumerate(faiss_indices[0])
            if idx >= 0
        }

        # BM25 global (keyword) — top GLOBAL_CANDIDATE_K
        bm25_ranks: dict[int, int] = {}
        if self._bm25 is not None:
            tokens = self._tokenize(question)
            bm25_scores = self._bm25.get_scores(tokens)
            top_bm25 = np.argsort(bm25_scores)[::-1][:GLOBAL_CANDIDATE_K]
            bm25_ranks = {int(idx): rank for rank, idx in enumerate(top_bm25)}

        # Year-era (wide) and year-near (tight) FAISS passes.
        # Wide window [year-1, year+8]: captures later bulletins with historical tables.
        # Tight window [year, year+5]: prioritises contemporary bulletins with original data.
        detected_years = self._extract_years(question)
        year_era_ranks: dict[int, int] = {}
        year_near_ranks: dict[int, int] = {}

        if detected_years:
            era_indices  = self._year_era_indices(detected_years, before=1, after=8)
            near_indices = self._year_era_indices(detected_years, before=0, after=5)
            logger.info(
                "Year-era: detected years=%s, wide=%d near=%d",
                detected_years, len(era_indices), len(near_indices),
            )

            def _era_search(pool: list[int]) -> dict[int, int]:
                if not pool:
                    return {}
                content_pool = self._year_content_indices(detected_years, set(pool))
                search = content_pool if content_pool else pool
                mat = np.array([self._index.reconstruct(i) for i in search], dtype=np.float32)
                scores = (mat @ query_vec.T).flatten()
                order = np.argsort(scores)[::-1][:ERA_CANDIDATE_K]
                return {search[int(i)]: rank for rank, i in enumerate(order)}

            year_era_ranks  = _era_search(era_indices)
            year_near_ranks = _era_search(near_indices)
            logger.info(
                "Year-era ranks: wide=%d near=%d",
                len(year_era_ranks), len(year_near_ranks),
            )

        # Four-way RRF: global FAISS + global BM25 + wide-era + near-era → top RRF_TOP_N
        all_idx = set(faiss_ranks) | set(bm25_ranks) | set(year_era_ranks) | set(year_near_ranks)
        rrf_scores: dict[int, float] = {}
        has_era  = bool(year_era_ranks)
        has_near = bool(year_near_ranks)
        fallback_rank = max(GLOBAL_CANDIDATE_K, ERA_CANDIDATE_K)
        for idx in all_idx:
            score = 0.0
            if has_era and has_near:
                score = (
                    0.25 / (RRF_K + faiss_ranks.get(idx, fallback_rank) + 1)
                    + 0.10 / (RRF_K + bm25_ranks.get(idx, fallback_rank) + 1)
                    + 0.25 / (RRF_K + year_era_ranks.get(idx, fallback_rank) + 1)
                    + 0.40 / (RRF_K + year_near_ranks.get(idx, fallback_rank) + 1)
                )
            elif has_era:
                score = (
                    0.40 / (RRF_K + faiss_ranks.get(idx, fallback_rank) + 1)
                    + 0.20 / (RRF_K + bm25_ranks.get(idx, fallback_rank) + 1)
                    + 0.40 / (RRF_K + year_era_ranks.get(idx, fallback_rank) + 1)
                )
            else:
                score = (
                    0.60 / (RRF_K + faiss_ranks.get(idx, fallback_rank) + 1)
                    + 0.40 / (RRF_K + bm25_ranks.get(idx, fallback_rank) + 1)
                )
            rrf_scores[idx] = score

        top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:RRF_TOP_N]
        candidates = [self._make_result(self._chunks[idx], score) for idx, score in top_rrf]
        return self._rerank(question, candidates)

    def _make_result(self, chunk: dict, score: float) -> RetrievedContext:
        content = chunk["content"]
        if chunk.get("type") == "table" and chunk.get("table_summary"):
            content = f"[Table Summary] {chunk['table_summary']}\n\n{content}"
        return RetrievedContext(source=chunk["source"], content=content[:8000], score=score)