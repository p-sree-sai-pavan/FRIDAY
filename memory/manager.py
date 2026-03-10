import asyncio
import atexit
import sys, os
import json
import uuid
import time
import sqlite3
import numpy as np
from groq import AsyncGroq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector, Prefetch, FusionQuery, Fusion
)
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

# ========================
# PATHS
# ========================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH  = os.path.abspath(os.path.join(BASE_DIR, "..", config.MEMORY_PATH, "qdrant"))
HISTORY_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", config.MEMORY_PATH, "memory.json"))
SEMANTIC_DB  = os.path.abspath(os.path.join(BASE_DIR, "..", config.MEMORY_PATH, "facts.db"))

os.makedirs(QDRANT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

# ========================
# SHARED GROQ CLIENT — single instance, imported by brain.py
# ========================
groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)

# ========================
# INIT — models + DB + Qdrant
# ========================
embedder        = SentenceTransformer(config.EMBEDDING_MODEL)
sparse_embedder = SparseTextEmbedding(model_name=config.SPARSE_MODEL)

# Fix D: derive vector size from model at init — never hardcode
VECTOR_SIZE = embedder.get_sentence_embedding_dimension()

# Fix C: bounded working memory vector cache
_working_cache: dict = {}
_CACHE_MAX  = 500
_CACHE_TRIM = 100

# ALL tables at init — never inside functions
db = sqlite3.connect(SEMANTIC_DB, check_same_thread=False)
db.execute("""CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY, category TEXT, key TEXT UNIQUE,
    value TEXT, updated_at INTEGER
)""")
db.execute("""CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY, query TEXT, context TEXT,
    answer TEXT, was_useful INTEGER, timestamp INTEGER
)""")
db.execute("""CREATE TABLE IF NOT EXISTS user_profile (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""")
db.commit()

# Fix 17: graceful shutdown
def _shutdown():
    try: db.close()
    except: pass
    try: qdrant.close()
    except: pass
atexit.register(_shutdown)

# Qdrant hybrid collection
qdrant = QdrantClient(path=QDRANT_PATH)
_cols  = [c.name for c in qdrant.get_collections().collections]
if config.MEMORY_COLLECTION not in _cols:
    qdrant.create_collection(
        collection_name=config.MEMORY_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)  # Fix D
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        }
    )
    print(f"Created hybrid collection: {config.MEMORY_COLLECTION} (dim={VECTOR_SIZE})")

# Stopwords for semantic search
STOPWORDS = {
    "the","and","for","are","but","not","you","all","can","had","her",
    "was","one","our","out","day","get","has","him","his","how","its",
    "may","new","now","old","see","two","who","did","have","what","when",
    "with","from","that","this","will","your","been","does","just","into",
    "over","such","than","then","they","them","also","pavan","tell",
    "show","give","find","make","like","know","want","need","help","about"
}


# ========================
# HELPERS
# ========================

def _cosine(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# Fix A+B: all CPU/IO ops wrapped in asyncio.to_thread

async def _encode(text: str) -> np.ndarray:
    return await asyncio.to_thread(
        lambda: embedder.encode(text, show_progress_bar=False)
    )

async def _sparse(text: str) -> SparseVector:
    """Encode text to sparse BM25 vector — offloaded to thread."""
    def _run():
        r = list(sparse_embedder.embed([text]))[0]
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
    return await asyncio.to_thread(_run)

async def _db_execute(sql: str, params: tuple = ()):
    """Run SQLite query in thread — never block event loop."""
    def _run():
        cur = db.execute(sql, params)
        db.commit()
        return cur.fetchall()
    return await asyncio.to_thread(_run)

async def _db_read(sql: str, params: tuple = ()):
    """Read-only SQLite query in thread."""
    def _run():
        db.row_factory = sqlite3.Row  # Enable dictionary-like access
        res = db.execute(sql, params).fetchall()
        db.row_factory = None  # Reset
        return res
    return await asyncio.to_thread(_run)

async def get_user_profile() -> dict:
    """Retrieve user profile data for form filling context."""
    try:
        rows = await _db_read("SELECT key, value FROM user_profile")
        return {row["key"]: row["value"] for row in rows}
    except Exception as e:
        print(f"User profile read error: {e}")
        return {}


# ========================
# HYDE
# ========================

async def hyde_transform(query: str) -> str:
    resp = await groq_client.chat.completions.create(
        model=config.FAST_MODEL, temperature=0.3, max_tokens=60,
        messages=[{"role": "user", "content":
            f"Expand into search keywords only. No answer. Query: {query}\nSearch terms:"}]
    )
    return f"{query} {resp.choices[0].message.content.strip()}"


# ========================
# SEARCH
# ========================

async def search_working(query: str) -> dict:
    try:
        if not os.path.exists(HISTORY_FILE):
            return {"source": "working", "content": [], "relevance": 0}

        # Fix B: file I/O in thread
        def _load():
            with open(HISTORY_FILE) as f:
                return json.load(f)
        history = await asyncio.to_thread(_load)

        if not history:
            return {"source": "working", "content": [], "relevance": 0}

        q_vec = await _encode(query)
        scored = []

        for msg in history:
            content = msg.get("content", "")
            if not content:
                continue

            # Fix C: bounded cache
            if content not in _working_cache:
                if len(_working_cache) >= _CACHE_MAX:
                    for k in list(_working_cache.keys())[:_CACHE_TRIM]:
                        del _working_cache[k]
                _working_cache[content] = await _encode(content)

            score = _cosine(q_vec, _working_cache[content])
            if score >= config.MIN_RELEVANCE:
                scored.append({"content": content, "role": msg.get("role", ""), "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:5]
        avg = sum(r["score"] for r in top) / len(top) if top else 0
        return {"source": "working", "content": top, "relevance": avg}

    except Exception as e:
        print(f"Working memory error: {e}")
        return {"source": "working", "content": [], "relevance": 0}


async def search_episodic(query: str) -> dict:
    try:
        d_vec = (await _encode(query)).tolist()
        s_vec = await _sparse(query)

        # Fix B: Qdrant query in thread
        def _query():
            return qdrant.query_points(
                collection_name=config.MEMORY_COLLECTION,
                prefetch=[
                    Prefetch(query=d_vec, using="dense", limit=10),
                    Prefetch(query=s_vec, using="sparse", limit=10)
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=5,
                with_vectors=True
            ).points
        results = await asyncio.to_thread(_query)

        if not results:
            return {"source": "episodic", "content": [], "relevance": 0}

        rescored = []
        for r in results:
            stored = r.vector.get("dense") if r.vector else None
            if stored is None:
                continue
            cos = _cosine(d_vec, stored)
            if cos >= config.MIN_RELEVANCE:
                rescored.append({"text": r.payload.get("text", ""), "score": cos})

        if not rescored:
            return {"source": "episodic", "content": [], "relevance": 0}

        rescored.sort(key=lambda x: x["score"], reverse=True)
        avg = sum(r["score"] for r in rescored) / len(rescored)
        return {"source": "episodic", "content": rescored, "relevance": avg}

    except Exception as e:
        print(f"Episodic error: {e}")
        return {"source": "episodic", "content": [], "relevance": 0}


async def search_semantic(query: str) -> dict:
    try:
        words = [w for w in query.lower().split() if len(w) >= 4 and w not in STOPWORDS]
        if not words:
            return {"source": "semantic", "content": [], "relevance": 0}

        conditions, params = [], []
        for word in words:
            conditions.append("(LOWER(category) LIKE ? OR LOWER(key) LIKE ? OR LOWER(value) LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%", f"%{word}%"])

        # Fix B: SQLite in thread
        rows = await _db_read(
            f"SELECT DISTINCT category, key, value FROM facts WHERE {' OR '.join(conditions)}",
            tuple(params)
        )

        if not rows:
            return {"source": "semantic", "content": [], "relevance": 0}

        return {
            "source": "semantic",
            "content": [{"category": r[0], "key": r[1], "value": r[2]} for r in rows],
            "relevance": 1.0
        }
    except Exception as e:
        print(f"Semantic error: {e}")
        return {"source": "semantic", "content": [], "relevance": 0}


# ========================
# WRITE
# ========================

async def write_episodic(prompt: str, response: str):
    try:
        text  = f"User: {prompt}\nFRIDAY: {response}"
        d_vec = (await _encode(text)).tolist()
        s_vec = await _sparse(text)

        # Fix 4: cosine duplicate check via stored dense vector
        def _dup_check():
            return qdrant.query_points(
                collection_name=config.MEMORY_COLLECTION,
                prefetch=[Prefetch(query=d_vec, using="dense", limit=1)],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=1, with_vectors=True
            ).points
        existing = await asyncio.to_thread(_dup_check)

        if existing:
            sv = existing[0].vector.get("dense") if existing[0].vector else None
            if sv and _cosine(d_vec, sv) > config.SIMILARITY_THRESHOLD:
                return

        # Fix 8: integer UUID, Fix B: upsert in thread
        point = PointStruct(
            id=uuid.uuid4().int >> 64,
            vector={"dense": d_vec, "sparse": s_vec},
            payload={"text": text, "prompt": prompt, "response": response, "timestamp": int(time.time())}
        )
        await asyncio.to_thread(qdrant.upsert, config.MEMORY_COLLECTION, [point])

    except Exception as e:
        print(f"Episodic write error: {e}")


async def write_semantic(category: str, key: str, value: str):
    try:
        await _db_execute(
            "INSERT INTO facts (category,key,value,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=?,updated_at=?",
            (category, key, value, int(time.time()), value, int(time.time()))
        )
    except Exception as e:
        print(f"Semantic write error: {e}")


# ========================
# COMPRESS
# ========================

def compress_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        src = r["source"]
        if src == "semantic":
            for item in r["content"]:
                parts.append(f"[FACT] {item['key']}: {item['value']}")
        elif src in ("episodic", "web_fallback"):
            for item in r["content"]:
                parts.append(f"[MEMORY] {item.get('text', '')}")
        elif src == "working":
            for item in r["content"]:
                parts.append(f"[RECENT] {item.get('role','')}: {item.get('content','')}")
    return "\n".join(parts)


# ========================
# CRAG
# ========================

async def _crag_grade_single(query: str, result: dict) -> str:
    avg_score = result.get("relevance", 0)

    # Use score thresholds first — skip LLM for clear cases
    if avg_score >= config.CRAG_CORRECT_THRESHOLD:
        return "CORRECT"
    if avg_score < config.CRAG_AMBIGUOUS_THRESHOLD:
        return "INCORRECT"

    # LLM only for middle-ground scores
    compressed = compress_results([result])
    resp = await groq_client.chat.completions.create(
        model=config.FAST_MODEL, temperature=0, max_tokens=10,
        messages=[{"role": "user", "content":
            f"Grade: CORRECT, AMBIGUOUS, or INCORRECT.\nQuery: {query}\nContext: {compressed}\nGrade:"}]
    )
    return resp.choices[0].message.content.strip().upper()


async def crag_grade(query: str, results: list) -> dict:
    if not results:
        return {"correct": [], "ambiguous": [], "incorrect": True}

    grades = await asyncio.gather(*[_crag_grade_single(query, r) for r in results])

    correct, ambiguous = [], []
    for result, grade in zip(results, grades):
        if "CORRECT" in grade and "INCORRECT" not in grade:
            correct.append(result)
        elif "AMBIGUOUS" in grade:
            ambiguous.append(result)

    return {
        "correct": correct,
        "ambiguous": ambiguous,
        "incorrect": len(correct) == 0 and len(ambiguous) == 0
    }


async def crag_web_search(query: str) -> list:
    """Real web search via tools/search.py (Tavily & DuckDuckGo)."""
    try:
        from tools.search import search_for_crag
        return await search_for_crag(query)
    except Exception as e:
        print(f"[crag_web_search] tools/search failed: {e}")

    # Pure RAG: Return empty list if search fails instead of hallucinating with an LLM fallback.
    return []


# ========================
# SELF-RAG + FEEDBACK — called from brain.py
# ========================

async def self_rag_reflect(query: str, answer: str, context: str) -> bool:
    resp = await groq_client.chat.completions.create(
        model=config.FAST_MODEL, temperature=0, max_tokens=5,
        messages=[{"role": "user", "content":
            f"Does this answer address the query? YES or NO.\nQuery: {query}\nAnswer: {answer}\nVerdict:"}]
    )
    return "YES" in resp.choices[0].message.content.strip().upper()


async def save_feedback(query: str, context: str, answer: str, was_useful: bool):
    try:
        await _db_execute(
            "INSERT INTO feedback (query,context,answer,was_useful,timestamp) VALUES(?,?,?,?,?)",
            (query, context, answer, 1 if was_useful else 0, int(time.time()))
        )
    except Exception as e:
        print(f"Feedback error: {e}")


# ========================
# PUBLIC INTERFACE
# ========================

async def read(query: str) -> list:
    """Full pipeline: HyDE → Hybrid search → CRAG grade → fallback."""
    # Skip HyDE if memory is empty — saves 1 API call
    def _count():
        return qdrant.count(collection_name=config.MEMORY_COLLECTION).count
    count = await asyncio.to_thread(_count)

    expanded = await hyde_transform(query) if count > 0 else query

    results = await asyncio.gather(
        search_working(expanded),
        search_episodic(expanded),
        search_semantic(query)   # semantic uses original query for word match
    )
    results = [r for r in results if r["content"]]
    results.sort(key=lambda x: x["relevance"], reverse=True)

    if not results:
        return await crag_web_search(query)

    grade = await crag_grade(query, results)

    if grade["correct"]:
        return grade["correct"]
    if grade["ambiguous"]:
        web = await crag_web_search(query)
        return grade["ambiguous"] + web

    return await crag_web_search(query)


async def write(prompt: str, response: str):
    await write_episodic(prompt, response)