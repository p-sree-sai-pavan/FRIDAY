<div align="center">

```
███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗
██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝
█████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝
██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝
██║     ██║  ██║██║██████╔╝██║  ██║   ██║
╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝
```

### Autonomous AI Assistant — Built from Scratch

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Ollama](https://img.shields.io/badge/Ollama-Offline%20Mode-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22C55E?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-6366F1?style=for-the-badge)](LICENSE)

<br/>

> *No LangChain. No LlamaIndex. No abstractions hiding the decisions.*
> *Every component — the agent loop, memory pipeline, tool registry, safety layer, and voice I/O — designed and wired together manually.*

<br/>

</div>

---

## What is FRIDAY?

FRIDAY is a personal AI assistant that runs on your machine. You talk to it — text or voice. It figures out what you need, decides whether to answer from memory or use a tool, executes tools with your permission where needed, and responds. Conversations are remembered across sessions using a three-layer memory architecture.

The goal was never to ship a product. It was to understand exactly how an autonomous AI agent works at every level — retrieval, grading, tool dispatch, safety, memory writes — by building each piece from the ground up.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                          main.py                                │
  │              Startup menus → REPL → graceful shutdown           │
  └────────────────────────────┬────────────────────────────────────┘
                               │
  ┌────────────────────────────▼────────────────────────────────────┐
  │                     core/orchestrator.py                        │
  │                                                                 │
  │   1. Load system prompt + history      (brain.py)               │
  │   2. Memory retrieval + context inject (memory/api.py)          │
  │   3. First LLM call                    (model_client.py)        │
  │   4. Tool loop if tool calls detected  (safety.py + registry)   │
  │   5. Faithfulness check                (brain.py)               │
  │   6. Save to memory + extract facts    (memory/api.py)          │
  └──────────┬──────────────────┬──────────────────────────────────-┘
             │                  │
  ┌──────────▼──────┐  ┌────────▼────────────────────────────────── ┐
  │  memory/api.py  │  │              tools/                         │
  │                 │  │                                             │
  │  HyDE expand    │  │  registry.py   ← tool catalog + schemas     │
  │  ↓              │  │  search.py     ← Tavily / DuckDuckGo        │
  │  Parallel search│  │  system.py     ← shell + system info        │
  │  ↓              │  │  files.py      ← file read/write            │
  │  CRAG grading   │  │  browser.py    ← browser automation         │
  │  ↓              │  │  apps/         ← Windows app launcher       │
  │  Web fallback   │  └─────────────────────────────────────────────┘
  └──────────┬──────┘
             │
  ┌──────────▼──────────────────────────────────────────────────────┐
  │                       Memory Layers                             │
  │                                                                 │
  │  working.py  → recent history (cosine-scored, JSON)             │
  │  qdrant.py   → episodic memory (dense + sparse hybrid, RRF)     │
  │  sqlite.py   → semantic facts (structured, auto-extracted)      │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                          io/                                    │
  │  ears.py  → mic → webrtcvad → Whisper (Groq online / GPU local) │
  │  mouth.py → edge-tts (online) / pyttsx3 (offline) → speakers   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## How a Request Flows — Step by Step

```
You speak or type
        │
        ▼
  ears.py captures mic → VAD detects speech → Whisper transcribes
        │
        ▼
  orchestrator.py receives prompt
        │
        ├─ brain.py loads system prompt + conversation history
        │
        ├─ memory/api.py runs retrieval pipeline:
        │       HyDE (query expansion)
        │         → parallel search: working + episodic + semantic
        │         → CRAG grades results: CORRECT / AMBIGUOUS / INCORRECT
        │         → web fallback if memory empty or all INCORRECT
        │         → inject relevant context into system prompt
        │
        ├─ LLM call (Groq / Ollama based on selected mode)
        │
        ├─ Plain text response?
        │       → faithfulness check (score 0–1)
        │       → if score ≥ 0.5: save to episodic memory
        │       → background: extract facts → save to SQLite
        │       → return response
        │
        └─ Tool calls detected?
                → safety.py checks risk level
                → READ: auto-execute
                → WRITE: auto-execute + notify you
                → SYSTEM / IRREVERSIBLE: ask confirmation
                → tools run in parallel (asyncio.gather)
                → results fed back to LLM
                → repeat up to 12 iterations
                → final plain text → faithfulness check → save
                        │
                        ▼
                  mouth.py speaks the response
```

---

## Memory System

FRIDAY's memory is not a single vector store. It is three coordinated layers.

<details>
<summary><b>Working Memory</b> — short-term, recent context</summary>

<br/>

**File:** `memory/core/working.py`

Reads from `memory.json` (the raw conversation history). Before injecting it, each message is scored against the current query using **cosine similarity** — only messages above a relevance threshold are included. This means FRIDAY doesn't blindly dump the last N messages into context; it picks the ones that actually matter.

```python
# Scored, filtered — not raw history dump
scored = [(msg, cosine_similarity(query_vec, msg_vec)) for msg in recent]
top = [m for m, score in scored if score >= MIN_RELEVANCE]
```

</details>

<details>
<summary><b>Episodic Memory</b> — long-term, vector search</summary>

<br/>

**File:** `memory/core/qdrant.py`

Stores every conversation turn in a local Qdrant instance. Each entry is indexed with **two vectors simultaneously**:

- **Dense vector** — `all-MiniLM-L6-v2` sentence embeddings (semantic meaning)
- **Sparse vector** — BM25 (keyword frequency)

At retrieval time, both are queried in parallel and results are merged using **Reciprocal Rank Fusion (RRF)**. Dense search finds semantically similar conversations; sparse search finds exact keyword matches. Together they catch what either alone would miss.

```python
results = qdrant.query_points(
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=10),
        Prefetch(query=sparse_vec, using="sparse", limit=10)
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5
)
```

</details>

<details>
<summary><b>Semantic Memory</b> — structured facts</summary>

<br/>

**File:** `memory/core/sqlite.py`

After every conversation turn, a background task runs a fast LLM call to extract explicit facts stated by you — deadlines, preferences, personal details, decisions. These are saved to `facts.db` (SQLite) with category, key, and value fields.

```json
{ "category": "work", "key": "internship_deadline", "value": "CV submission in 2 months" }
```

No inference, no assumptions — only facts you explicitly stated.

</details>

<details>
<summary><b>CRAG Pipeline</b> — grading what gets used</summary>

<br/>

**File:** `memory/core/pipeline.py`

Not all retrieved results are useful. CRAG (Corrective RAG) grades each result before it enters the LLM context:

| Grade | Condition | Action |
|---|---|---|
| `CORRECT` | Relevance score ≥ 0.7 | Use as-is |
| `AMBIGUOUS` | Score between 0.4–0.7 | Use + supplement with web search |
| `INCORRECT` | Score < 0.4 | Discard, fall back to web |

Before retrieval, **HyDE** (Hypothetical Document Embeddings) expands the query using a fast LLM call — generating keywords that improve vector search recall without changing the actual query.

**Faithfulness gate:** after the LLM responds, a 0–1 score checks whether the answer actually reflects the retrieved context. Responses scoring below 0.5 are not written back to episodic memory — hallucinated answers can't corrupt future retrievals.

</details>

---

## Tool System

<details>
<summary><b>Registry</b> — self-registering, schema-exporting</summary>

<br/>

**File:** `tools/registry.py`

Every tool registers itself at import time. The registry automatically exports all tools as **Groq function-calling schema** — no manual schema writing per tool.

```python
registry.register(Tool(
    name="web_search",
    description="...",
    parameters={ ... },   # JSON Schema
    risk=RiskLevel.READ,
    handler=_search
))

# Orchestrator calls:
tool_schemas = registry.to_groq_tools()   # ready to pass to Groq API
```

</details>

<details>
<summary><b>Safety Layer</b> — 4-tier risk classification</summary>

<br/>

**File:** `core/safety.py`

Every tool call passes through the safety layer before execution. No exceptions.

| Risk Level | Behaviour | Examples |
|---|---|---|
| `READ` | Auto-execute silently | web search, system info, read file |
| `WRITE` | Auto-execute, prints what it's doing | open app, write file |
| `SYSTEM` | Asks your confirmation first | run shell command |
| `IRREVERSIBLE` | Always asks, no override | delete file, send message |

</details>

<details>
<summary><b>Web Search</b> — two-tier fallback</summary>

<br/>

**File:** `tools/search.py`

| Tier | Provider | Key required | Quality |
|---|---|---|---|
| 1 | Tavily | Yes (free: 1000 req/month) | RAG-optimised, pre-extracted content |
| 2 | DuckDuckGo | No | Always available, free |

Tavily is tried first. If it fails or returns empty, DuckDuckGo runs automatically. The CRAG pipeline also calls this as a fallback when memory has nothing relevant.

</details>

<details>
<summary><b>Available Tools</b></summary>

<br/>

| Tool | File | Risk | Description |
|---|---|---|---|
| `web_search` | `tools/search.py` | READ | Tavily + DuckDuckGo with auto-fallback |
| `run_command` | `tools/system.py` | SYSTEM | Execute any shell command (30s timeout) |
| `get_system_info` | `tools/system.py` | READ | CPU, RAM, disk, battery, network, uptime |
| `read_file` | `tools/files.py` | READ | Read any file from disk |
| `write_file` | `tools/files.py` | WRITE | Write or create files |
| `open_app` | `tools/apps/` | WRITE | Launch Windows applications by name |
| `browser` | `tools/browser.py` | WRITE | Browser automation |

</details>

---

## LLM Routing

**File:** `core/model_client.py` + `config.py`

Two modes, selected at startup via an interactive menu. Switchable mid-session with the `switch` command.

```
Online mode  (Groq API)              Offline mode  (Ollama — local GPU)
─────────────────────────────        ──────────────────────────────────
Primary  : llama-3.3-70b-versatile   Primary  : llama3.1:8b
Fast     : llama-3.1-8b-instant      Fast     : llama3.2:3b
Tools    : llama-3.3-70b-versatile   Tools    : llama3.1:8b
Fallback : gemini-2.5-flash          Fallback : none (fully local)
```

The **fast model** handles lightweight calls — query classification, HyDE expansion, CRAG grading, fact extraction, faithfulness checks. The **primary model** handles the actual conversation and tool calls. This reduces quota usage significantly on trivial requests.

---

## Voice I/O

<details>
<summary><b>Input — ears.py</b></summary>

<br/>

```
Microphone (sounddevice, 16kHz)
        │
        ▼
WebRTC VAD — 30ms frames, detects speech vs silence
        │
        ▼
Buffer until 1.5s silence after speech ends
        │
        ├── Online  → Groq Whisper API (whisper-large-v3) — ~sub-1s
        └── Offline → faster-whisper on GPU (float16, CUDA)
```

A 10-frame rolling pre-buffer prevents clipping the start of your utterance. Short noise bursts (< 5 voiced frames) are discarded before STT is called.

</details>

<details>
<summary><b>Output — mouth.py</b></summary>

<br/>

```
LLM response text
        │
        ▼
_clean_for_speech() — strips **markdown**, # headers, `code`, bullet points
        │
        ├── Online  → edge-tts (Microsoft Neural TTS, en-US-JennyNeural)
        │             MP3 stream → pydub decode → sounddevice playback
        └── Offline → pyttsx3 (Windows SAPI) → WAV → sounddevice playback
```

Interrupt support — calling `speak()` while audio is playing stops the current speech immediately before starting the new one. Low-latency: checks stop flag every 100ms during playback.

</details>

---

## Setup

### Prerequisites

```
Python 3.11+
ffmpeg (for voice output MP3 decode)
```

Install ffmpeg:
```bash
winget install ffmpeg
```

### Install

```bash
git clone https://github.com/p-sree-sai-pavan/FRIDAY.git
cd FRIDAY
pip install -r requirements.txt
```

For voice support:
```bash
pip install sounddevice webrtcvad edge-tts pydub pyttsx3 faster-whisper
```

### Configure

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here        # optional — Groq fallback
TAVILY_API_KEY=your_key_here        # optional — falls back to DuckDuckGo
```

| Key | Where to get | Required |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Yes (for online mode) |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | No |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) | No |

### For offline mode

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
ollama pull llama3.2:3b
ollama serve
```

### Run

```bash
python main.py
```

You'll see an interactive startup menu:

```
  Select AI Mode
  ──────────────────────────────────────
  [1]  ● Online   — Groq API
         Primary : llama-3.3-70b-versatile
         ✓ API key found

  [2]  ● Offline  — Ollama (local)
         Primary : llama3.1:8b
         ✓ Ollama running | 3 model(s) available
```

---

## Project Structure

```
FRIDAY/
├── main.py                      # entry point, startup menus, REPL
├── config.py                    # all settings, model names, paths, thresholds
├── requirements.txt
│
├── core/
│   ├── orchestrator.py          # agent loop — the heart of FRIDAY
│   ├── brain.py                 # history, system prompt, faithfulness check
│   ├── model_client.py          # Groq / Ollama client factory (lazy init)
│   ├── safety.py                # risk-level gating for every tool call
│   └── signal.py                # graceful shutdown on SIGINT/SIGTERM
│
├── memory/
│   ├── api.py                   # public read() / write() interface
│   └── core/
│       ├── pipeline.py          # HyDE query expansion + CRAG grading
│       ├── qdrant.py            # episodic memory — hybrid vector search
│       ├── sqlite.py            # semantic facts — structured SQLite storage
│       ├── working.py           # short-term memory — cosine-scored history
│       ├── resources.py         # Qdrant client + embedding model init
│       └── utils.py             # encode_dense, encode_sparse, cosine_similarity
│
├── tools/
│   ├── registry.py              # Tool dataclass, RiskLevel, ToolRegistry
│   ├── search.py                # Tavily + DuckDuckGo with auto-fallback
│   ├── system.py                # shell command execution + system info
│   ├── files.py                 # file read/write operations
│   ├── browser.py               # browser automation (browser-use)
│   └── apps/
│       ├── discovery.py         # scans installed Windows apps
│       ├── handlers.py          # app-specific launch logic
│       ├── resolution.py        # fuzzy name → executable path
│       ├── win32_utils.py       # Windows API helpers
│       └── constants.py         # known app paths + aliases
│
├── io/
│   ├── ears.py                  # mic capture → VAD → Whisper STT
│   └── mouth.py                 # LLM text → TTS → speakers
│
└── data/
    ├── prompts/
    │   └── system_prompt.txt    # FRIDAY's persona + instructions
    ├── memory/
    │   ├── memory.json          # rolling conversation history
    │   ├── facts.db             # SQLite — extracted personal facts
    │   └── qdrant/              # local Qdrant vector store
    └── logs/
        └── friday.log
```

---

## Status

| Component | Status |
|---|---|
| Orchestrator + agent loop | ✅ Complete |
| CRAG + HyDE memory pipeline | ✅ Complete |
| Tool registry + safety layer | ✅ Complete |
| Web search (Tavily + DuckDuckGo) | ✅ Complete |
| System tools (shell + sysinfo) | ✅ Complete |
| File operations | ✅ Complete |
| Windows app launcher | ✅ Complete |
| Groq / Ollama / Gemini routing | ✅ Complete |
| Voice I/O (ears + mouth) | ✅ Written — testing in progress |
| Browser automation | 🔄 In progress |

---

## Design Decisions

**Why no LangChain or LlamaIndex?**
Every abstraction in this project is something I needed to understand — how retrieval grading works, how tool dispatch connects to function calling, how memory writes should be gated. Using a framework would have hidden those decisions behind configuration. Building it manually meant every design choice had to be made explicitly and defended.

**Why three memory layers instead of one vector store?**
Working memory is fast but shallow. Episodic memory is deep but requires embedding. Semantic facts need structured lookup, not similarity search. Each layer solves a different problem. A single vector store would either miss the structured fact case or add unnecessary overhead for recent history.

**Why a faithfulness gate on memory writes?**
An LLM can produce a confident-sounding wrong answer. If that answer gets written back to episodic memory, future retrievals will surface it as context, compounding the error. The faithfulness check breaks that feedback loop.

**Why parallel tool execution?**
If a request requires web search + system info + file read, running them sequentially wastes time. `asyncio.gather()` runs them concurrently. The LLM waits only as long as the slowest tool, not the sum of all.

---

<div align="center">

<br/>

Built by [Pittala Sree Sai Pavan](https://github.com/p-sree-sai-pavan) · IIT Guwahati

<br/>

</div>