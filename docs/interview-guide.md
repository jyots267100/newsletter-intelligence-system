# Interview Guide

Short, speakable answers for the technical decisions behind this project.
Each section is meant to be answerable out loud in 30-90 seconds.

## Why RAG instead of just asking the LLM directly?

An LLM has no knowledge of the specific PDF a user just uploaded, and
asking it to "remember" a whole document in the prompt doesn't scale past
a few pages and gives no way to verify where an answer came from. RAG
(retrieve relevant chunks, then generate from only those chunks) keeps the
model grounded in the actual document, keeps prompts small and cheap, and
- critically for this project - lets me attach a verifiable page citation
to every claim, because I know exactly which chunks were retrieved.

## Why ChromaDB?

It's an embedded, persistent vector store with no separate server process
to run or operate - `pip install chromadb` and a local directory is the
whole setup. For a single-user, portfolio-scale project that's the right
trade-off: a managed vector DB (Pinecone, Weaviate Cloud) would add
network latency, an account, and a bill for a workload this small.
ChromaDB's Python API is also simple enough that the `VectorRetriever`
wrapper is about 70 lines.

## Why Hugging Face sentence-transformers for embeddings?

They run locally, for free, on CPU, with no API key and no per-call cost
or rate limit - which matters a lot for a project built around a free
Gemini tier where I want to spend that quota on actual generation, not on
embedding every chunk of every test document. `all-MiniLM-L6-v2`
specifically is small (~80MB), fast, and good enough at semantic
similarity for short newsletter-style text; the model name is
configurable via `EMBEDDING_MODEL` if higher quality were ever needed.

## Why BM25 alongside embeddings?

Embeddings are good at paraphrase/meaning, weak at exact terms (names,
numbers, acronyms). BM25 is the reverse. A newsletter question like "what
was Q3 revenue growth?" needs the model to find "23%" and "$4.2 million" -
BM25 handles exact figures far better than cosine similarity on dense
vectors does. Running both and fusing the results gets the benefit of
each without picking one and accepting its blind spot.

## What is Reciprocal Rank Fusion (RRF) and why use it over just averaging scores?

RRF combines two ranked lists by `score = sum(1 / (k + rank))` across
whichever lists a chunk appears in. The key benefit: it only needs each
retriever's *rank order*, not its raw score - so I never have to normalize
or compare a cosine distance against a BM25 score, which live on
completely different, corpus-dependent scales. That made it both simpler
to implement correctly and easier to explain: "a chunk both retrievers
like rises to the top" is intuitive, versus trying to justify a weighted
average of two incomparable numbers.

## Why not semantic search only?

Tried to always ask myself "where would pure semantic search fail here?"
- and the answer is anything keyword-shaped: a specific dollar figure, a
person's name, a product SKU. If a user asks "what did the newsletter say
about CloudNet?" and "CloudNet" only appears once in the whole document,
BM25 will find that chunk reliably; a dense embedding might rank it behind
chunks that are topically similar but don't actually mention CloudNet.
Hybrid retrieval closes that gap.

## How was chunk size chosen?

1000 characters with 150 overlap, chunked *per page* rather than across
the whole document. Per-page chunking is what makes page citations
unambiguous - a chunk never spans two pages, so every chunk has exactly
one correct page number. The size itself (roughly 150-220 words) is a
pragmatic middle ground for newsletter-length prose: large enough to hold
a complete sentence or two of context (so an insight extracted from it
makes sense on its own), small enough that a retrieved chunk is actually
about what the query asked, not a whole page of mixed topics. Both values
are configurable (`CHUNK_SIZE`/`CHUNK_OVERLAP`) because the right numbers
depend on document style - I'd tune them differently for a dense technical
report than for a newsletter.

## How do citations work, and how are hallucinated citations prevented?

The core rule: **the LLM is never the source of truth for a citation.**
For chat, I attach the metadata (`document_id`, `filename`, `page_number`,
`chunk_id`) of whatever chunks were actually retrieved and placed in the
prompt - the model never states a page number that ends up in the UI. For
analysis (summary/insights/actions), I ask the model to reference excerpt
*numbers* like `[Excerpt 3]` rather than page numbers, then resolve those
numbers back to the real chunk metadata server-side. If the model
hallucinates a number outside the range of excerpts it was shown, that
reference is just dropped - it can never produce a citation to content
that wasn't actually retrieved.

## How are hallucinations (not just citations) reduced more generally?

A few layers: (1) answers are generated only from retrieved chunks, never
from the model's general knowledge, and the system prompt says so
explicitly; (2) if retrieval returns zero chunks, the system returns a
fixed "I couldn't find enough information" message *without even calling
the LLM* - there's nothing to hallucinate from; (3) for structured
analysis output, every insight/action item has a `source_type` of
"explicit" or "inferred" so a reader can tell interpretation apart from
what's literally stated; (4) action items are specifically instructed
never to invent an owner or deadline, and a Pydantic validator additionally
strips out any placeholder value like "TBD"/"unknown" that slips through.

## How is PDF prompt injection handled?

PDF content is treated as untrusted input end-to-end. Every prompt that
includes document excerpts wraps them in clear delimiters and explicitly
tells the model that any instruction-like text inside those delimiters
("ignore previous instructions", "reveal your system prompt") is document
content to analyze, never a command to follow. That's a mitigation, not a
guarantee - no prompt can make a generative model 100% immune to being
*influenced* by adversarial input. What actually bounds the damage
structurally is that the model has no tools, no code execution, and no
ability to set `document_id` or any citation field itself - the worst case
is it writes an unwanted sentence, which schema validation or an odd
answer would surface, not a security breach. I included a real example of
an injection attempt in the sample newsletter
(`data/sample/sample_newsletter.pdf`, page 3) to make this testable rather
than theoretical.

## How does conversational memory work?

Client-side, bounded history: the UI keeps the chat transcript and sends
the last `CHAT_HISTORY_TURNS` (default 6) turns with each request - the
server itself is stateless across requests. That history is used two
ways, and kept carefully separate: (1) a small rule-based check (short
message, or contains a pronoun like "it"/"that") decides whether to
prepend the previous user question to the *retrieval* query, so "what are
its risks?" actually retrieves risk-related chunks instead of chunks about
the word "its"; (2) the same bounded history is shown to the LLM as
"conversation so far, context only" - explicitly *not* evidence about the
document. I didn't use an extra LLM call to rewrite the query, specifically
to keep follow-up handling free, fast, deterministic, and testable
without mocking Gemini.

## How is Gemini integrated, and what happens when it fails?

Through Google's official `google-genai` SDK, wrapped behind a small
`LLMClient` abstract interface so services depend on an interface, not a
concrete SDK. The model name is a config value (`GEMINI_MODEL`), not
hardcoded, because free-tier model availability changes over time.
Failure handling distinguishes transient from permanent: a bad API key or
an unknown model fails immediately with a clear error (retrying won't
help); a rate limit, timeout, or 5xx is retried a small, bounded number of
times (`LLM_MAX_RETRIES`, exponential backoff via `tenacity`) before
surfacing a typed exception that the API layer maps to the right HTTP
status (429, 504, 502, 503). Nothing is silently swallowed into a
fabricated "success" - a failure is always a failure the caller can see.

## How does testing work, and why mock Gemini?

Three layers: unit tests (services/retrieval/schemas in isolation, with
lightweight stub retrievers and a scriptable `FakeLLMClient`), API tests
(FastAPI `TestClient` with service-level dependency overrides, so routes
are tested without touching real storage or the real LLM), and integration
tests (the real ChromaDB + BM25 + PyMuPDF pipeline end-to-end, still with
no Gemini calls). Gemini is mocked in all of the above specifically so the
suite runs for free, deterministically, and without needing a configured
API key in CI. The one place real Gemini calls happen is optional:
`scripts/evaluate_retrieval.py --with-llm`, run manually, which does use
quota.

## What would change for a real production deployment?

- Swap the JSON-file document registry for a real database (even SQLite
  would remove the "read-modify-write whole file" pattern; Postgres for
  multi-instance deployments).
- Add authentication/authorization and per-user document isolation - today
  there's no concept of a "user" at all.
- Move PDF processing off the request thread into a background
  job/queue so uploads don't block on embedding + Gemini latency, with a
  polling or websocket status endpoint instead of a synchronous response.
- Add structured, queryable logging/tracing (e.g. OpenTelemetry) instead
  of line-based logs, plus real dashboards for retrieval/LLM latency.
- Add OCR for scanned PDFs if that's a real user need.
- Reassess chunking/retrieval parameters against a larger, real evaluation
  set once real usage data exists, and only then consider a reranker if
  precision is actually shown to be a problem.
- Add rate limiting and request size/timeout hardening at the API gateway
  level, not just in application code.
