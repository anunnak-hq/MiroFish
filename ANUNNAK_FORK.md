# Anunnak MiroFish Fork

This is the Anunnak AI (`anunnak-hq`) fork of MiroFish — a multi-agent prediction engine built on CAMEL-AI's OASIS. It is deployed as **one feature** of the Anunnak platform (a full AI business launch suite — builder, media studio, bots, ads, CRM, creative intelligence) served at [anunnak.com](https://anunnak.com).

## Why we fork

MiroFish upstream (`github.com/666ghj/MiroFish`) is excellent for Chinese-domestic LLM providers (Qwen via Bailian) but needs small adjustments to run cleanly against the Anthropic API and our internal billing proxy. We also add a JSON-mode entry point so our higher-level agent Gosha can feed it research output directly without the Vue file-upload UI.

## Branches

- `main` — tracks upstream `666ghj/MiroFish:main`. Never commit to this branch directly. Rebase periodically from upstream.
- `anunnak-prod` — production branch for the Anunnak deployment. Three commits on top of a base upstream commit:
  1. **Anthropic compat + per-request model via `X-LLM-Model` header** (`backend/app/utils/llm_client.py`, `backend/app/services/oasis_profile_generator.py`, `backend/app/services/simulation_config_generator.py`)
  2. **Rich-brief JSON mode** for `/api/graph/ontology/generate` (`backend/app/api/graph.py`)
  3. **Fork documentation** (this file)

All three are additive. The upstream form-data file-upload flow is untouched, so running the Anunnak fork against the original Vue frontend behaves identically to upstream.

## What we changed — summary

### 1. Anthropic-compatible LLM calls

Anthropic's `/v1/messages` endpoint rejects the OpenAI-style `response_format={"type":"json_object"}` param. Our fork strips that param when the configured `LLM_BASE_URL` points at Anthropic or at the Anunnak proxy (`anunnak.com`), which is a drop-in Anthropic replacement. The callers already prompt for JSON explicitly, and the downstream parsers tolerate plain JSON text.

### 2. Per-request model override via HTTP header

Anunnak's Gosha UI lets the user pick a Claude model (Haiku / Sonnet / Opus) per-prompt. Rather than restarting the MiroFish container every time the user switches models, we accept an `X-LLM-Model` header on any request into MiroFish, and `llm_client.LLMClient` uses that model for every downstream LLM call on the same request thread. Fallback priority: explicit constructor arg → `X-LLM-Model` header → env `LLM_MODEL_NAME`.

### 3. Rich-brief JSON mode for ontology generation

`POST /api/graph/ontology/generate` now accepts a JSON payload with a `documents[]` array:

```json
{
  "simulation_requirement": "short user question",
  "project_name": "anunnak_<timestamp>",
  "additional_context": "optional",
  "documents": [
    {"filename": "industry_overview.md", "text": "<kilobytes>"},
    {"filename": "competitor_snapshot.md", "text": "<kilobytes>"},
    ...
  ]
}
```

Gosha's research phase (`web_fetch` + `delegate_subtask` MCP tools) populates this array with real web content before calling MiroFish. Each document becomes a separate input to the ontology extractor, so the extractor sees the same text volume it would from a manual file upload through the Vue frontend.

**Why this matters**: MiroFish's total simulation agent count is a function of how many entity types the ontology extractor finds in its input text. A 200-char seed produces 1-2 entities, hence 1-2 agents, hence a trivial simulation. A 5-500 KB brief produces 15-300+ entities, hence realistic multi-persona debate. This JSON mode is what makes the "thousands of intelligent agents" narrative practical through an API call — Gosha does the legwork to produce the volume.

### What we did NOT change

- Upstream form-data file-upload path — intact, identical behavior.
- OASIS engine internals — untouched.
- Persona generation logic in `oasis_profile_generator.py` — only Anthropic compat fix, no schema changes.
- Report generation ReACT loop in `report_agent.py` — untouched (upstream non-convergence bugs still apply, see upstream issues).
- Simulation runner / IPC layer — untouched.

## AGPL-3.0 compliance

Upstream MiroFish is licensed under AGPL-3.0. Because Anunnak hosts a modified version of this code as a network service, AGPL-3.0 §13 requires us to make the source available to users of that service.

**This fork is public at** [github.com/anunnak-hq/MiroFish](https://github.com/anunnak-hq/MiroFish). The `anunnak-prod` branch is the exact source running in production at the time of release. Deployment tags will match git commits 1:1.

If you are an Anunnak user and want the source of the specific version you interacted with, open an issue on this repo or email `legal@anunnak.com` with the request and an approximate date/time.

Upstream license preserved: see [LICENSE](./LICENSE).

## Running the Anunnak deployment

Our production container is built from the `anunnak-prod` branch of this fork and published as `ghcr.io/anunnak-hq/mirofish:latest`. Env vars the Anunnak deployment sets:

```
LLM_BASE_URL   = https://anunnak.com/api/claw/proxy/{user_uuid}/v1
LLM_API_KEY    = <placeholder, auth is URL-path-based on the proxy>
LLM_MODEL_NAME = claude-haiku-4-5-20251001
```

Every LLM call inside MiroFish is thus routed through the Anunnak Anthropic-compatible proxy, which applies a ×3 markup and debits the per-user wallet for the call. The `X-LLM-Model` header lets the Anunnak caller swap the model per-request without restarting the container.

## Credits

- Upstream MiroFish: [666ghj/MiroFish](https://github.com/666ghj/MiroFish), built on [CAMEL-AI OASIS](https://github.com/camel-ai/oasis), supported by Shanda Group.
- Anunnak integration: the Anunnak AI team, Las Vegas.
