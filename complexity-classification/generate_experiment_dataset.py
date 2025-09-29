#!/usr/bin/env python3
"""
Generate needle-in-the-haystack tool-retrieval datasets (Simple/Hard buckets).

Inputs (default to files in current directory):
- server_scores.json: contains per-server level classification (Simple/Hard)
- servers_parsed.json: server metadata (id, name)
- tools_parsed.json: flattened tools with server_id, tool_name, tool_description, input_schema

Outputs:
- dataset_simple.json: 50 items
- dataset_hard.json: 50 items

Each dataset item is a pair:
{ "request": <natural language request>,
  "target_tool": {
      "server_id": str,
      "server_name": str,
      "tool_name": str
  },
  "bucket": "Simple"|"Hard"
}

Generation approach:
- Uniform-by-server sampling within each bucket; sample tools uniformly within server.
- Request generator sees ONLY the single target tool (name/description/input_schema).
- Two-stage prompting inspired by mcp-bench/synthesis/task_synthesis.py:
  1) Detailed task description (single-tool solvable)
  2) Natural, conversational fuzzy request derived from the detailed task

Note: This script requires vLLM. If you pass a Hugging Face model id to
--model and have network access, vLLM can download weights/tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------- IO Helpers -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------- Data Structures -----------------------

@dataclass
class ToolRecord:
    server_id: str
    server_name: str
    tool_name: str
    tool_description: str
    input_schema: Dict[str, Any]


# ---------------------- vLLM Wrapper -------------------------

class VLLMClient:
    def __init__(self, model: str):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "vLLM is required. Please install vllm and ensure the model/tokenizer are available locally."
            ) from e

        # Use conservative defaults; user can change model weights locally
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model, trust_remote_code=True)

    def generate(self, prompt: str, max_new_tokens: int = 800, temperature: float = 0.7, top_p: float = 0.95) -> str:
        params = self.SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        outputs = self.llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()


# ---------------------- Prompt Builders ----------------------

OPENAPI_SPECIAL = """
CRITICAL REQUIREMENTS FOR OPENAPI TASKS:
- The task MUST be about ANALYZING/UNDERSTANDING an API specification using the provided tool, NOT calling external APIs
- Do NOT reference external services (Stripe, Slack, etc.) or perform real operations
- Focus on operations, parameters, response schemas, authentication/security schemes, deprecations, or pagination patterns
""".strip()


def build_detailed_task_prompt(server_name: str, tool: ToolRecord) -> str:
    schema_str = json.dumps(tool.input_schema, ensure_ascii=False, indent=2)
    special = OPENAPI_SPECIAL if "openapi" in server_name.lower() else ""
    return f"""
You are a task designer for testing AI agents with MCP tools.
{special}

ONLY ONE TOOL IS AVAILABLE (the agent will only have this single tool):
Tool: `{tool.tool_name}` (Server: {tool.server_name})
Description: {tool.tool_description}
Input Schema (JSON Schema):
```json
{schema_str}
```

GOAL: Create ONE realistic user task that can be SOLVED by calling THIS tool alone.

CRITICAL DATA REQUIREMENTS (adapted from mcp-bench):
1) The task must be self-contained and executable WITHOUT external resources
   - No URLs, no local files, no databases, no “our system”
2) Include ALL concrete inputs needed for this tool’s schema
   - Names/IDs/ranges, thresholds, constraints, etc.
3) Avoid vague references like “user-provided”, “to be determined”, or “based on preferences”
4) ALWAYS USE relative dates/times (e.g., “next 7 days”, “past 3 months”) if dates are needed
5) Do NOT enumerate steps. Do NOT mention tools or schemas. Describe only the user goal.
6) The task must clearly be solvable by THIS tool based on its function and input schema.

Return ONLY this JSON object (no code fences):
{{
  "task_description": "<natural language task description that requires exactly this tool>"
}}
""".strip()


def build_fuzzy_prompt(task_description: str) -> str:
    return f"""
You create natural user requests that require evidence-based responses.

Rewrite the following task as a single natural, conversational user request. Make it sound like a real user asking for help, not a list of steps. Include the necessary details implicitly so an agent could act immediately. Avoid bullet points and numbered steps.

Task:
{task_description}

STYLE (inspired by mcp-bench):
- Conversational tone (first-person OK)
- Prefer concrete, verifiable outputs when appropriate
- No step enumeration, no JSON, no explicit tool hints
- If time is relevant, use relative dates/times (e.g., “next week”, “past 3 months”)

Return ONLY the final request text, nothing else.
""".strip()


# ---------------------- Parsing Helpers ----------------------

TASK_JSON_RE = re.compile(r"\{[^{}]*\"task_description\"[^{}]*\}", re.DOTALL)


def parse_task_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "task_description" in obj:
            return obj
    except Exception:
        pass

    m = TASK_JSON_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "task_description" in obj:
                return obj
        except Exception:
            return None
    return None


# ---------------------- Dataset Logic ------------------------

def build_tool_index(tools_json: List[Dict[str, Any]]) -> Dict[str, List[ToolRecord]]:
    by_server: Dict[str, List[ToolRecord]] = {}
    for t in tools_json:
        sid = t.get("server_id")
        name = t.get("tool_name")
        desc = t.get("tool_description")
        schema = t.get("input_schema")
        sname = t.get("server_name") or sid
        if not (sid and isinstance(name, str) and isinstance(desc, str) and isinstance(schema, dict)):
            continue
        by_server.setdefault(sid, []).append(
            ToolRecord(server_id=sid, server_name=str(sname), tool_name=name, tool_description=desc, input_schema=schema)
        )
    # Deduplicate tools by name per server
    for sid, lst in list(by_server.items()):
        seen = set()
        uniq: List[ToolRecord] = []
        for rec in lst:
            if rec.tool_name not in seen:
                uniq.append(rec)
                seen.add(rec.tool_name)
        by_server[sid] = uniq
    return by_server


def servers_by_bucket(scores_json: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    simple, hard = [], []
    for row in scores_json.get("server_scores", []):
        sid = row.get("server_id")
        lvl = row.get("level")
        if not sid or not isinstance(lvl, str):
            continue
        if lvl.lower() == "simple":
            simple.append(sid)
        elif lvl.lower() == "hard":
            hard.append(sid)
    return simple, hard


def map_server_names(servers_json: List[Dict[str, Any]]) -> Dict[str, str]:
    return {s.get("id"): s.get("name") for s in servers_json if s.get("id")}


def uniform_by_server_sampler(server_ids: List[str], tools_by_server: Dict[str, List[ToolRecord]], total: int, seed: int = 123) -> List[ToolRecord]:
    rnd = random.Random(seed)
    # Filter servers that have valid tools
    servers = [sid for sid in server_ids if tools_by_server.get(sid)]
    if not servers:
        return []
    rnd.shuffle(servers)

    picked: List[ToolRecord] = []
    per_server_iters = 0
    # Round-robin until we reach total or exceed safety iterations
    while len(picked) < total and per_server_iters < total * 10:
        for sid in servers:
            if len(picked) >= total:
                break
            tl = tools_by_server.get(sid, [])
            if not tl:
                continue
            rec = rnd.choice(tl)
            picked.append(rec)
        per_server_iters += 1
    # If still short (few servers), allow sampling with replacement from pooled
    if len(picked) < total:
        pool = [rec for sid in servers for rec in tools_by_server.get(sid, [])]
        while len(picked) < total and pool:
            picked.append(rnd.choice(pool))
    return picked[:total]


def generate_dataset_for_bucket(
    bucket_name: str,
    tool_samples: List[ToolRecord],
    vllm: VLLMClient,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    dataset: List[Dict[str, Any]] = []

    for rec in tool_samples:
        # Stage 1: detailed task
        prompt_task = build_detailed_task_prompt(rec.server_name, rec)
        detailed_raw = vllm.generate(prompt_task)
        parsed = parse_task_json(detailed_raw)
        # Fallback: if parsing fails, craft a minimal task directly from description
        task_description = (
            parsed.get("task_description") if parsed else f"Using the available capability, complete a realistic user goal relevant to: {rec.tool_description}"
        )

        # Stage 2: fuzzy request
        prompt_fuzzy = build_fuzzy_prompt(task_description)
        fuzzy = vllm.generate(prompt_fuzzy)
        if not fuzzy:
            # Simple fallback: use task description directly
            fuzzy = task_description

        dataset.append(
            {
                "request": fuzzy.strip(),
                "target_tool": {
                    "server_id": rec.server_id,
                    "server_name": rec.server_name,
                    "tool_name": rec.tool_name,
                },
                "bucket": bucket_name,
            }
        )

    return dataset


# --------------------------- CLI -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate tool-retrieval datasets for Simple/Hard buckets using vLLM")
    ap.add_argument("--server-scores", default="server_scores.json", help="Path to server_scores.json")
    ap.add_argument("--servers", default="servers_parsed.json", help="Path to servers_parsed.json")
    ap.add_argument("--tools", default="tools_parsed.json", help="Path to tools_parsed.json")
    ap.add_argument("--model", required=True, help="HF repo id or local path for vLLM model (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    ap.add_argument("--out-simple", default="dataset_simple.json", help="Output JSON for Simple bucket")
    ap.add_argument("--out-hard", default="dataset_hard.json", help="Output JSON for Hard bucket")
    ap.add_argument("--count", type=int, default=50, help="Items per bucket (default 50)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    args = ap.parse_args(argv)

    # Load inputs
    scores = load_json(args.server_scores)
    servers = load_json(args.servers)
    tools = load_json(args.tools)

    model = args.model

    # Index tools
    tools_by_server = build_tool_index(tools)
    name_by_id = map_server_names(servers)

    # Split by bucket via scores file
    simple_ids, hard_ids = servers_by_bucket(scores)

    # Sample tools uniformly across servers per bucket
    simple_samples = uniform_by_server_sampler(simple_ids, tools_by_server, args.count, seed=args.seed)
    hard_samples = uniform_by_server_sampler(hard_ids, tools_by_server, args.count, seed=args.seed + 1)

    if len(simple_samples) < args.count:
        print(f"Warning: Only {len(simple_samples)} valid samples available for Simple bucket", file=sys.stderr)
    if len(hard_samples) < args.count:
        print(f"Warning: Only {len(hard_samples)} valid samples available for Hard bucket", file=sys.stderr)

    # Initialize vLLM
    vllm_client = VLLMClient(model=model)

    # Generate datasets
    simple_ds = generate_dataset_for_bucket("Simple", simple_samples, vllm_client, seed=args.seed)
    hard_ds = generate_dataset_for_bucket("Hard", hard_samples, vllm_client, seed=args.seed + 7)

    # Save
    save_json(simple_ds, args.out_simple)
    save_json(hard_ds, args.out_hard)

    print(f"Wrote {len(simple_ds)} items to {args.out_simple}")
    print(f"Wrote {len(hard_ds)} items to {args.out_hard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
