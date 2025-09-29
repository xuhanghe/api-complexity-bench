#!/usr/bin/env python3
"""
Run a needle-in-the-haystack tool retrieval experiment using vLLM.

Inputs:
- One or more dataset files produced by generate_experiment_dataset.py (JSON arrays)
- tools_parsed.json for tool metadata (name/description/input_schema)
- server_scores.json to identify Simple/Hard buckets by server_name

Behavior:
- For each dataset item (request + target_tool), build a haystack of tools from
  servers in the same bucket (Simple/Hard) including the target server and a
  sampled set of distraction servers.
- Show ONLY the tool list (names/desc/schema) + the request to the model.
- Ask model to return top-K tool choices as JSON. Evaluate success if the target
  appears in top-K. Report success rates per bucket and per model group (SLM/LLM).

Note: This is a retrieval-only harness; it does not call tools. It focuses on
the selection step under controlled haystack conditions as discussed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tools_by_server(tools: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_sid: Dict[str, List[Dict[str, Any]]] = {}
    for t in tools:
        sid = t.get("server_id")
        if not sid:
            continue
        name = t.get("tool_name")
        desc = t.get("tool_description")
        schema = t.get("input_schema")
        sname = t.get("server_name") or sid
        if not (isinstance(name, str) and isinstance(desc, str) and isinstance(schema, dict)):
            continue
        by_sid.setdefault(sid, []).append(t)
    # Dedup by tool_name per server
    for sid, lst in list(by_sid.items()):
        seen = set()
        uniq = []
        for t in lst:
            n = t.get("tool_name")
            if n in seen:
                continue
            uniq.append(t)
            seen.add(n)
        by_sid[sid] = uniq
    return by_sid


def server_bucket_by_name(scores: Dict[str, Any]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for row in scores.get("server_scores", []):
        name = row.get("server_name")
        lvl = row.get("level")
        if isinstance(name, str) and isinstance(lvl, str):
            m[name] = lvl.capitalize()
    return m


def server_name_by_id(servers: List[Dict[str, Any]]) -> Dict[str, str]:
    return {s.get("id"): s.get("name") for s in servers if s.get("id") and s.get("name")}


def format_tools_for_prompt(tool_list: List[Dict[str, Any]], max_schema_chars: int) -> str:
    chunks = []
    for t in tool_list:
        sname = t.get("server_name")
        tname = t.get("tool_name")
        desc = t.get("tool_description") or ""
        schema = t.get("input_schema") or {}
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        if len(schema_str) > max_schema_chars:
            schema_str = schema_str[:max_schema_chars] + "\n... (truncated)"
        full_key = f"{sname}:{tname}"
        chunks.append(
            f"Tool: `{full_key}` (Server: {sname})\n  Description: {desc}\n  Input Schema:\n```json\n{schema_str}\n```\n"
        )
    return "\n".join(chunks)


def build_selection_prompt(request: str, tool_block: str, top_k: int) -> str:
    return f"""
You are given a list of available tools and a user request.
Select the MOST appropriate tool(s) that best match the user's request.

Available tools:
{tool_block}

User request:
{request}

Rules:
- Return exactly {top_k} tools in order of preference as JSON:
  {{"selected_tools": ["Server A:tool_x", "Server B:tool_y", ...]}}
- If fewer than {top_k} are suitable, return as many as you can (non-empty).
- Do NOT include any other fields or commentary.
""".strip()


class VLLMModel:
    def __init__(self, model_id: str):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError("vLLM is required. Please install vllm.") from e
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model_id, trust_remote_code=True)

    def complete(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.95) -> str:
        params = self.SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        out = self.llm.generate([prompt], params)
        if not out or not out[0].outputs:
            return ""
        return out[0].outputs[0].text.strip()


def parse_selected_tools(text: str) -> List[str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("selected_tools"), list):
            return [str(x) for x in obj.get("selected_tools")][:50]
    except Exception:
        pass
    # Try to find JSON snippet
    import re
    m = re.search(r"\{\s*\"selected_tools\"\s*:\s*\[(.*?)\]", text, re.DOTALL)
    if m:
        arr = m.group(1)
        # naive split by quotes
        tools = re.findall(r'\"([^\"]+)\"', arr)
        return tools[:50]
    return []


def sample_haystack(
    rnd: random.Random,
    bucket: str,
    target_sid: str,
    tools_by_sid: Dict[str, List[Dict[str, Any]]],
    sid_to_name: Dict[str, str],
    name_to_bucket: Dict[str, str],
    distraction_servers: int,
    tools_cap_per_server: int,
    target_total_tools: int,
    target_server_name: str,
    target_tool_name: str,
) -> List[Dict[str, Any]]:
    # Identify candidate server_ids of same bucket
    # Map server_id -> server_name, then filter by bucket match
    same_bucket_sids = []
    for sid, tlist in tools_by_sid.items():
        sname = sid_to_name.get(sid, sid)
        if name_to_bucket.get(sname) == bucket:
            same_bucket_sids.append(sid)
    # Exclude target server from distractors
    distractor_pool = [sid for sid in same_bucket_sids if sid != target_sid]
    rnd.shuffle(distractor_pool)
    selected = [target_sid] + distractor_pool[:max(0, distraction_servers)]

    # Gather tools, optionally cap per server
    haystack = []
    for sid in selected:
        tlist = tools_by_sid.get(sid, [])
        if tools_cap_per_server and len(tlist) > tools_cap_per_server:
            tlist = rnd.sample(tlist, tools_cap_per_server)
        # Normalize server_name presence
        sname = sid_to_name.get(sid, sid)
        for t in tlist:
            t = dict(t)
            t["server_name"] = sname
            haystack.append(t)

    # Ensure target tool is present
    def has_target(lst: List[Dict[str, Any]]) -> bool:
        for h in lst:
            if h.get("server_name") == target_server_name and h.get("tool_name") == target_tool_name:
                return True
        return False

    if not has_target(haystack):
        # Try inject exact tool from its server
        for t in tools_by_sid.get(target_sid, []):
            if t.get("tool_name") == target_tool_name:
                inj = dict(t)
                inj["server_name"] = target_server_name
                haystack.append(inj)
                break

    # Deduplicate by server:tool key
    dedup = {}
    for h in haystack:
        key = f"{h.get('server_name')}:{h.get('tool_name')}"
        dedup[key] = h
    haystack = list(dedup.values())

    # If too few tools, pull more from same-bucket servers
    if len(haystack) < target_total_tools:
        # Build pool from all same-bucket servers (including those not selected)
        pool = []
        for sid, tlist in tools_by_sid.items():
            sname = sid_to_name.get(sid, sid)
            if name_to_bucket.get(sname) != bucket:
                continue
            for t in tlist:
                x = dict(t)
                x["server_name"] = sname
                pool.append(x)
        # Remove duplicates already in haystack
        existing = {f"{h.get('server_name')}:{h.get('tool_name')}" for h in haystack}
        candidates = [p for p in pool if f"{p.get('server_name')}:{p.get('tool_name')}" not in existing]
        rnd.shuffle(candidates)
        # Take until target_total_tools
        to_take = target_total_tools - len(haystack)
        haystack.extend(candidates[:max(0, to_take)])

    # If still short, sample with replacement from current haystack
    while len(haystack) < target_total_tools and haystack:
        haystack.append(rnd.choice(haystack))

    # If too many tools, downsample but keep target
    if len(haystack) > target_total_tools:
        # Ensure target kept
        target_key = f"{target_server_name}:{target_tool_name}"
        rnd.shuffle(haystack)
        # Move target to front
        haystack.sort(key=lambda h: 0 if f"{h.get('server_name')}:{h.get('tool_name')}" == target_key else 1)
        keep = [haystack[0]]  # target first
        remaining = haystack[1:]
        keep.extend(remaining[:max(0, target_total_tools - 1)])
        haystack = keep

    # Final guarantee target presence
    if not has_target(haystack) and tools_by_sid.get(target_sid):
        inj = dict(tools_by_sid[target_sid][0])
        inj["server_name"] = target_server_name
        if haystack:
            haystack[0] = inj
        else:
            haystack.append(inj)

    return haystack[:target_total_tools]


def sample_haystack_fixed_tools(
    rnd: random.Random,
    bucket: str,
    target_sid: str,
    target_server_name: str,
    target_tool_name: str,
    tools_by_sid: Dict[str, List[Dict[str, Any]]],
    sid_to_name: Dict[str, str],
    name_to_bucket: Dict[str, str],
    total_tools: int,
) -> List[Dict[str, Any]]:
    """Sample a fixed number of tools from the same bucket, ensuring target is included.

    - Build a pool of all tools from servers in the same bucket.
    - Exclude duplicate server:tool pairs.
    - Ensure the target tool is present, then sample the rest uniformly.
    """
    # Build pool of same-bucket tools
    pool: List[Dict[str, Any]] = []
    seen_keys = set()
    for sid, tlist in tools_by_sid.items():
        sname = sid_to_name.get(sid, sid)
        if name_to_bucket.get(sname) != bucket:
            continue
        for t in tlist:
            key = f"{sname}:{t.get('tool_name')}"
            if key in seen_keys:
                continue
            x = dict(t)
            x["server_name"] = sname
            pool.append(x)
            seen_keys.add(key)

    # Identify target tool
    target_key = f"{target_server_name}:{target_tool_name}"
    target_item = None
    for x in pool:
        if f"{x.get('server_name')}:{x.get('tool_name')}" == target_key:
            target_item = x
            break

    # If target not in pool, attempt to construct from tools_by_sid
    if target_item is None:
        for t in tools_by_sid.get(target_sid, []):
            if t.get("tool_name") == target_tool_name:
                target_item = dict(t)
                target_item["server_name"] = target_server_name
                break
    if target_item is None:
        return []  # cannot proceed if target missing

    # Remove target from pool to avoid double counting
    pool = [p for p in pool if f"{p.get('server_name')}:{p.get('tool_name')}" != target_key]
    rnd.shuffle(pool)

    # Assemble haystack: target + (N-1) random tools
    n_other = max(0, total_tools - 1)
    others = pool[:n_other] if len(pool) >= n_other else pool
    hay = [target_item] + others

    # If still short, pad by sampling with replacement from pool (if any)
    while len(hay) < total_tools and pool:
        hay.append(rnd.choice(pool))

    # Truncate to exact size
    return hay[:total_tools]


def run_experiment(
    dataset: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    server_scores: Dict[str, Any],
    servers: List[Dict[str, Any]],
    model_slm: Optional[str],
    model_llm: Optional[str],
    distraction_servers: int,
    tools_cap_per_server: int,
    haystack_size: Optional[int],
    top_k: int,
    max_items_per_bucket: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    rnd = random.Random(seed)
    name_to_bucket = server_bucket_by_name(server_scores)
    sid_to_name = server_name_by_id(servers)
    tools_by_sid = tools_by_server(tools)

    # Split dataset by bucket
    buckets = {"Simple": [], "Hard": []}
    for item in dataset:
        b = item.get("bucket")
        if b in buckets:
            buckets[b].append(item)

    # Limit items per bucket if requested
    if max_items_per_bucket:
        for b in buckets:
            rnd.shuffle(buckets[b])
            buckets[b] = buckets[b][:max_items_per_bucket]

    # Precompute per-item haystacks so all models see the exact same haystack
    # Determine target total tools per item
    if haystack_size and haystack_size > 0:
        target_total_tools = haystack_size
    else:
        target_total_tools = (1 + max(0, distraction_servers)) * (tools_cap_per_server if tools_cap_per_server > 0 else 30)
    precomputed: Dict[str, Dict[str, Any]] = {"Simple": {}, "Hard": {}}

    for bucket_name, items in buckets.items():
        for idx, item in enumerate(items):
            req = item.get("request", "").strip()
            tgt = item.get("target_tool", {})
            tgt_server = tgt.get("server_name")
            tgt_tool = tgt.get("tool_name")
            # Map server_name to server_id
            tgt_sid = None
            for sid, sname in sid_to_name.items():
                if sname == tgt_server:
                    tgt_sid = sid
                    break
            if not (req and tgt_server and tgt_tool and tgt_sid):
                continue

            if haystack_size and haystack_size > 0:
                hay = sample_haystack_fixed_tools(
                    rnd,
                    bucket_name,
                    tgt_sid,
                    tgt_server,
                    tgt_tool,
                    tools_by_sid,
                    sid_to_name,
                    name_to_bucket,
                    target_total_tools,
                )
            else:
                hay = sample_haystack(
                    rnd,
                    bucket_name,
                    tgt_sid,
                    tools_by_sid,
                    sid_to_name,
                    name_to_bucket,
                    distraction_servers,
                    tools_cap_per_server,
                    target_total_tools,
                    tgt_server,
                    tgt_tool,
                )
            precomputed[bucket_name][idx] = {"request": req, "target_key": f"{tgt_server}:{tgt_tool}", "hay": hay}

    # Prepare models
    results = {"summary": {}, "details": {}}
    model_specs = [("SLM", model_slm), ("LLM", model_llm)]
    active_models = [(label, mid) for (label, mid) in model_specs if mid]
    if not active_models:
        raise RuntimeError("No models provided. Use --model-slm and/or --model-llm.")

    for label, mid in active_models:
        model = VLLMModel(mid)
        label_key = f"model:{label}:{mid}"
        results["details"][label_key] = {}

        for bucket_name, items in buckets.items():
            successes_topk = 0
            total = 0
            per_item = []

            for idx, _item in enumerate(items):
                pc = precomputed[bucket_name].get(idx)
                if not pc:
                    continue
                req = pc["request"]
                hay = pc["hay"]
                target_key = pc["target_key"]

                tool_block = format_tools_for_prompt(hay, max_schema_chars=4000)
                prompt = build_selection_prompt(req, tool_block, top_k)
                out = model.complete(prompt)
                preds = parse_selected_tools(out)
                total += 1

                # Normalize predictions and target
                norm_preds = [p.strip() for p in preds]
                hit_topk = any(p == target_key for p in norm_preds[:top_k])
                if hit_topk:
                    successes_topk += 1

                per_item.append({
                    "bucket": bucket_name,
                    "request": req,
                    "target_tool": target_key,
                    "predictions": norm_preds,
                    "hit_topk": hit_topk,
                    "haystack_server_count": len({h.get('server_name') for h in hay}),
                    "haystack_tool_count": len(hay)
                })

            rate = successes_topk / total if total else 0.0
            results["details"][label_key][bucket_name] = {
                "success_rate_topk": rate,
                "total": total,
                "successes": successes_topk,
                "items": per_item,
            }

        # Explicitly release the model before moving to the next one to avoid
        # concurrent residency in memory/VRAM. This ensures strictly sequential
        # evaluation across models and clean context between runs.
        try:
            # Drop strong refs to encourage immediate teardown
            del model
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        # Best-effort GPU memory cleanup when using CUDA
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Build summary
    for label_key, by_bucket in results["details"].items():
        for bname, info in by_bucket.items():
            results["summary"].setdefault(bname, {})[label_key] = {
                "success_rate_topk": info["success_rate_topk"],
                "total": info["total"],
                "successes": info["successes"],
            }

    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run tool retrieval experiment with vLLM")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument(
        "--dataset",
        help=(
            "Comma-separated dataset JSON files. "
            "For example: dataset_simple.json,dataset_hard.json or "
            "dataset_simple_cleaned.json,dataset_hard_cleaned.json"
        ),
    )
    mx.add_argument(
        "--use-cleaned",
        action="store_true",
        help=(
            "Use the cleaned datasets in the current directory: "
            "dataset_simple_cleaned.json and dataset_hard_cleaned.json"
        ),
    )
    ap.add_argument("--tools", default="tools_parsed.json", help="Path to tools_parsed.json")
    ap.add_argument("--server-scores", default="server_scores.json", help="Path to server_scores.json (for bucket mapping)")
    ap.add_argument("--servers", default="servers_parsed.json", help="Path to servers_parsed.json (for id->name mapping)")
    # One or both models can be provided; they are run sequentially (never concurrently)
    ap.add_argument("--model-slm", help="HF repo id for SLM model (optional)")
    ap.add_argument("--model-llm", help="HF repo id for LLM model (optional)")
    ap.add_argument("--distraction-servers", type=int, default=10, help="Number of distraction servers from same bucket per item")
    ap.add_argument("--tools-cap-per-server", type=int, default=30, help="Max tools per server included in haystack (0 = no cap)")
    ap.add_argument("--topk", type=int, default=1, help="Evaluate success if target appears in top-K (default 1)")
    ap.add_argument("--haystack-size", type=int, default=None, help="Fixed total number of tools per item (overrides server/distraction config if set)")
    ap.add_argument("--max-items-per-bucket", type=int, default=None, help="Optional limit for items per bucket")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--out", default="retrieval_results.json", help="Output results JSON path")
    args = ap.parse_args(argv)

    # Load dataset files
    if args.use_cleaned:
        dataset_files = [
            "dataset_simple_cleaned.json",
            "dataset_hard_cleaned.json",
        ]
    else:
        dataset_files = [p.strip() for p in (args.dataset or "").split(',') if p.strip()]
        if not dataset_files:
            print("No dataset files specified", file=sys.stderr)
            return 2
    dataset = []
    for p in dataset_files:
        dataset.extend(load_json(p))

    tools = load_json(args.tools)
    scores = load_json(args.server_scores)

    if not args.model_slm and not args.model_llm:
        print("Error: No model provided. Use --model-slm and/or --model-llm.", file=sys.stderr)
        return 2

    results = run_experiment(
        dataset=dataset,
        tools=tools,
        server_scores=scores,
        servers=load_json(args.servers),
        model_slm=args.model_slm,
        model_llm=args.model_llm,
        distraction_servers=args.distraction_servers,
        tools_cap_per_server=args.tools_cap_per_server,
        top_k=args.topk,
        haystack_size=args.haystack_size,
        max_items_per_bucket=args.max_items_per_bucket,
        seed=args.seed,
    )

    save_json(results, args.out)
    # Print concise summary
    print("Summary (success_rate_topk by bucket and model):")
    for bucket, models in results.get("summary", {}).items():
        print(f"  {bucket}:")
        for mlabel, info in models.items():
            rate = info.get("success_rate_topk", 0.0)
            total = info.get("total", 0)
            suc = info.get("successes", 0)
            print(f"    {mlabel}: {rate:.3f} ({suc}/{total})")
    print(f"Results saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
