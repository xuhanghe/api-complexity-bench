#!/usr/bin/env python3
"""
Preprocess MCP server/tool data and compute complexity scores.

Features
- Skips servers with zero valid tools.
- Skips tools missing name, description, or input_schema.
- Computes per-server metrics (mu_p, mu_n, mu_d, n_tools) and complexity factor.
- Verifies tool counts vs servers file and reports mismatches.
- Tokenization via vLLM's Llama 3.1 tokenizer (local-only), with optional approximate fallback.

Usage
  python preprocess_servers.py \
    --servers servers_parsed.json \
    --tools tools_parsed.json \
    --out server_scores.json \
    [--threshold 50] \
    [--tokenizer vllm|approx] \
    [--tokenizer-path /path/to/local/llama31/tokenizer]

Notes
- vLLM Llama 3.1 tokenizer requires local tokenizer files. This script first tries
  to load a tokenizer via vLLM utilities (local-only). If it cannot be loaded and
  --tokenizer=vllm is set, the script will error out unless you set --tokenizer=approx
  to use an approximate tokenizer (whitespace + _/- splits).

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import the provided complexity function
try:
    from complexity_expression import complexity_factor
except Exception as e:
    print("Failed to import complexity_expression.complexity_factor:", e, file=sys.stderr)
    sys.exit(1)


class TokenCounter:
    def __init__(self, mode: str = "vllm", tokenizer_path: Optional[str] = None) -> None:
        self.mode = mode
        self.tokenizer = None
        self.backend = None
        self.info = {}

        if mode == "approx":
            self.backend = "approx"
            self.info = {"backend": self.backend}
            return

        # vLLM tokenizer load (local-only)
        model_path = (
            tokenizer_path
            or os.environ.get("VLLM_TOKENIZER")
            or os.environ.get("LLAMA_TOKENIZER")
            or "meta-llama/Meta-Llama-3.1-8B"
        )

        vllm_errs: List[str] = []
        try:
            from vllm.transformers_utils import tokenizer as vtok_mod  # type: ignore
            # Try common factory names across vLLM versions
            factory_candidates = []
            if hasattr(vtok_mod, "get_tokenizer"):
                factory_candidates.append(("get_tokenizer", getattr(vtok_mod, "get_tokenizer")))
            if hasattr(vtok_mod, "load_tokenizer"):
                factory_candidates.append(("load_tokenizer", getattr(vtok_mod, "load_tokenizer")))
            if hasattr(vtok_mod, "Tokenizer"):
                factory_candidates.append(("Tokenizer", getattr(vtok_mod, "Tokenizer")))

            last_exc: Optional[Exception] = None
            for name, fn in factory_candidates:
                try:
                    if name == "Tokenizer":
                        try:
                            self.tokenizer = fn(model_path, tokenizer_mode="auto", trust_remote_code=True)  # type: ignore
                        except TypeError:
                            self.tokenizer = fn(model_path)  # type: ignore
                    else:
                        try:
                            self.tokenizer = fn(model_path, tokenizer_mode="auto", trust_remote_code=True)  # type: ignore
                        except TypeError:
                            self.tokenizer = fn(model_path)  # type: ignore
                    self.backend = "vllm"
                    self.info = {"backend": self.backend, "model": model_path, "factory": name}
                    break
                except Exception as e:
                    last_exc = e
                    vllm_errs.append(f"{name}: {e}")
            if self.tokenizer is not None:
                return
            if last_exc is not None:
                raise last_exc
        except Exception as e:
            vllm_errs.append(str(e))

        # Fallback to HF tokenizer locally if vLLM path fails
        try:
            from transformers import AutoTokenizer  # type: ignore
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.backend = "transformers_fallback"
            self.info = {
                "backend": self.backend,
                "model": model_path,
                "note": "vLLM tokenizer load failed; using HF fallback",
                "vllm_errors": vllm_errs,
            }
            return
        except Exception:
            pass

        raise RuntimeError(
            "vLLM tokenizer not found locally. Provide --tokenizer-path or set VLLM_TOKENIZER/LLAMA_TOKENIZER "
            "to a local directory, or use --tokenizer=approx.\n"
            f"Attempted model_path={model_path!r}; vLLM errors: {vllm_errs}"
        )

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.backend == "approx":
            tmp = text.replace("_", " ").replace("-", " ")
            return len(tmp.split())
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))  # type: ignore
        except Exception:
            try:
                return len(self.tokenizer.tokenize(text))  # type: ignore
            except Exception:
                tmp = text.replace("_", " ").replace("-", " ")
                return len(tmp.split())


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def param_count_from_schema(input_schema: Dict[str, Any]) -> int:
    """
    Count the number of top-level parameters defined in the input schema.
    Following the user's guidance: count all parameters that should be provided
    by the user; operationalized here as the number of keys under top-level
    `properties`.
    """
    if not isinstance(input_schema, dict):
        return 0
    props = input_schema.get("properties")
    if isinstance(props, dict):
        return len(props)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Preprocess servers and compute complexity scores.")
    p.add_argument("--servers", required=True, help="Path to servers_parsed.json")
    p.add_argument("--tools", required=True, help="Path to tools_parsed.json")
    p.add_argument("--out", required=True, help="Path to output scores JSON")
    p.add_argument("--threshold", type=int, default=50, help="Threshold for level labeling (default 50)")
    p.add_argument("--tokenizer", choices=["vllm", "approx"], default="vllm",
                   help="Tokenization mode: vllm (requires local tokenizer) or approx")
    p.add_argument("--tokenizer-path", default=None, help="Local path to Llama 3.1 tokenizer directory (HF format)")
    args = p.parse_args(argv)

    # Initialize tokenizer
    try:
        tok = TokenCounter(mode=args.tokenizer, tokenizer_path=args.tokenizer_path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    servers = load_json(args.servers)
    tools = load_json(args.tools)

    # Index servers by id
    servers_by_id: Dict[str, Dict[str, Any]] = {}
    for s in servers:
        sid = s.get("id")
        if sid:
            servers_by_id[sid] = s

    # Build tool groupings and apply tool-level filter
    tools_by_server: Dict[str, List[Dict[str, Any]]] = {}
    total_tools_raw_by_server: Dict[str, int] = {}
    invalid_tools = 0
    for t in tools:
        sid = t.get("server_id")
        if not sid:
            invalid_tools += 1
            continue

        total_tools_raw_by_server[sid] = total_tools_raw_by_server.get(sid, 0) + 1

        name = t.get("tool_name")
        desc = t.get("tool_description")
        schema = t.get("input_schema")
        if not (name and isinstance(name, str)):
            invalid_tools += 1
            continue
        if not (desc and isinstance(desc, str)):
            invalid_tools += 1
            continue
        if not (schema and isinstance(schema, dict)):
            invalid_tools += 1
            continue

        tools_by_server.setdefault(sid, []).append(t)

    # Compute per-server metrics and factor
    results: List[Dict[str, Any]] = []
    skipped_servers: List[Dict[str, Any]] = []

    for sid, s in servers_by_id.items():
        all_tools_raw = total_tools_raw_by_server.get(sid, 0)
        valid_tools_for_server = tools_by_server.get(sid, [])

        if len(valid_tools_for_server) == 0:
            skipped_servers.append({
                "server_id": sid,
                "server_name": s.get("name"),
                "reason": "zero_valid_tools"
            })
            continue

        # Metrics over valid tools
        param_counts: List[int] = []
        name_token_counts: List[int] = []
        desc_token_counts: List[int] = []

        for t in valid_tools_for_server:
            schema = t.get("input_schema", {})
            param_counts.append(param_count_from_schema(schema))
            name_token_counts.append(tok.count(t.get("tool_name", "")))
            desc_token_counts.append(tok.count(t.get("tool_description", "")))

        mu_p = (sum(param_counts) / len(param_counts)) if param_counts else 0.0
        mu_n = (sum(name_token_counts) / len(name_token_counts)) if name_token_counts else 0.0
        mu_d = (sum(desc_token_counts) / len(desc_token_counts)) if desc_token_counts else 0.0

        n_tools = len(valid_tools_for_server)

        # Output schema presence unknown in provided data; assume False for now.
        has_output_schema = False

        factor, level = complexity_factor(
            mu_p=mu_p,
            mu_n=mu_n,
            mu_d=mu_d,
            n_tools=n_tools,
            has_output_schema=has_output_schema,
            threshold=args.threshold,
        )

        reported_tools_count = s.get("tools_count")
        mismatch = (reported_tools_count is not None) and (int(reported_tools_count) != int(all_tools_raw))

        results.append({
            "server_id": sid,
            "server_name": s.get("name"),
            "factor": factor,
            "level": level,
            "threshold": args.threshold,
            "mu_p": mu_p,
            "mu_n": mu_n,
            "mu_d": mu_d,
            "n_tools": n_tools,
            "has_output_schema": has_output_schema,
            "tokenizer": tok.info,
            "tools_count_reported": reported_tools_count,
            "tools_count_actual": all_tools_raw,
            "tools_count_valid": n_tools,
            "tools_mismatch": bool(mismatch),
        })

    # Sort by factor descending for easier inspection
    results.sort(key=lambda x: x["factor"], reverse=True)

    out = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "servers_file": os.path.abspath(args.servers),
            "tools_file": os.path.abspath(args.tools),
            "invalid_tools_skipped": invalid_tools,
            "skipped_servers_count": len(skipped_servers),
        },
        "server_scores": results,
        "skipped_servers": skipped_servers,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote scores for {len(results)} servers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
