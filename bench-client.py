"""MCP benchmark client with configurable, vLLM-backed tool retrieval strategies."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import AnyUrl

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError as exc:  # pragma: no cover - vLLM optional import guard
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    VLLM_IMPORT_ERROR = exc
else:
    VLLM_IMPORT_ERROR = None

# Make the local SDK importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_SRC = REPO_ROOT / "python-sdk" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

import mcp
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.session import ProgressFnT
from mcp import types


@dataclass
class ModelConfig:
    """Shared configuration for vLLM model instances."""

    model: str
    max_output_tokens: int = 256
    temperature: float = 0.0
    trust_remote_code: bool = False


@dataclass
class PromptedModelConfig(ModelConfig):
    system_prompt: str | None = None


@dataclass
class ServerConfig:
    """Configuration for a single MCP server connection."""

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass
class ToolRetrievalConfig:
    """Configuration for tool retrieval behaviour."""

    strategy: str = "context_injection"
    top_k: int = 3
    rag: dict[str, Any] = field(default_factory=dict)
    llm: PromptedModelConfig | None = None
    slm: PromptedModelConfig | None = None
    embedding: ModelConfig | None = None
    security_slm: PromptedModelConfig | None = None


@dataclass
class BenchClientConfig:
    """Top-level configuration for the benchmark client."""

    system_prompt: str
    servers: list[ServerConfig]
    tool_retrieval: ToolRetrievalConfig
    component_naming: str = "server_prefixed"


def _parse_model_config(data: dict[str, Any] | None, *, prompted: bool) -> ModelConfig | None:
    if not data:
        return None
    if prompted:
        return PromptedModelConfig(**data)
    return ModelConfig(**data)


def load_config(path: Path) -> BenchClientConfig:
    data = json.loads(path.read_text())

    tool_retrieval_data = data.get("tool_retrieval", {})

    llm_model = _parse_model_config(tool_retrieval_data.get("llm"), prompted=True)
    slm_model = _parse_model_config(tool_retrieval_data.get("slm"), prompted=True)
    embedding_model = _parse_model_config(tool_retrieval_data.get("embedding"), prompted=False)
    security_slm_model = _parse_model_config(tool_retrieval_data.get("security_slm"), prompted=True)

    tool_retrieval = ToolRetrievalConfig(
        strategy=tool_retrieval_data.get("strategy", "context_injection"),
        top_k=int(tool_retrieval_data.get("top_k", 3)),
        rag=tool_retrieval_data.get("rag", {}),
        llm=llm_model,
        slm=slm_model,
        security_slm=security_slm_model,
        embedding=embedding_model,
    )

    servers = [
        ServerConfig(
            name=server_cfg["name"],
            transport=server_cfg.get("transport", "stdio"),
            command=server_cfg.get("command"),
            args=list(server_cfg.get("args", [])),
            env=dict(server_cfg.get("env", {})),
            cwd=server_cfg.get("cwd"),
        )
        for server_cfg in data.get("servers", [])
    ]

    if not servers:
        raise ValueError("At least one server must be configured")

    return BenchClientConfig(
        system_prompt=data.get("system_prompt", ""),
        servers=servers,
        tool_retrieval=tool_retrieval,
        component_naming=data.get("component_naming", "server_prefixed"),
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _text_to_vector(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine_similarity(vec_a: Mapping[str, int], vec_b: Mapping[str, int]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a[token] * vec_b[token] for token in vec_a.keys() & vec_b.keys())
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _dense_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_tool_names(text: str, candidates: set[str], *, k: int) -> list[str]:
    """Parse LLM output for tool names that appear in the candidate set."""
    raw_segments = re.split(r"[\n,]+", text)
    seen: list[str] = []
    for segment in raw_segments:
        cleaned = re.sub(r"^[\s\-\*\d\.)]+", "", segment.strip())
        if not cleaned:
            continue
        if cleaned in candidates and cleaned not in seen:
            seen.append(cleaned)
        else:
            for option in candidates:
                if option.lower() == cleaned.lower() and option not in seen:
                    seen.append(option)
                    break
        if len(seen) >= k:
            break
    return seen


class ToolSelectionError(RuntimeError):
    pass


class VLLMTextGenerator:
    """Thin async wrapper around vLLM text generation."""

    def __init__(self, config: PromptedModelConfig):
        if VLLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "vLLM is required for text generation but is not installed"
            ) from VLLM_IMPORT_ERROR

        assert LLM is not None
        assert SamplingParams is not None

        self._system_prompt = config.system_prompt
        self._llm = LLM(model=config.model, trust_remote_code=config.trust_remote_code)
        self._sampling = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
        )

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()

        def _run() -> str:
            full_prompt = f"{self._system_prompt}\n\n{prompt}" if self._system_prompt else prompt
            outputs = self._llm.generate([full_prompt], sampling_params=self._sampling)
            if not outputs:
                return ""
            first = outputs[0]
            if not first.outputs:
                return ""
            return first.outputs[0].text.strip()

        return await loop.run_in_executor(None, _run)


class VLLMEmbeddingGenerator:
    """Async wrapper around vLLM embedding inference."""

    def __init__(self, config: ModelConfig):
        if VLLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "vLLM is required for embeddings but is not installed"
            ) from VLLM_IMPORT_ERROR
        assert LLM is not None
        self._llm = LLM(model=config.model, trust_remote_code=config.trust_remote_code)
        if not hasattr(self._llm, "get_embedding"):
            raise RuntimeError("The installed vLLM version does not support embeddings")

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()

        def _run() -> list[float]:
            outputs = self._llm.get_embedding([text])  # type: ignore[attr-defined]
            if not outputs:
                return []
            first = outputs[0]
            # vLLM returns either an embedding attribute directly or nested within outputs.
            if hasattr(first, "embedding"):
                return list(first.embedding)  # type: ignore[attr-defined]
            nested = getattr(first, "outputs", None)
            if nested and hasattr(nested[0], "embedding"):
                return list(nested[0].embedding)  # type: ignore[attr-defined]
            raise RuntimeError("Unexpected embedding output format from vLLM")

        return await loop.run_in_executor(None, _run)


@dataclass
class ToolRecord:
    qualified_name: str
    tool: types.Tool
    session: mcp.ClientSession
    server_config: ServerConfig
    server_info: types.Implementation
    original_name: str
    bow_vector: Counter[str]
    embedding: list[float] | None


@dataclass
class ResourceRecord:
    qualified_name: str
    resource: types.Resource
    session: mcp.ClientSession
    server_config: ServerConfig
    server_info: types.Implementation


@dataclass
class ResourceTemplateRecord:
    qualified_name: str
    template: types.ResourceTemplate
    session: mcp.ClientSession
    server_config: ServerConfig
    server_info: types.Implementation


@dataclass
class PromptRecord:
    qualified_name: str
    prompt: types.Prompt
    session: mcp.ClientSession
    server_config: ServerConfig
    server_info: types.Implementation


class BenchMCPClient:
    """Configurable MCP client with vLLM-backed retrieval."""

    def __init__(self, config_path: Path) -> None:
        self.config = load_config(config_path)
        self._exit_stack = AsyncExitStack()
        self._sessions: list[mcp.ClientSession] = []
        self._session_info: dict[mcp.ClientSession, types.Implementation] = {}
        self._session_config: dict[mcp.ClientSession, ServerConfig] = {}
        self._tool_records: dict[str, ToolRecord] = {}
        self._resource_records: dict[str, ResourceRecord] = {}
        self._resource_template_records: dict[str, ResourceTemplateRecord] = {}
        self._resources_by_uri: dict[str, ResourceRecord] = {}
        self._prompt_records: dict[str, PromptRecord] = {}

        self._llm_runner: VLLMTextGenerator | None = (
            VLLMTextGenerator(self.config.tool_retrieval.llm)
            if self.config.tool_retrieval.llm
            else None
        )
        self._slm_runner: VLLMTextGenerator | None = (
            VLLMTextGenerator(self.config.tool_retrieval.slm)
            if self.config.tool_retrieval.slm
            else None
        )
        self._embedding_runner: VLLMEmbeddingGenerator | None = (
            VLLMEmbeddingGenerator(self.config.tool_retrieval.embedding)
            if self.config.tool_retrieval.embedding
            else None
        )
        self._security_runner: VLLMTextGenerator | None = (
            VLLMTextGenerator(self.config.tool_retrieval.security_slm)
            if self.config.tool_retrieval.security_slm
            else None
        )

    async def __aenter__(self) -> BenchMCPClient:
        await self._exit_stack.__aenter__()
        await self._connect_servers()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._exit_stack.aclose()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def _connect_servers(self) -> None:
        for server_cfg in self.config.servers:
            if server_cfg.transport != "stdio":
                raise NotImplementedError(f"Unsupported transport: {server_cfg.transport}")
            if not server_cfg.command:
                raise ValueError(f"Server {server_cfg.name} is missing a command")

            params = StdioServerParameters(
                command=server_cfg.command,
                args=server_cfg.args,
                env=server_cfg.env or None,
                cwd=server_cfg.cwd,
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = await self._exit_stack.enter_async_context(mcp.ClientSession(read, write))
            init_result = await session.initialize()

            self._sessions.append(session)
            self._session_info[session] = init_result.serverInfo
            self._session_config[session] = server_cfg

            await self._bootstrap_components(session, server_cfg, init_result.serverInfo)

    def _qualify_name(self, component_name: str, server_cfg: ServerConfig, server_info: types.Implementation) -> str:
        strategy = self.config.component_naming
        if strategy == "server_prefixed":
            prefix = server_cfg.name or server_info.name or "server"
            return f"{prefix}.{component_name}"
        if strategy == "serverinfo_prefixed":
            prefix = server_info.name or server_cfg.name or "server"
            return f"{prefix}.{component_name}"
        return component_name

    async def _bootstrap_components(
        self,
        session: mcp.ClientSession,
        server_cfg: ServerConfig,
        server_info: types.Implementation,
    ) -> None:
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            qualified = self._qualify_name(tool.name, server_cfg, server_info)
            if qualified in self._tool_records:
                raise ToolSelectionError(f"Duplicate tool name detected: {qualified}")
            tool_copy = tool.model_copy(update={"name": qualified})
            bow = _text_to_vector(tool.description or "")
            embedding: list[float] | None = None
            if self._embedding_runner and tool.description:
                try:
                    vector = await self._embedding_runner.embed(tool.description)
                    embedding = vector or None
                except Exception as exc:  # noqa: PERF203 - continue with fallback
                    print(f"Warning: embedding failed for tool '{qualified}': {exc}")
            record = ToolRecord(
                qualified_name=qualified,
                tool=tool_copy,
                session=session,
                server_config=server_cfg,
                server_info=server_info,
                original_name=tool.name,
                bow_vector=bow,
                embedding=embedding,
            )
            self._tool_records[qualified] = record

        try:
            resources_result = await session.list_resources()
            for resource in resources_result.resources:
                qualified = self._qualify_name(resource.name, server_cfg, server_info)
                resource_copy = resource.model_copy(update={"name": qualified})
                record = ResourceRecord(
                    qualified_name=qualified,
                    resource=resource_copy,
                    session=session,
                    server_config=server_cfg,
                    server_info=server_info,
                )
                self._resource_records[qualified] = record
                self._resources_by_uri[str(resource.uri)] = record
        except Exception:
            pass

        try:
            templates_result = await session.list_resource_templates()
            for template in templates_result.resourceTemplates:
                qualified = self._qualify_name(template.name, server_cfg, server_info)
                template_copy = template.model_copy(update={"name": qualified})
                record = ResourceTemplateRecord(
                    qualified_name=qualified,
                    template=template_copy,
                    session=session,
                    server_config=server_cfg,
                    server_info=server_info,
                )
                self._resource_template_records[qualified] = record
        except Exception:
            pass

        try:
            prompts_result = await session.list_prompts()
            for prompt in prompts_result.prompts:
                qualified = self._qualify_name(prompt.name, server_cfg, server_info)
                prompt_copy = prompt.model_copy(update={"name": qualified})
                record = PromptRecord(
                    qualified_name=qualified,
                    prompt=prompt_copy,
                    session=session,
                    server_config=server_cfg,
                    server_info=server_info,
                )
                self._prompt_records[qualified] = record
        except Exception:
            pass

    # ------------------------------------------------------------------
    # SDK-compatible methods
    # ------------------------------------------------------------------
    async def initialize(self) -> dict[str, types.Implementation]:
        return {
            self._session_config[session].name: self._session_info[session]
            for session in self._sessions
        }

    async def send_ping(self) -> dict[str, types.EmptyResult]:
        return {self._session_config[s].name: await s.send_ping() for s in self._sessions}

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        for session in self._sessions:
            await session.send_progress_notification(progress_token, progress, total=total, message=message)

    async def set_logging_level(self, level: types.LoggingLevel) -> dict[str, types.EmptyResult]:
        return {self._session_config[s].name: await s.set_logging_level(level) for s in self._sessions}

    async def list_resources(self) -> types.ListResourcesResult:
        resources = [record.resource for record in self._resource_records.values()]
        return types.ListResourcesResult(resources=resources, nextCursor=None)

    async def list_resource_templates(self) -> types.ListResourceTemplatesResult:
        templates = [record.template for record in self._resource_template_records.values()]
        return types.ListResourceTemplatesResult(resourceTemplates=templates, nextCursor=None)

    async def read_resource(self, uri: AnyUrl) -> types.ReadResourceResult:
        record = self._resources_by_uri.get(str(uri))
        if record is None:
            raise ToolSelectionError(f"No resource found for URI {uri}")
        return await record.session.read_resource(uri)

    async def subscribe_resource(self, uri: AnyUrl) -> dict[str, types.EmptyResult]:
        record = self._resources_by_uri.get(str(uri))
        if record is None:
            raise ToolSelectionError(f"No resource found for URI {uri}")
        result = await record.session.subscribe_resource(uri)
        return {record.server_config.name: result}

    async def unsubscribe_resource(self, uri: AnyUrl) -> dict[str, types.EmptyResult]:
        record = self._resources_by_uri.get(str(uri))
        if record is None:
            raise ToolSelectionError(f"No resource found for URI {uri}")
        result = await record.session.unsubscribe_resource(uri)
        return {record.server_config.name: result}

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
    ) -> types.CallToolResult:
        record = self._tool_records.get(name)
        if record is None:
            raise ToolSelectionError(f"Unknown tool: {name}")
        return await record.session.call_tool(
            record.original_name,
            arguments=arguments,
            read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
        )

    async def list_prompts(self) -> types.ListPromptsResult:
        prompts = [record.prompt for record in self._prompt_records.values()]
        return types.ListPromptsResult(prompts=prompts, nextCursor=None)

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
        record = self._prompt_records.get(name)
        if record is None:
            raise ToolSelectionError(f"Unknown prompt: {name}")
        return await record.session.get_prompt(record.prompt.name, arguments=arguments)

    async def complete(
        self,
        ref: types.ResourceTemplateReference | types.PromptReference,
        argument: dict[str, str],
        context_arguments: dict[str, str] | None = None,
    ) -> types.CompleteResult:
        last_error: Exception | None = None
        for session in self._sessions:
            try:
                return await session.complete(ref, argument, context_arguments)
            except Exception as exc:  # noqa: PERF203
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ToolSelectionError("No sessions available to complete the request")

    async def list_tools(self) -> types.ListToolsResult:
        tools = [record.tool for record in self._tool_records.values()]
        return types.ListToolsResult(tools=tools, nextCursor=None)

    async def send_roots_list_changed(self) -> None:
        for session in self._sessions:
            await session.send_roots_list_changed()

    # ------------------------------------------------------------------
    # Tool retrieval orchestration
    # ------------------------------------------------------------------
    async def tool_retrieving(
        self,
        request: str,
        k: int,
        strategy: str,
        previous_context: Sequence[str] | None = None,
    ) -> list[str] | str:
        previous_context = previous_context or []
        query = " ".join(filter(None, [request, " ".join(previous_context)]))

        if strategy == "context_injection":
            return self.tool_retrieving_context_injection()
        if strategy == "rag":
            return await self.tool_retrieving_rag(query, k)
        if strategy == "slm":
            return await self.tool_retrieving_slm(query, k)
        raise ToolSelectionError(f"Unknown tool retrieval strategy: {strategy}")

    def tool_retrieving_context_injection(self) -> str:
        lines: list[str] = []
        for record in self._tool_records.values():
            description = record.tool.description or ""
            lines.append(f"[{record.qualified_name}] {description}")
        return "\n".join(lines)

    async def tool_retrieving_rag(self, query: str, k: int) -> list[str]:
        query_vector = _text_to_vector(query)
        query_embedding: list[float] | None = None
        if self._embedding_runner:
            try:
                embedding = await self._embedding_runner.embed(query)
                if embedding:
                    query_embedding = embedding
            except Exception as exc:  # noqa: PERF203
                print(f"Warning: embedding failed for query: {exc}")

        scored: list[tuple[float, ToolRecord]] = []
        for record in self._tool_records.values():
            if query_embedding and record.embedding:
                similarity = _dense_cosine_similarity(query_embedding, record.embedding)
            else:
                similarity = _cosine_similarity(query_vector, record.bow_vector)
            scored.append((similarity, record))
        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored:
            return []

        top_candidates = [record for _, record in scored[: max(1, k * 3)]]
        candidate_names = [record.qualified_name for record in top_candidates]

        if not self._llm_runner:
            return candidate_names[:k]

        tool_context = "\n".join(
            f"- {record.qualified_name}: {record.tool.description or 'No description provided.'}"
            for record in top_candidates
        )

        prompt = (
            "\nTools available:\n"
            f"{tool_context}\n"
            "\nTask: Based on the user request, list the best tool names in descending order of usefulness."
            f"\nUser request: {query}\n"
            f"Return exactly {k} tool names, separated by newlines."
        )

        response = await self._llm_runner.generate(prompt)
        parsed = _extract_tool_names(response, set(candidate_names), k=k)
        if parsed:
            return parsed[:k]
        return candidate_names[:k]

    async def tool_retrieving_slm(self, query: str, k: int) -> list[str]:
        if not self._slm_runner:
            raise ToolSelectionError("SLM strategy requested but no SLM model configured")

        tool_context = "\n".join(
            f"- {record.qualified_name}: {record.tool.description or 'No description provided.'}"
            for record in self._tool_records.values()
        )
        prompt = (
            "\nTools:\n"
            f"{tool_context}\n"
            f"Return exactly {k} tool names that best match the user input, one per line."
            f"\nUser input: {query}\n"
        )

        response = await self._slm_runner.generate(prompt)
        parsed = _extract_tool_names(response, set(self._tool_records.keys()), k=k)
        if parsed:
            return parsed[:k]
        # Fallback to first k tools if parsing fails.
        return list(self._tool_records.keys())[:k]

    async def handle_user_request(
        self,
        request: str,
        previous_context: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        retrieval_cfg = self.config.tool_retrieval
        result = await self.tool_retrieving(
            request=request,
            k=retrieval_cfg.top_k,
            strategy=retrieval_cfg.strategy,
            previous_context=previous_context,
        )
        return {
            "strategy": retrieval_cfg.strategy,
            "result": result,
        }


async def async_main(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    async with BenchMCPClient(config_path) as client:
        previous_context = args.previous_context or []
        selection = await client.handle_user_request(args.request, previous_context=previous_context)
        if isinstance(selection["result"], list):
            print("Selected tools:")
            for name in selection["result"]:
                print(f" - {name}")
        else:
            print("Context injection output:")
            print(selection["result"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MCP client")
    parser.add_argument("--config", required=True, help="Path to client configuration JSON")
    parser.add_argument("--request", required=True, help="User request text")
    parser.add_argument(
        "--previous-context",
        nargs="*",
        default=None,
        help="Optional additional context items maintained for this request",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
