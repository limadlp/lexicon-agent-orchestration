"""In-process graph runtime for environments without LangGraph."""

from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


START = "__start__"
END = "__end__"

@dataclass(frozen=True)
class Send:
    node: str
    arg: Any

class MemorySaver:
    def __init__(self) -> None:
        self.storage: dict[str, list[dict[str, Any]]] = {}

    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        self.storage.setdefault(thread_id, []).append(copy.deepcopy(state))

    def get_latest(self, thread_id: str) -> dict[str, Any] | None:
        history = self.storage.get(thread_id, [])
        return copy.deepcopy(history[-1]) if history else None

class _GraphIntrospection:
    def __init__(
        self,
        nodes: list[str],
        edges: list[tuple[str, str]],
        conditional_edges: list[tuple[str, str, list[str]]],
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges

    def draw_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for src, dst in self.edges:
            lines.append(f"    {src} --> {dst}")
        for src, router_name, targets in self.conditional_edges:
            for tgt in targets:
                lines.append(f'    {src} -. "{router_name}" .-> {tgt}')
        return "\n".join(lines)

class _CompiledStateGraph:
    def __init__(
        self,
        state_schema: type,
        nodes: dict[str, Callable[..., Any]],
        edges: dict[str, str],
        conditional_edges: dict[str, tuple[Callable[..., Any], list[str]]],
        checkpointer: MemorySaver | None = None,
    ) -> None:
        self.state_schema = state_schema
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.checkpointer = checkpointer
        self.reducers = self._extract_reducers(state_schema)

    @staticmethod
    def _extract_reducers(schema: type) -> dict[str, Callable[[Any, Any], Any]]:
        reducers: dict[str, Callable[[Any, Any], Any]] = {}
        try:
            hints = get_type_hints(schema, include_extras=True)
        except Exception:
            return reducers
        for field_name, hint in hints.items():
            metadata = getattr(hint, "__metadata__", None)
            if metadata and callable(metadata[0]):
                reducers[field_name] = metadata[0]
        return reducers

    def _merge_update(self, state: dict[str, Any], update: dict[str, Any] | None) -> None:
        if not update:
            return
        for key, value in update.items():
            if key in self.reducers and key in state and state[key] is not None:
                state[key] = self.reducers[key](state[key], value)
            else:
                state[key] = value

    async def _invoke_callable(self, fn: Callable[..., Any], arg: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(arg)
        return fn(arg)

    async def ainvoke(
        self,
        initial_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state: dict[str, Any] = copy.deepcopy(initial_state)
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        max_concurrency = (config or {}).get("max_concurrency")
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None

        async def run_branch(packet: Send) -> Any:
            if semaphore is None:
                return await self._invoke_callable(self.nodes[packet.node], packet.arg)
            async with semaphore:
                return await self._invoke_callable(self.nodes[packet.node], packet.arg)

        current: str | list[Send] = self.edges.get(START, END)
        while current != END:
            if isinstance(current, list):
                sends: list[Send] = current
                if not sends:
                    break
                updates = await asyncio.gather(*(run_branch(packet) for packet in sends))
                for upd in updates:
                    self._merge_update(state, upd)
                if self.checkpointer:
                    self.checkpointer.save(thread_id, state)
                first_node = sends[0].node
                current = self._resolve_next(first_node, state)
            else:
                node_name = current
                node_fn = self.nodes[node_name]
                update = await self._invoke_callable(node_fn, state)
                self._merge_update(state, update)
                if self.checkpointer:
                    self.checkpointer.save(thread_id, state)
                current = self._resolve_next(node_name, state)

        return state

    def invoke(
        self,
        initial_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self.ainvoke(initial_state, config=config))

    def _resolve_next(self, node_name: str, state: dict[str, Any]) -> str | list[Send]:
        if node_name in self.conditional_edges:
            router_fn, _ = self.conditional_edges[node_name]
            return router_fn(state)
        return self.edges.get(node_name, END)

    def get_graph(self) -> _GraphIntrospection:
        edge_list = list(self.edges.items())
        cond_list = [
            (src, fn.__name__, targets)
            for src, (fn, targets) in self.conditional_edges.items()
        ]
        return _GraphIntrospection(list(self.nodes.keys()), edge_list, cond_list)

class StateGraph:
    def __init__(self, state_schema: type) -> None:
        self.state_schema = state_schema
        self._nodes: dict[str, Callable[..., Any]] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, tuple[Callable[..., Any], list[str]]] = {}

    def add_node(self, name: str, action: Callable[..., Any]) -> StateGraph:
        self._nodes[name] = action
        return self

    def add_edge(self, start_key: str, end_key: str) -> StateGraph:
        self._edges[start_key] = end_key
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Any],
        path_map: list[str] | dict[Any, str] | None = None,
    ) -> StateGraph:
        if isinstance(path_map, dict):
            targets = list(path_map.values())
        elif isinstance(path_map, list):
            targets = path_map
        else:
            targets = []
        self._conditional_edges[source] = (path, targets)
        return self

    def compile(self, checkpointer: MemorySaver | None = None) -> _CompiledStateGraph:
        return _CompiledStateGraph(
            state_schema=self.state_schema,
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            checkpointer=checkpointer,
        )
