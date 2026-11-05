"""Minimal agent runtime for environments without Google ADK."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable


@dataclass
class EventActions:
    state_delta: dict[str, Any] = field(default_factory=dict)
    transfer_to_agent: str | None = None
    escalate: bool = False

@dataclass
class Event:
    author: str
    content: str = ""
    actions: EventActions = field(default_factory=EventActions)

@dataclass
class Session:
    id: str
    app_name: str
    user_id: str
    state: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

@dataclass
class InvocationContext:
    session: Session
    agent: BaseAgent
    invocation_id: str = "inv-1"

class FunctionTool:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        self.name = getattr(func, "__name__", "tool")
        self.description = getattr(func, "__doc__", "") or ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

class BaseAgent:
    def __init__(
        self,
        name: str,
        description: str = "",
        sub_agents: list[BaseAgent] | None = None,
        **fields: Any,
    ) -> None:
        self.__dict__.update(fields)
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = list(sub_agents or [])

    async def run_async(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async for event in self._run_async_impl(ctx):
            if event.actions and event.actions.state_delta:
                ctx.session.state.update(event.actions.state_delta)
            ctx.session.events.append(event)
            yield event

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raise NotImplementedError
        yield  # pragma: no cover

class LlmAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        model: str = "gemini-2.5-flash",
        description: str = "",
        instruction: str = "",
        tools: list[FunctionTool | Callable[..., Any]] | None = None,
        output_key: str | None = None,
        handler: Callable[[InvocationContext], Any] | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.model = model
        self.instruction = instruction
        self.tools = list(tools or [])
        self.output_key = output_key
        self._handler = handler

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if self._handler is not None:
            result = await self._handler(ctx)
            delta = {self.output_key: result} if self.output_key else {}
            yield Event(
                author=self.name,
                content=str(result),
                actions=EventActions(state_delta=delta),
            )

class SequentialAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        for sub_agent in self.sub_agents:
            sub_ctx = InvocationContext(
                session=ctx.session,
                agent=sub_agent,
                invocation_id=ctx.invocation_id,
            )
            async for event in sub_agent.run_async(sub_ctx):
                yield event

class ParallelAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async def _collect(sub_agent: BaseAgent) -> list[Event]:
            sub_ctx = InvocationContext(
                session=ctx.session,
                agent=sub_agent,
                invocation_id=ctx.invocation_id,
            )
            evs: list[Event] = []
            async for ev in sub_agent.run_async(sub_ctx):
                evs.append(ev)
            return evs

        branches = await asyncio.gather(*(_collect(a) for a in self.sub_agents))
        for branch_events in branches:
            for ev in branch_events:
                yield ev

class InMemorySessionService:
    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, str], Session] = {}

    async def create_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        state: dict[str, Any] | None = None,
    ) -> Session:
        sess = Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=dict(state or {}),
        )
        self._sessions[(app_name, user_id, session_id)] = sess
        return sess

    async def get_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> Session:
        return self._sessions[(app_name, user_id, session_id)]

class Runner:
    def __init__(
        self,
        agent: BaseAgent,
        app_name: str,
        session_service: InMemorySessionService,
    ) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service

    async def run_async(self, user_id: str, session_id: str) -> AsyncGenerator[Event, None]:
        sess = await self.session_service.get_session(self.app_name, user_id, session_id)
        ctx = InvocationContext(session=sess, agent=self.agent)
        async for event in self.agent.run_async(ctx):
            yield event
