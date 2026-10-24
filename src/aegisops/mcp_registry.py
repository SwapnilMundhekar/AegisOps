"""Governed MCP capability registration for the AegisOps control plane."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from aegisops.domain import (
    ActionType,
    ProposedAction,
    RiskLevel,
)
from aegisops.mcp_trust import (
    MCPClientLike,
    MCPTrustGateway,
)
from aegisops.tool_runtime import ToolRuntime


class MCPCapabilityRegistryError(RuntimeError):
    """Base error raised by the governed MCP capability registry."""


class MCPBindingNotFoundError(MCPCapabilityRegistryError):
    """Raised when an MCP tool has no explicit AegisOps binding."""


class MCPBindingConflictError(MCPCapabilityRegistryError):
    """Raised when two bindings collide or contradict one another."""


class MCPServerAlreadyActivatedError(MCPCapabilityRegistryError):
    """Raised when a server is activated twice in the same registry."""


class MCPRuntimeToolCollisionError(MCPCapabilityRegistryError):
    """Raised when an MCP capability would replace another runtime tool."""


@dataclass(frozen=True, slots=True)
class MCPCapabilityBinding:
    """Authoritative AegisOps policy for one MCP tool.

    MCP-provided annotations are deliberately not consulted here.

    This application-owned binding determines the security and
    execution contract regardless of what the remote MCP server claims
    through readOnlyHint, destructiveHint, idempotentHint, or related
    metadata.
    """

    server_id: str
    tool_name: str

    action_type: ActionType
    risk: RiskLevel

    idempotent: bool

    requires_approval: bool = False

    timeout_seconds: float | None = 30.0

    runtime_name: str | None = None

    idempotency_argument: str | None = None

    description: str | None = None

    def __post_init__(
        self,
    ) -> None:
        server_id = self.server_id.strip()
        tool_name = self.tool_name.strip()

        if not server_id:
            raise ValueError(
                "server_id must not be empty."
            )

        if not tool_name:
            raise ValueError(
                "tool_name must not be empty."
            )

        object.__setattr__(
            self,
            "server_id",
            server_id,
        )

        object.__setattr__(
            self,
            "tool_name",
            tool_name,
        )

        runtime_name = self.runtime_name

        if runtime_name is None:
            runtime_name = (
                f"mcp.{server_id}.{tool_name}"
            )

        runtime_name = runtime_name.strip()

        if not runtime_name:
            raise ValueError(
                "runtime_name must not be empty."
            )

        object.__setattr__(
            self,
            "runtime_name",
            runtime_name,
        )

        if (
            self.timeout_seconds is not None
            and self.timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be greater than zero "
                "or None."
            )

        idempotency_argument = (
            self.idempotency_argument
        )

        if idempotency_argument is not None:
            idempotency_argument = (
                idempotency_argument.strip()
            )

            if not idempotency_argument:
                raise ValueError(
                    "idempotency_argument must not be empty."
                )

            if not self.idempotent:
                raise ValueError(
                    "idempotency_argument requires "
                    "idempotent=True."
                )

            object.__setattr__(
                self,
                "idempotency_argument",
                idempotency_argument,
            )

        description = self.description

        if description is not None:
            description = description.strip()

            if not description:
                description = None

            object.__setattr__(
                self,
                "description",
                description,
            )

        self._validate_security_invariants()

    def _validate_security_invariants(
        self,
    ) -> None:
        """Reject obviously unsafe capability declarations."""

        if (
            self.action_type
            in {
                ActionType.DEPLOY,
                ActionType.ROLLBACK,
            }
            and not self.requires_approval
        ):
            raise ValueError(
                "DEPLOY and ROLLBACK MCP capabilities must "
                "explicitly require human approval."
            )

        if (
            self.risk
            in {
                RiskLevel.HIGH,
                RiskLevel.CRITICAL,
            }
            and self.action_type
            is not ActionType.READ
            and not self.requires_approval
        ):
            raise ValueError(
                "HIGH or CRITICAL mutating MCP capabilities "
                "must explicitly require human approval."
            )

        if (
            self.action_type
            is ActionType.READ
            and self.risk
            is RiskLevel.CRITICAL
        ):
            raise ValueError(
                "READ capabilities must not be classified "
                "as CRITICAL. Use EXECUTE or a mutating action "
                "type when the operation has critical effects."
            )


@dataclass(frozen=True, slots=True)
class MCPRegisteredCapability:
    """Runtime view of one activated MCP capability."""

    server_id: str
    tool_name: str
    runtime_name: str

    action_type: ActionType
    risk: RiskLevel

    idempotent: bool
    requires_approval: bool

    timeout_seconds: float | None
    idempotency_argument: str | None

    description: str | None


class MCPCapabilityRegistry:
    """Bind validated MCP tools into governed AegisOps capabilities."""

    def __init__(
        self,
        *,
        gateway: MCPTrustGateway,
        tool_runtime: ToolRuntime,
        bindings: tuple[
            MCPCapabilityBinding,
            ...,
        ],
    ) -> None:
        if not bindings:
            raise ValueError(
                "At least one MCP capability binding is required."
            )

        self._gateway = gateway
        self._tool_runtime = tool_runtime

        self._bindings: dict[
            tuple[str, str],
            MCPCapabilityBinding,
        ] = {}

        self._runtime_names: dict[
            str,
            tuple[str, str],
        ] = {}

        self._active: dict[
            tuple[str, str],
            MCPRegisteredCapability,
        ] = {}

        self._active_servers: set[
            str
        ] = set()

        for binding in bindings:
            identity = (
                binding.server_id,
                binding.tool_name,
            )

            if identity in self._bindings:
                raise MCPBindingConflictError(
                    "Duplicate MCP capability binding: "
                    f"{binding.server_id}/{binding.tool_name}"
                )

            assert binding.runtime_name is not None

            normalized_runtime_name = (
                binding.runtime_name.casefold()
            )

            existing_identity = self._runtime_names.get(
                normalized_runtime_name
            )

            if existing_identity is not None:
                raise MCPBindingConflictError(
                    "Two MCP capabilities use the same "
                    f"runtime_name {binding.runtime_name!r}: "
                    f"{existing_identity[0]}/"
                    f"{existing_identity[1]} and "
                    f"{binding.server_id}/"
                    f"{binding.tool_name}."
                )

            self._bindings[
                identity
            ] = binding

            self._runtime_names[
                normalized_runtime_name
            ] = identity

    @property
    def bindings(
        self,
    ) -> Mapping[
        tuple[str, str],
        MCPCapabilityBinding,
    ]:
        """Return an immutable view of configured bindings."""

        return MappingProxyType(
            self._bindings
        )

    @property
    def active_capabilities(
        self,
    ) -> tuple[
        MCPRegisteredCapability,
        ...,
    ]:
        """Return activated capabilities in deterministic order."""

        return tuple(
            sorted(
                self._active.values(),
                key=lambda capability: (
                    capability.server_id,
                    capability.tool_name,
                ),
            )
        )

    async def activate_server(
        self,
        *,
        client: MCPClientLike,
        server_id: str,
    ) -> tuple[
        MCPRegisteredCapability,
        ...,
    ]:
        """Discover and register governed capabilities for one server.

        Activation is deliberately one-shot per registry instance.

        Reconnecting an MCP client should create a new registry/runtime
        binding lifecycle rather than silently replacing handlers that
        may still be executing.
        """

        if server_id in self._active_servers:
            raise MCPServerAlreadyActivatedError(
                f"MCP server is already activated: {server_id}"
            )

        advertised = await self._gateway.refresh_server(
            client=client,
            server_id=server_id,
        )

        server_bindings = {
            tool_name: binding
            for (
                binding_server,
                tool_name,
            ), binding in self._bindings.items()
            if binding_server == server_id
        }

        if not server_bindings:
            raise MCPBindingNotFoundError(
                "No AegisOps capability bindings exist for "
                f"MCP server {server_id!r}."
            )

        advertised_set = set(
            advertised
        )

        missing_bindings = (
            advertised_set
            - set(
                server_bindings
            )
        )

        if missing_bindings:
            raise MCPBindingNotFoundError(
                "Allowlisted MCP tools were discovered without "
                "authoritative AegisOps bindings: "
                + ", ".join(
                    sorted(
                        missing_bindings
                    )
                )
            )

        unavailable_bound_tools = (
            set(
                server_bindings
            )
            - advertised_set
        )

        if unavailable_bound_tools:
            raise MCPBindingNotFoundError(
                "Configured MCP capabilities are not currently "
                "advertised by the trusted gateway: "
                + ", ".join(
                    sorted(
                        unavailable_bound_tools
                    )
                )
            )

        registered: list[
            MCPRegisteredCapability
        ] = []

        for tool_name in sorted(
            advertised_set
        ):
            binding = server_bindings[
                tool_name
            ]

            capability = self._activate_binding(
                client=client,
                binding=binding,
            )

            registered.append(
                capability
            )

        self._active_servers.add(
            server_id
        )

        return tuple(
            registered
        )

    def build_action(
        self,
        *,
        server_id: str,
        tool_name: str,
        agent: str,
        arguments: Mapping[
            str,
            object,
        ],
        description: str | None = None,
    ) -> ProposedAction:
        """Create an authoritative ProposedAction for an MCP capability.

        Callers supply business arguments only. Security properties are
        copied from the application-owned capability binding.
        """

        identity = (
            server_id,
            tool_name,
        )

        capability = self._active.get(
            identity
        )

        if capability is None:
            raise MCPBindingNotFoundError(
                "MCP capability is not active: "
                f"{server_id}/{tool_name}"
            )

        agent_name = agent.strip()

        if not agent_name:
            raise ValueError(
                "agent must not be empty."
            )

        action_description = description

        if action_description is None:
            action_description = (
                capability.description
                or (
                    "Execute governed MCP capability "
                    f"{server_id}/{tool_name}."
                )
            )

        action_description = (
            action_description.strip()
        )

        if not action_description:
            raise ValueError(
                "description must not be empty."
            )

        return ProposedAction(
            agent=agent_name,
            action_type=(
                capability.action_type
            ),
            description=action_description,
            tool_name=(
                capability.runtime_name
            ),
            tool_arguments=dict(
                arguments
            ),
            risk=capability.risk,
            requires_approval=(
                capability.requires_approval
            ),
        )

    def get_capability(
        self,
        *,
        server_id: str,
        tool_name: str,
    ) -> MCPRegisteredCapability:
        """Return one activated MCP capability."""

        try:
            return self._active[
                (
                    server_id,
                    tool_name,
                )
            ]

        except KeyError as exc:
            raise MCPBindingNotFoundError(
                "MCP capability is not active: "
                f"{server_id}/{tool_name}"
            ) from exc

    def _activate_binding(
        self,
        *,
        client: MCPClientLike,
        binding: MCPCapabilityBinding,
    ) -> MCPRegisteredCapability:
        """Register one validated MCP tool into ToolRuntime."""

        assert binding.runtime_name is not None

        existing_runtime_tools = {
            name.casefold()
            for name in self._tool_runtime.registered_tools()
        }

        if (
            binding.runtime_name.casefold()
            in existing_runtime_tools
        ):
            raise MCPRuntimeToolCollisionError(
                "MCP capability runtime name already exists: "
                f"{binding.runtime_name}"
            )

        handler = self._gateway.build_tool_handler(
            client=client,
            server_id=binding.server_id,
            tool_name=binding.tool_name,
        )

        self._tool_runtime.register(
            name=binding.runtime_name,
            handler=handler,
            idempotent=binding.idempotent,
            timeout_seconds=(
                binding.timeout_seconds
            ),
            idempotency_argument=(
                binding.idempotency_argument
            ),
        )

        capability = MCPRegisteredCapability(
            server_id=binding.server_id,
            tool_name=binding.tool_name,
            runtime_name=binding.runtime_name,
            action_type=binding.action_type,
            risk=binding.risk,
            idempotent=binding.idempotent,
            requires_approval=(
                binding.requires_approval
            ),
            timeout_seconds=(
                binding.timeout_seconds
            ),
            idempotency_argument=(
                binding.idempotency_argument
            ),
            description=binding.description,
        )

        identity = (
            binding.server_id,
            binding.tool_name,
        )

        self._active[
            identity
        ] = capability

        return capability