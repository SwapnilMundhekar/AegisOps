"""Tests for governed MCP capability registration in AegisOps."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from mcp.types import ToolAnnotations

from aegisops.action_gate import ActionGate
from aegisops.approval import ApprovalManager
from aegisops.audit import AuditLedger
from aegisops.domain import (
    ActionType,
    IncidentSeverity,
    IncidentState,
    RiskLevel,
)
from aegisops.execution_journal import (
    ExecutionStatus,
    SQLiteExecutionJournal,
)
from aegisops.mcp_registry import (
    MCPBindingConflictError,
    MCPBindingNotFoundError,
    MCPCapabilityBinding,
    MCPCapabilityRegistry,
    MCPRuntimeToolCollisionError,
    MCPServerAlreadyActivatedError,
)
from aegisops.policy import PolicyEngine
from aegisops.tool_runtime import (
    ToolApprovalRequiredError,
    ToolRuntime,
)


class FakeMCPClient:
    """Opaque client object used by the fake trust gateway."""


class FakeTrustGateway:
    """Deterministic trust-gateway stand-in for registry tests.

    reported_annotations deliberately represent MCP-provided hints.
    The capability registry must never use these hints to determine
    ActionType, RiskLevel, idempotency, or approval requirements.
    """

    def __init__(
        self,
        *,
        advertised: Mapping[
            str,
            tuple[str, ...],
        ],
        reported_annotations: Mapping[
            tuple[str, str],
            ToolAnnotations,
        ]
        | None = None,
    ) -> None:
        self._advertised = dict(
            advertised
        )

        self.reported_annotations = dict(
            reported_annotations
            or {}
        )

        self.refresh_calls: list[
            str
        ] = []

        self.handler_calls: list[
            dict[str, object]
        ] = []

    async def refresh_server(
        self,
        *,
        client: object,
        server_id: str,
    ) -> tuple[str, ...]:
        """Return validated tool names for the requested fake server."""

        del client

        self.refresh_calls.append(
            server_id
        )

        return self._advertised.get(
            server_id,
            (),
        )

    def build_tool_handler(
        self,
        *,
        client: object,
        server_id: str,
        tool_name: str,
    ):
        """Build a ToolRuntime-compatible fake MCP handler."""

        del client

        async def handler(
            arguments: dict[str, object],
        ) -> object:
            captured = dict(
                arguments
            )

            self.handler_calls.append(
                {
                    "server_id": server_id,
                    "tool_name": tool_name,
                    "arguments": captured,
                }
            )

            return {
                "trust": "untrusted_external_mcp_output",
                "server_id": server_id,
                "tool_name": tool_name,
                "structured_content": {
                    "received": captured,
                },
                "structured_content_schema_validated": True,
                "untrusted_text": [],
                "text_truncated": False,
                "content_types": [],
            }

        return handler


def build_runtime(
    *,
    execution_journal: SQLiteExecutionJournal | None = None,
) -> tuple[
    ToolRuntime,
    AuditLedger,
]:
    """Build a normal AegisOps execution boundary."""

    ledger = AuditLedger()

    policy = PolicyEngine()

    approvals = ApprovalManager(
        audit_ledger=ledger,
    )

    gate = ActionGate(
        policy_engine=policy,
        audit_ledger=ledger,
    )

    runtime = ToolRuntime(
        action_gate=gate,
        approval_manager=approvals,
        audit_ledger=ledger,
        execution_journal=execution_journal,
        worker_id="registry-test-worker",
    )

    return (
        runtime,
        ledger,
    )


def build_incident(
    *,
    environment: str = "staging",
) -> IncidentState:
    """Create a representative incident."""

    return IncidentState(
        title="Payment API degradation",
        summary="Investigate and remediate production behaviour.",
        service="payment-api",
        environment=environment,
        severity=IncidentSeverity.SEV2,
    )


@pytest.mark.asyncio
async def test_activation_registers_governed_runtime_capability() -> None:
    """An activated binding should become a usable ToolRuntime tool."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
            ),
        }
    )

    runtime, ledger = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_issue",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
                requires_approval=False,
                description="Read a GitHub issue.",
            ),
        ),
    )

    activated = await registry.activate_server(
        client=FakeMCPClient(),  # type: ignore[arg-type]
        server_id="github-mcp",
    )

    assert len(
        activated
    ) == 1

    capability = activated[
        0
    ]

    assert (
        capability.server_id
        == "github-mcp"
    )

    assert (
        capability.tool_name
        == "github.read_issue"
    )

    assert (
        capability.runtime_name
        == "mcp.github-mcp.github.read_issue"
    )

    assert (
        capability.action_type
        is ActionType.READ
    )

    assert (
        capability.risk
        is RiskLevel.LOW
    )

    assert capability.idempotent is True

    assert (
        capability.requires_approval
        is False
    )

    assert runtime.registered_tools() == (
        "mcp.github-mcp.github.read_issue",
    )

    action = registry.build_action(
        server_id="github-mcp",
        tool_name="github.read_issue",
        agent="investigator-agent",
        arguments={
            "owner": "SwapnilMundhekar",
            "repo": "AegisOps",
            "issue_number": 42,
        },
    )

    assert (
        action.action_type
        is ActionType.READ
    )

    assert (
        action.risk
        is RiskLevel.LOW
    )

    assert (
        action.requires_approval
        is False
    )

    assert (
        action.tool_name
        == "mcp.github-mcp.github.read_issue"
    )

    result = await runtime.execute(
        incident=build_incident(),
        action=action,
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["trust"]
        == "untrusted_external_mcp_output"
    )

    assert len(
        gateway.handler_calls
    ) == 1

    assert ledger.verify_integrity() is True


@pytest.mark.asyncio
async def test_mcp_annotations_cannot_downgrade_aegisops_authority() -> None:
    """Server hints must not override application-owned security policy."""

    misleading_annotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    gateway = FakeTrustGateway(
        advertised={
            "kubernetes-mcp": (
                "kubernetes.deploy",
            ),
        },
        reported_annotations={
            (
                "kubernetes-mcp",
                "kubernetes.deploy",
            ): misleading_annotations,
        },
    )

    runtime, _ = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="kubernetes-mcp",
                tool_name="kubernetes.deploy",
                action_type=ActionType.DEPLOY,
                risk=RiskLevel.HIGH,
                idempotent=False,
                requires_approval=True,
                description="Deploy a production workload.",
            ),
        ),
    )

    await registry.activate_server(
        client=FakeMCPClient(),  # type: ignore[arg-type]
        server_id="kubernetes-mcp",
    )

    action = registry.build_action(
        server_id="kubernetes-mcp",
        tool_name="kubernetes.deploy",
        agent="deployment-agent",
        arguments={
            "service": "payment-api",
            "version": "1.8.3",
        },
    )

    assert (
        gateway.reported_annotations[
            (
                "kubernetes-mcp",
                "kubernetes.deploy",
            )
        ].read_only_hint
        is True
    )

    assert (
        action.action_type
        is ActionType.DEPLOY
    )

    assert (
        action.risk
        is RiskLevel.HIGH
    )

    assert (
        action.requires_approval
        is True
    )

    capability = registry.get_capability(
        server_id="kubernetes-mcp",
        tool_name="kubernetes.deploy",
    )

    assert capability.idempotent is False


@pytest.mark.asyncio
async def test_dangerous_binding_requires_approval_at_runtime() -> None:
    """A governed deployment must stop at the human approval gate."""

    gateway = FakeTrustGateway(
        advertised={
            "kubernetes-mcp": (
                "kubernetes.deploy",
            ),
        }
    )

    runtime, ledger = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="kubernetes-mcp",
                tool_name="kubernetes.deploy",
                action_type=ActionType.DEPLOY,
                risk=RiskLevel.HIGH,
                idempotent=False,
                requires_approval=True,
            ),
        ),
    )

    await registry.activate_server(
        client=FakeMCPClient(),  # type: ignore[arg-type]
        server_id="kubernetes-mcp",
    )

    action = registry.build_action(
        server_id="kubernetes-mcp",
        tool_name="kubernetes.deploy",
        agent="deployment-agent",
        arguments={
            "service": "payment-api",
            "version": "1.8.3",
        },
    )

    with pytest.raises(
        ToolApprovalRequiredError,
        match="requires human approval",
    ):
        await runtime.execute(
            incident=build_incident(
                environment="production",
            ),
            action=action,
        )

    assert gateway.handler_calls == []

    assert ledger.verify_integrity() is True


def test_deploy_binding_without_approval_is_rejected() -> None:
    """Deployment capability declarations must fail closed."""

    with pytest.raises(
        ValueError,
        match="DEPLOY and ROLLBACK",
    ):
        MCPCapabilityBinding(
            server_id="kubernetes-mcp",
            tool_name="kubernetes.deploy",
            action_type=ActionType.DEPLOY,
            risk=RiskLevel.HIGH,
            idempotent=False,
            requires_approval=False,
        )


def test_high_risk_mutation_without_approval_is_rejected() -> None:
    """HIGH-risk mutations cannot be configured for autonomous use."""

    with pytest.raises(
        ValueError,
        match="HIGH or CRITICAL",
    ):
        MCPCapabilityBinding(
            server_id="admin-mcp",
            tool_name="admin.rotate_credentials",
            action_type=ActionType.EXECUTE,
            risk=RiskLevel.HIGH,
            idempotent=False,
            requires_approval=False,
        )


def test_duplicate_runtime_name_is_rejected() -> None:
    """Different MCP capabilities must not alias the same runtime tool."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
                "github.read_file",
            ),
        }
    )

    runtime, _ = build_runtime()

    with pytest.raises(
        MCPBindingConflictError,
        match="same runtime_name",
    ):
        MCPCapabilityRegistry(
            gateway=gateway,  # type: ignore[arg-type]
            tool_runtime=runtime,
            bindings=(
                MCPCapabilityBinding(
                    server_id="github-mcp",
                    tool_name="github.read_issue",
                    action_type=ActionType.READ,
                    risk=RiskLevel.LOW,
                    idempotent=True,
                    runtime_name="mcp.shared",
                ),
                MCPCapabilityBinding(
                    server_id="github-mcp",
                    tool_name="github.read_file",
                    action_type=ActionType.READ,
                    risk=RiskLevel.LOW,
                    idempotent=True,
                    runtime_name="mcp.shared",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_existing_runtime_tool_cannot_be_replaced() -> None:
    """MCP activation must never overwrite an existing runtime handler."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
            ),
        }
    )

    runtime, _ = build_runtime()

    async def existing_handler(
        arguments: dict[str, object],
    ) -> str:
        return str(
            arguments
        )

    runtime.register(
        name="mcp.github-mcp.github.read_issue",
        handler=existing_handler,
        idempotent=True,
    )

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_issue",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
            ),
        ),
    )

    with pytest.raises(
        MCPRuntimeToolCollisionError,
        match="already exists",
    ):
        await registry.activate_server(
            client=FakeMCPClient(),  # type: ignore[arg-type]
            server_id="github-mcp",
        )


@pytest.mark.asyncio
async def test_discovered_tool_without_binding_fails_activation() -> None:
    """Every trusted-gateway tool still requires AegisOps authority."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
                "github.create_release",
            ),
        }
    )

    runtime, _ = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_issue",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
            ),
        ),
    )

    with pytest.raises(
        MCPBindingNotFoundError,
        match="without authoritative",
    ):
        await registry.activate_server(
            client=FakeMCPClient(),  # type: ignore[arg-type]
            server_id="github-mcp",
        )

    assert runtime.registered_tools() == ()


@pytest.mark.asyncio
async def test_bound_tool_missing_from_server_fails_activation() -> None:
    """Configured capabilities must not silently disappear at discovery."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
            ),
        }
    )

    runtime, _ = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_issue",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
            ),
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_file",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
            ),
        ),
    )

    with pytest.raises(
        MCPBindingNotFoundError,
        match="not currently advertised",
    ):
        await registry.activate_server(
            client=FakeMCPClient(),  # type: ignore[arg-type]
            server_id="github-mcp",
        )

    assert runtime.registered_tools() == ()


def test_capability_cannot_build_action_before_activation() -> None:
    """Configured metadata alone must not make a tool executable."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
            ),
        }
    )

    runtime, _ = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_issue",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
            ),
        ),
    )

    with pytest.raises(
        MCPBindingNotFoundError,
        match="not active",
    ):
        registry.build_action(
            server_id="github-mcp",
            tool_name="github.read_issue",
            agent="investigator-agent",
            arguments={
                "issue_number": 42,
            },
        )


@pytest.mark.asyncio
async def test_same_server_cannot_be_activated_twice() -> None:
    """Activation should be one lifecycle operation per registry."""

    gateway = FakeTrustGateway(
        advertised={
            "github-mcp": (
                "github.read_issue",
            ),
        }
    )

    runtime, _ = build_runtime()

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="github-mcp",
                tool_name="github.read_issue",
                action_type=ActionType.READ,
                risk=RiskLevel.LOW,
                idempotent=True,
            ),
        ),
    )

    client = FakeMCPClient()

    await registry.activate_server(
        client=client,  # type: ignore[arg-type]
        server_id="github-mcp",
    )

    with pytest.raises(
        MCPServerAlreadyActivatedError,
        match="already activated",
    ):
        await registry.activate_server(
            client=client,  # type: ignore[arg-type]
            server_id="github-mcp",
        )


@pytest.mark.asyncio
async def test_registry_propagates_stable_downstream_idempotency_key(
    tmp_path: Path,
) -> None:
    """Registry binding should connect MCP dedupe semantics to journal."""

    journal = SQLiteExecutionJournal(
        tmp_path / "execution.db"
    )

    gateway = FakeTrustGateway(
        advertised={
            "payments-mcp": (
                "payments.upsert",
            ),
        }
    )

    runtime, ledger = build_runtime(
        execution_journal=journal,
    )

    registry = MCPCapabilityRegistry(
        gateway=gateway,  # type: ignore[arg-type]
        tool_runtime=runtime,
        bindings=(
            MCPCapabilityBinding(
                server_id="payments-mcp",
                tool_name="payments.upsert",
                action_type=ActionType.WRITE,
                risk=RiskLevel.MEDIUM,
                idempotent=True,
                requires_approval=False,
                idempotency_argument="idempotency_key",
            ),
        ),
    )

    await registry.activate_server(
        client=FakeMCPClient(),  # type: ignore[arg-type]
        server_id="payments-mcp",
    )

    incident = build_incident()

    action = registry.build_action(
        server_id="payments-mcp",
        tool_name="payments.upsert",
        agent="finance-agent",
        arguments={
            "customer_id": "cust-001",
            "amount": 125,
        },
    )

    assert (
        "idempotency_key"
        not in action.tool_arguments
    )

    expected_key = (
        SQLiteExecutionJournal.build_idempotency_key(
            incident_id=incident.id,
            action_id=action.id,
            tool_name=action.tool_name,
        )
    )

    result = await runtime.execute(
        incident=incident,
        action=action,
    )

    assert isinstance(
        result,
        dict,
    )

    assert len(
        gateway.handler_calls
    ) == 1

    arguments = gateway.handler_calls[
        0
    ][
        "arguments"
    ]

    assert isinstance(
        arguments,
        dict,
    )

    assert (
        arguments[
            "idempotency_key"
        ]
        == expected_key
    )

    assert (
        "idempotency_key"
        not in action.tool_arguments
    )

    record = journal.load(
        expected_key
    )

    assert (
        record.status
        is ExecutionStatus.SUCCEEDED
    )

    assert record.attempt_count == 1

    assert ledger.verify_integrity() is True