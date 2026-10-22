"""Security boundary for untrusted Model Context Protocol servers."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from jsonschema import (
    Draft202012Validator,
    FormatChecker,
)
from jsonschema.exceptions import (
    SchemaError,
    ValidationError as JSONSchemaValidationError,
)
from mcp import MCPError
from mcp.types import (
    CallToolResult,
    InputRequiredResult,
    ListToolsResult,
    RequestParamsMeta,
    TextContent,
    Tool,
)
from pydantic import ValidationError as PydanticValidationError

from aegisops.observability import current_context


JSONValue = (
    None
    | bool
    | int
    | float
    | str
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


class MCPTrustError(RuntimeError):
    """Base exception raised by the MCP trust boundary."""


class MCPServerNotAllowedError(MCPTrustError):
    """Raised when an MCP server has no explicit trust policy."""


class MCPToolNotAllowedError(MCPTrustError):
    """Raised when a server tool is not explicitly allowlisted."""


class MCPToolNotAdvertisedError(MCPTrustError):
    """Raised when an allowlisted tool is absent from discovery."""


class MCPSchemaRejectedError(MCPTrustError):
    """Raised when an advertised schema is unsafe or invalid."""


class MCPInputValidationError(MCPTrustError):
    """Raised when proposed MCP tool arguments violate input schema."""


class MCPOutputValidationError(MCPTrustError):
    """Raised when MCP structured output violates output schema."""


class MCPPayloadTooLargeError(MCPTrustError):
    """Raised when an MCP payload exceeds configured trust limits."""


class MCPToolReturnedError(MCPTrustError):
    """Raised when the MCP server returns is_error=True."""


class MCPInteractiveInputBlockedError(MCPTrustError):
    """Raised when an MCP server requests an interactive follow-up."""


class MCPUnexpectedResultError(MCPTrustError):
    """Raised when an MCP server returns an unsupported result shape."""


class MCPUpstreamProtocolError(MCPTrustError):
    """Raised for an MCP JSON-RPC/protocol-level failure."""


class MCPInvalidPeerResponseError(MCPTrustError):
    """Raised when an MCP peer returns schema-invalid protocol data."""


class MCPUpstreamFailureError(MCPTrustError):
    """Raised for unexpected MCP transport or client failures."""


class MCPClientSessionLike(Protocol):
    """Minimal MCP session contract used by the trust gateway."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
        *,
        meta: RequestParamsMeta | None = None,
        allow_input_required: bool = False,
        allow_claimed: bool = False,
    ) -> CallToolResult | InputRequiredResult | Any:
        """Invoke an MCP tool."""


class MCPClientLike(Protocol):
    """Minimal connected MCP Client contract."""

    @property
    def session(
        self,
    ) -> MCPClientSessionLike:
        """Return the connected low-level client session."""

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
    ) -> ListToolsResult:
        """Discover tools from the connected server."""


@dataclass(frozen=True, slots=True)
class MCPServerPolicy:
    """Explicit trust policy for one MCP server."""

    server_id: str
    allowed_tools: frozenset[str]

    read_timeout_seconds: float = 30.0

    max_input_bytes: int = 64 * 1024
    max_output_bytes: int = 256 * 1024
    max_schema_bytes: int = 64 * 1024

    max_schema_depth: int = 32
    max_content_blocks: int = 64
    max_untrusted_text_chars: int = 16_000
    max_discovery_pages: int = 20

    require_output_schema: bool = False

    def __post_init__(
        self,
    ) -> None:
        if not self.server_id.strip():
            raise ValueError(
                "server_id must not be empty."
            )

        if not self.allowed_tools:
            raise ValueError(
                "allowed_tools must contain at least one tool."
            )

        if any(
            not tool.strip()
            for tool in self.allowed_tools
        ):
            raise ValueError(
                "allowed_tools must not contain empty names."
            )

        if self.read_timeout_seconds <= 0:
            raise ValueError(
                "read_timeout_seconds must be greater than zero."
            )

        integer_limits = {
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_schema_bytes": self.max_schema_bytes,
            "max_schema_depth": self.max_schema_depth,
            "max_content_blocks": self.max_content_blocks,
            "max_untrusted_text_chars": (
                self.max_untrusted_text_chars
            ),
            "max_discovery_pages": self.max_discovery_pages,
        }

        for name, value in integer_limits.items():
            if value < 1:
                raise ValueError(
                    f"{name} must be at least 1."
                )


@dataclass(frozen=True, slots=True)
class MCPValidatedResult:
    """Validated result returned across the MCP trust boundary.

    structured_content may be used by application code after schema
    validation.

    untrusted_text is explicitly labelled external content. It must
    never be treated as system/developer instructions.
    """

    server_id: str
    tool_name: str

    structured_content: JSONValue | None
    output_schema_validated: bool

    untrusted_text: tuple[str, ...]
    text_truncated: bool

    content_types: tuple[str, ...]

    def as_agent_payload(
        self,
    ) -> dict[str, object]:
        """Create an explicitly labelled payload for agent consumption."""

        return {
            "trust": "untrusted_external_mcp_output",
            "server_id": self.server_id,
            "tool_name": self.tool_name,
            "structured_content": self.structured_content,
            "structured_content_schema_validated": (
                self.output_schema_validated
            ),
            "untrusted_text": list(
                self.untrusted_text
            ),
            "text_truncated": self.text_truncated,
            "content_types": list(
                self.content_types
            ),
        }


MCPToolHandler = Callable[
    [dict[str, object]],
    Awaitable[object],
]


class MCPTrustGateway:
    """Validate and constrain MCP discovery and tool execution.

    This gateway deliberately treats every MCP server as untrusted,
    even when its transport connection is authenticated.
    """

    def __init__(
        self,
        policies: Mapping[
            str,
            MCPServerPolicy,
        ],
    ) -> None:
        if not policies:
            raise ValueError(
                "At least one MCP server policy is required."
            )

        self._policies: dict[
            str,
            MCPServerPolicy,
        ] = {}

        self._tools: dict[
            tuple[str, str],
            Tool,
        ] = {}

        for policy_key, policy in policies.items():
            normalized_key = policy_key.strip()

            if not normalized_key:
                raise ValueError(
                    "MCP policy map contains an empty server ID."
                )

            if normalized_key != policy.server_id:
                raise ValueError(
                    "MCP policy key must exactly match "
                    f"policy.server_id: {normalized_key!r} != "
                    f"{policy.server_id!r}."
                )

            if normalized_key in self._policies:
                raise ValueError(
                    f"Duplicate MCP policy: {normalized_key}"
                )

            self._policies[
                normalized_key
            ] = policy

    async def refresh_server(
        self,
        *,
        client: MCPClientLike,
        server_id: str,
    ) -> tuple[str, ...]:
        """Discover, validate, and cache allowlisted server tools."""

        policy = self._get_policy(
            server_id
        )

        discovered: dict[
            str,
            Tool,
        ] = {}

        cursor: str | None = None

        for _ in range(
            policy.max_discovery_pages
        ):
            try:
                listing = await client.list_tools(
                    cursor=cursor
                )

            except MCPError as exc:
                raise MCPUpstreamProtocolError(
                    "MCP tools/list failed for "
                    f"{server_id}: {exc}"
                ) from exc

            except PydanticValidationError as exc:
                raise MCPInvalidPeerResponseError(
                    "MCP server returned invalid tools/list "
                    f"data: {server_id}"
                ) from exc

            except Exception as exc:
                raise MCPUpstreamFailureError(
                    "Unexpected MCP discovery failure for "
                    f"{server_id}: {type(exc).__name__}"
                ) from exc

            for tool in listing.tools:
                if (
                    tool.name
                    not in policy.allowed_tools
                ):
                    continue

                if tool.name in discovered:
                    raise MCPSchemaRejectedError(
                        "MCP server advertised the same allowlisted "
                        f"tool more than once: {tool.name}"
                    )

                self._validate_tool_definition(
                    tool=tool,
                    policy=policy,
                )

                discovered[
                    tool.name
                ] = tool

            cursor = listing.next_cursor

            if cursor is None:
                break

        else:
            raise MCPUpstreamProtocolError(
                "MCP tool discovery exceeded the configured "
                f"pagination limit for {server_id}."
            )

        stale_keys = [
            key
            for key in self._tools
            if key[0] == server_id
        ]

        for key in stale_keys:
            del self._tools[
                key
            ]

        for tool_name, tool in discovered.items():
            self._tools[
                (
                    server_id,
                    tool_name,
                )
            ] = tool

        return tuple(
            sorted(
                discovered
            )
        )

    def advertised_tools(
        self,
        *,
        server_id: str,
    ) -> tuple[str, ...]:
        """Return validated tools cached for one server."""

        self._get_policy(
            server_id
        )

        return tuple(
            sorted(
                tool_name
                for (
                    cached_server,
                    tool_name,
                ) in self._tools
                if cached_server == server_id
            )
        )

    async def call_tool(
        self,
        *,
        client: MCPClientLike,
        server_id: str,
        tool_name: str,
        arguments: Mapping[
            str,
            object,
        ],
    ) -> MCPValidatedResult:
        """Validate, invoke, and validate an MCP tool result."""

        policy = self._get_policy(
            server_id
        )

        self._require_allowed_tool(
            policy=policy,
            tool_name=tool_name,
        )

        tool = self._tools.get(
            (
                server_id,
                tool_name,
            )
        )

        if tool is None:
            await self.refresh_server(
                client=client,
                server_id=server_id,
            )

            tool = self._tools.get(
                (
                    server_id,
                    tool_name,
                )
            )

        if tool is None:
            raise MCPToolNotAdvertisedError(
                f"MCP tool {tool_name!r} is allowlisted but "
                f"not advertised by server {server_id!r}."
            )

        payload = dict(
            arguments
        )

        self._enforce_json_size(
            value=payload,
            limit=policy.max_input_bytes,
            label="MCP input",
        )

        self._validate_instance(
            schema=tool.input_schema,
            value=payload,
            error_type=MCPInputValidationError,
            label=(
                f"MCP input for "
                f"{server_id}/{tool_name}"
            ),
        )

        meta = self._build_request_meta(
            server_id=server_id,
            tool_name=tool_name,
        )

        try:
            result = await client.session.call_tool(
                tool_name,
                payload,
                read_timeout_seconds=(
                    policy.read_timeout_seconds
                ),
                meta=meta,
                allow_input_required=True,
                allow_claimed=False,
            )

        except MCPError as exc:
            raise MCPUpstreamProtocolError(
                "MCP tools/call protocol failure for "
                f"{server_id}/{tool_name}: {exc}"
            ) from exc

        except PydanticValidationError as exc:
            raise MCPInvalidPeerResponseError(
                "MCP server returned protocol-invalid tool "
                f"result: {server_id}/{tool_name}"
            ) from exc

        except Exception as exc:
            raise MCPUpstreamFailureError(
                "Unexpected MCP tool-call failure for "
                f"{server_id}/{tool_name}: "
                f"{type(exc).__name__}"
            ) from exc

        if isinstance(
            result,
            InputRequiredResult,
        ):
            raise MCPInteractiveInputBlockedError(
                "MCP tool requested an additional interactive "
                "round. AegisOps blocks implicit MCP elicitation/"
                f"sampling at this trust boundary: "
                f"{server_id}/{tool_name}"
            )

        if not isinstance(
            result,
            CallToolResult,
        ):
            raise MCPUnexpectedResultError(
                "MCP tool returned an unsupported result shape: "
                f"{type(result).__name__}"
            )

        self._enforce_json_size(
            value=result.model_dump(
                mode="json",
                by_alias=True,
            ),
            limit=policy.max_output_bytes,
            label="MCP output",
        )

        if len(
            result.content
        ) > policy.max_content_blocks:
            raise MCPPayloadTooLargeError(
                "MCP output contains too many content blocks: "
                f"{len(result.content)} > "
                f"{policy.max_content_blocks}"
            )

        if result.is_error:
            raise MCPToolReturnedError(
                self._safe_tool_error_message(
                    server_id=server_id,
                    tool_name=tool_name,
                    result=result,
                )
            )

        structured_content = (
            result.structured_content
        )

        output_validated = False

        if tool.output_schema is not None:
            if structured_content is None:
                raise MCPOutputValidationError(
                    "MCP tool advertises output_schema but "
                    "returned no structured_content: "
                    f"{server_id}/{tool_name}"
                )

            self._validate_instance(
                schema=tool.output_schema,
                value=structured_content,
                error_type=MCPOutputValidationError,
                label=(
                    f"MCP output for "
                    f"{server_id}/{tool_name}"
                ),
            )

            output_validated = True

        elif policy.require_output_schema:
            raise MCPOutputValidationError(
                "MCP trust policy requires output_schema, "
                f"but {server_id}/{tool_name} does not "
                "advertise one."
            )

        if structured_content is not None:
            self._enforce_json_size(
                value=structured_content,
                limit=policy.max_output_bytes,
                label="MCP structured output",
            )

        (
            untrusted_text,
            text_truncated,
        ) = self._extract_untrusted_text(
            result=result,
            maximum_characters=(
                policy.max_untrusted_text_chars
            ),
        )

        content_types = tuple(
            str(
                getattr(
                    block,
                    "type",
                    type(
                        block
                    ).__name__,
                )
            )
            for block in result.content
        )

        return MCPValidatedResult(
            server_id=server_id,
            tool_name=tool_name,
            structured_content=structured_content,
            output_schema_validated=output_validated,
            untrusted_text=untrusted_text,
            text_truncated=text_truncated,
            content_types=content_types,
        )

    def build_tool_handler(
        self,
        *,
        client: MCPClientLike,
        server_id: str,
        tool_name: str,
    ) -> MCPToolHandler:
        """Build a ToolRuntime-compatible MCP handler.

        The returned payload retains a mandatory trust label so model
        code cannot accidentally receive raw MCP text without explicit
        untrusted-content metadata.
        """

        policy = self._get_policy(
            server_id
        )

        self._require_allowed_tool(
            policy=policy,
            tool_name=tool_name,
        )

        async def handler(
            arguments: dict[str, object],
        ) -> object:
            validated = await self.call_tool(
                client=client,
                server_id=server_id,
                tool_name=tool_name,
                arguments=arguments,
            )

            return validated.as_agent_payload()

        return handler

    def _get_policy(
        self,
        server_id: str,
    ) -> MCPServerPolicy:
        """Resolve one explicitly configured server policy."""

        try:
            return self._policies[
                server_id
            ]

        except KeyError as exc:
            raise MCPServerNotAllowedError(
                f"MCP server is not allowlisted: {server_id}"
            ) from exc

    @staticmethod
    def _require_allowed_tool(
        *,
        policy: MCPServerPolicy,
        tool_name: str,
    ) -> None:
        """Fail closed for tools outside the explicit allowlist."""

        if tool_name not in policy.allowed_tools:
            raise MCPToolNotAllowedError(
                f"MCP tool is not allowlisted for "
                f"{policy.server_id}: {tool_name}"
            )

    def _validate_tool_definition(
        self,
        *,
        tool: Tool,
        policy: MCPServerPolicy,
    ) -> None:
        """Validate advertised schemas before caching a tool."""

        self._validate_schema(
            schema=tool.input_schema,
            policy=policy,
            label=(
                f"input schema for "
                f"{policy.server_id}/{tool.name}"
            ),
            require_object_root=True,
        )

        if tool.output_schema is not None:
            self._validate_schema(
                schema=tool.output_schema,
                policy=policy,
                label=(
                    f"output schema for "
                    f"{policy.server_id}/{tool.name}"
                ),
                require_object_root=False,
            )

        elif policy.require_output_schema:
            raise MCPSchemaRejectedError(
                "MCP trust policy requires an output schema: "
                f"{policy.server_id}/{tool.name}"
            )

    def _validate_schema(
        self,
        *,
        schema: Mapping[
            str,
            Any,
        ],
        policy: MCPServerPolicy,
        label: str,
        require_object_root: bool,
    ) -> None:
        """Validate JSON Schema structure and trust constraints."""

        schema_dict = dict(
            schema
        )

        self._enforce_json_size(
            value=schema_dict,
            limit=policy.max_schema_bytes,
            label=label,
        )

        if (
            require_object_root
            and schema_dict.get(
                "type"
            )
            != "object"
        ):
            raise MCPSchemaRejectedError(
                f"{label} must have type='object' "
                "at the schema root."
            )

        try:
            Draft202012Validator.check_schema(
                schema_dict
            )

        except SchemaError as exc:
            raise MCPSchemaRejectedError(
                f"Invalid JSON Schema 2020-12: {label}"
            ) from exc

        self._inspect_schema_node(
            node=schema_dict,
            depth=0,
            maximum_depth=(
                policy.max_schema_depth
            ),
            label=label,
        )

    def _inspect_schema_node(
        self,
        *,
        node: object,
        depth: int,
        maximum_depth: int,
        label: str,
    ) -> None:
        """Reject excessive depth and external schema references."""

        if depth > maximum_depth:
            raise MCPSchemaRejectedError(
                f"{label} exceeds maximum schema depth "
                f"{maximum_depth}."
            )

        if isinstance(
            node,
            Mapping,
        ):
            for key, value in node.items():
                if (
                    key
                    in {
                        "$ref",
                        "$dynamicRef",
                    }
                    and isinstance(
                        value,
                        str,
                    )
                    and not value.startswith(
                        "#"
                    )
                ):
                    raise MCPSchemaRejectedError(
                        f"{label} contains an external "
                        f"{key} reference. Only local '#...' "
                        "references are permitted."
                    )

                self._inspect_schema_node(
                    node=value,
                    depth=depth + 1,
                    maximum_depth=maximum_depth,
                    label=label,
                )

            return

        if isinstance(
            node,
            list,
        ):
            for value in node:
                self._inspect_schema_node(
                    node=value,
                    depth=depth + 1,
                    maximum_depth=maximum_depth,
                    label=label,
                )

    @staticmethod
    def _validate_instance(
        *,
        schema: Mapping[
            str,
            Any,
        ],
        value: object,
        error_type: type[
            MCPTrustError
        ],
        label: str,
    ) -> None:
        """Validate a JSON value against Draft 2020-12."""

        validator = Draft202012Validator(
            dict(
                schema
            ),
            format_checker=FormatChecker(),
        )

        try:
            validator.validate(
                value
            )

        except JSONSchemaValidationError as exc:
            path = (
                ".".join(
                    str(
                        part
                    )
                    for part in exc.absolute_path
                )
                or "<root>"
            )

            raise error_type(
                f"{label} failed schema validation "
                f"at {path}: {exc.message}"
            ) from exc

    @staticmethod
    def _enforce_json_size(
        *,
        value: object,
        limit: int,
        label: str,
    ) -> None:
        """Enforce size limits using strict JSON serialization."""

        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(
                "utf-8"
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise MCPTrustError(
                f"{label} is not valid JSON data."
            ) from exc

        if len(
            encoded
        ) > limit:
            raise MCPPayloadTooLargeError(
                f"{label} exceeds configured limit: "
                f"{len(encoded)} bytes > {limit} bytes."
            )

    @staticmethod
    def _extract_untrusted_text(
        *,
        result: CallToolResult,
        maximum_characters: int,
    ) -> tuple[
        tuple[str, ...],
        bool,
    ]:
        """Extract bounded text while preserving its untrusted label."""

        remaining = maximum_characters

        values: list[
            str
        ] = []

        truncated = False

        for block in result.content:
            if not isinstance(
                block,
                TextContent,
            ):
                continue

            text = block.text

            if remaining <= 0:
                truncated = True
                break

            if len(
                text
            ) <= remaining:
                values.append(
                    text
                )

                remaining -= len(
                    text
                )

                continue

            values.append(
                text[
                    :remaining
                ]
            )

            remaining = 0
            truncated = True

        return (
            tuple(
                values
            ),
            truncated,
        )

    @staticmethod
    def _safe_tool_error_message(
        *,
        server_id: str,
        tool_name: str,
        result: CallToolResult,
    ) -> str:
        """Return a bounded error message from untrusted tool content."""

        text = next(
            (
                block.text
                for block in result.content
                if isinstance(
                    block,
                    TextContent,
                )
            ),
            "",
        )

        bounded = text[
            :512
        ]

        if bounded:
            return (
                f"MCP tool returned an error: "
                f"{server_id}/{tool_name}: {bounded}"
            )

        return (
            "MCP tool returned an error without textual "
            f"detail: {server_id}/{tool_name}"
        )

    @staticmethod
    def _build_request_meta(
        *,
        server_id: str,
        tool_name: str,
    ) -> RequestParamsMeta:
        """Build non-sensitive business correlation metadata.

        MCP Python v2 already propagates OpenTelemetry trace context.
        These fields add AegisOps business correlation only.
        """

        meta: RequestParamsMeta = {
            "aegisops.server_id": server_id,
            "aegisops.tool_name": tool_name,
            "aegisops.trust_boundary": "external_mcp",
        }

        context = current_context()

        if context is None:
            return meta

        if context.incident_id is not None:
            meta[
                "aegisops.incident_id"
            ] = context.incident_id

        if context.action_id is not None:
            meta[
                "aegisops.action_id"
            ] = context.action_id

        if context.agent_name is not None:
            meta[
                "aegisops.agent_name"
            ] = context.agent_name

        return meta