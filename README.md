

\# AegisOps



\### Governed Agentic AI for Incident Response and Autonomous Operations



\*\*AegisOps\*\* is an open-source Python engineering project exploring how autonomous AI agents can safely investigate incidents, propose remediation, interact with external tools, and operate under explicit security, reliability, and human-approval controls.



Rather than focusing only on LLM prompting and agent frameworks, AegisOps focuses on the engineering systems required to make agentic workflows controllable, observable, recoverable, and testable.



\*\*Author:\*\* \[Swapnil Mundhekar](https://github.com/SwapnilMundhekar)  

\*\*Repository:\*\* https://github.com/SwapnilMundhekar/AegisOps  

\*\*Language:\*\* Python 3.11+  

\*\*Status:\*\* Active development — control-plane foundation and integration tests



\---



\## 1. The Engineering Problem



Building an AI agent that calls a tool is relatively straightforward.



Building an agent that safely calls tools in a distributed production environment is substantially harder.



Consider an autonomous incident-response agent that wants to deploy a Kubernetes workload.



\- Who determines whether the agent is authorized?

\- What happens if a human must approve the deployment?

\- What happens if the process crashes after the deployment request is submitted?

\- Can another worker accidentally execute the same operation?

\- How are tool failures, retries, and timeouts handled?

\- Can a malicious MCP server persuade the agent to bypass security controls?

\- How do engineers reconstruct what happened across agents and tools?

\- Can the workflow resume from persisted state without repeating unsafe side effects?



AegisOps investigates these problems through a modular Python control-plane architecture.



\## 2. Architecture



```text

&#x20;                 Incident / User Request

&#x20;                           |

&#x20;                           v

&#x20;                 +--------------------+

&#x20;                 | Supervisor Router  |

&#x20;                 | Capability Matching|

&#x20;                 +--------------------+

&#x20;                           |

&#x20;                           v

&#x20;                 +--------------------+

&#x20;                 | Typed Agent Task   |

&#x20;                 | Budget / Deadline  |

&#x20;                 +--------------------+

&#x20;                           |

&#x20;                           v

&#x20;                 +--------------------+

&#x20;                 | Specialist Agent   |

&#x20;                 | Investigation /    |

&#x20;                 | Planning / Proposal|

&#x20;                 +--------------------+

&#x20;                           |

&#x20;                           v

&#x20;                 +--------------------+

&#x20;                 | Incident           |

&#x20;                 | Orchestrator       |

&#x20;                 +--------------------+

&#x20;                           |

&#x20;                           v

&#x20;                 +--------------------+

&#x20;                 | Policy Engine      |

&#x20;                 | Action Gate        |

&#x20;                 +--------------------+

&#x20;                           |

&#x20;                 +---------+---------+

&#x20;                 |                   |

&#x20;               ALLOW           REQUIRE APPROVAL

&#x20;                 |                   |

&#x20;                 |             Human Review

&#x20;                 |                   |

&#x20;                 +---------+---------+

&#x20;                           |

&#x20;                           v

&#x20;                 +--------------------+

&#x20;                 | ToolRuntime        |

&#x20;                 | Registered Tools   |

&#x20;                 +--------------------+

&#x20;                           |

&#x20;                 +---------+----------+

&#x20;                 |                    |

&#x20;                 v                    v

&#x20;         Execution Journal      MCP Trust Gateway

&#x20;         Leases / Replay        Schema Validation

&#x20;         Protection             Server Allowlist

&#x20;                 |                    |

&#x20;                 +---------+----------+

&#x20;                           |

&#x20;                           v

&#x20;                   External Systems

&#x20;                 GitHub / APIs / MCP

&#x20;                 Monitoring / Infra



&#x20;    Cross-cutting capabilities:

&#x20;    -----------------------------------------

&#x20;    Audit Ledger | Checkpoints | OpenTelemetry

&#x20;    Circuit Breakers | Timeouts | Retry Policy

```



\*\*Architecture note:\*\* This diagram represents the intended integrated architecture. Some components are implemented as independent modules or tested integration paths; complete end-to-end specialist-agent execution is still under development.



\## 3. Core Components



\### 3.1 Agent Contracts and Supervisor Routing



Relevant modules:



\- `src/aegisops/agent\_contracts.py`

\- `src/aegisops/supervisor.py`



AegisOps defines typed contracts for agent specifications, tasks, findings, evidence, results, and supervisor routing decisions.



The supervisor uses deterministic eligibility checks rather than allowing an LLM to assign permissions.



Routing evaluates:



\- Required agent capabilities

\- Permitted action types

\- Maximum execution steps

\- Task deadlines

\- Preferred-agent eligibility

\- Deterministic specialist selection



The supervisor currently validates and prepares agent tasks. Full specialist execution and lifecycle management are separate development milestones.



\### 3.2 Policy Enforcement and Human Approval



Relevant modules:



\- `src/aegisops/policy.py`

\- `src/aegisops/action\_gate.py`

\- `src/aegisops/approval.py`

\- `src/aegisops/orchestrator.py`



Agents propose actions; they do not grant themselves permission to execute them.



AegisOps separates:



1\. Proposed action

2\. Policy evaluation

3\. Approval requirement

4\. Authorized execution

5\. Audit recording



The architecture supports allow, deny, and approval-required decisions.



High-risk operational actions are intended to pass through explicit human review before execution.



\*\*Security principle:\*\* An agent's confidence, reasoning, or generated text is never sufficient authorization for a privileged operation.



\### 3.3 Durable Incident State



Relevant modules:



\- `src/aegisops/checkpoint.py`

\- `src/aegisops/orchestrator.py`



The checkpoint layer uses SQLite to persist incident state and support workflow restoration.



Optimistic concurrency control detects stale checkpoint writers.



This helps protect against multiple workers overwriting the same incident state after independently restoring an earlier version.



Checkpoint persistence does not, by itself, guarantee exactly-once external side effects.



\### 3.4 Durable Execution Journal



Relevant module:



\- `src/aegisops/execution\_journal.py`



The execution journal records external operation intent and lifecycle state.



```text

PREPARED

&#x20;   |

&#x20;   v

IN\_PROGRESS

&#x20;   |

&#x20;   +----> SUCCEEDED

&#x20;   |

&#x20;   +----> FAILED

&#x20;   |

&#x20;   +----> UNCERTAIN

```



Features include:



\- Stable idempotency keys

\- Atomic operation claims

\- Worker leases

\- Lease ownership tokens

\- Execution attempt tracking

\- Successful-operation deduplication

\- Stale-token rejection

\- Explicit uncertain-outcome handling



An expired non-idempotent operation is treated conservatively when its external outcome cannot be established.



The journal is designed to avoid blindly replaying potentially completed side effects.



\*\*Guarantee boundary:\*\* The journal provides durable intent and replay controls. It does not establish exactly-once execution across arbitrary external services.



Downstream idempotency support, reconciliation, and appropriate transactional boundaries remain necessary.



\### 3.5 Resilient Tool Execution



Relevant modules:



\- `src/aegisops/tool\_runtime.py`

\- `src/aegisops/resilience.py`



ToolRuntime provides a controlled execution boundary for registered tools.



Implemented design elements include:



\- Explicit tool registration

\- Policy and approval checks

\- Configurable execution timeouts

\- Retry policies

\- Per-tool circuit breakers

\- Idempotency-aware execution

\- Execution-journal integration

\- Structured audit events

\- Optional OpenTelemetry instrumentation



Retries are not automatically safe for mutating operations.



The execution contract must distinguish genuinely idempotent operations from operations that could produce duplicate side effects.



\### 3.6 MCP Security and Capability Governance



Relevant modules:



\- `src/aegisops/mcp\_trust.py`

\- `src/aegisops/mcp\_registry.py`



AegisOps includes a Model Context Protocol (MCP) trust boundary.



Security mechanisms include:



\- Explicit server allowlists

\- Explicit tool allowlists

\- JSON Schema input validation

\- Structured output validation

\- Payload-size restrictions

\- Restricted schema-reference handling

\- Controlled interactive MCP behavior

\- Untrusted-content labelling

\- Application-owned capability bindings



MCP server annotations are treated as metadata rather than authorization.



For example, an MCP server advertising a tool as read-only does not automatically make that tool safe or permitted.



AegisOps maintains its own action type, risk classification, approval requirement, and idempotency contract.



\*\*Important:\*\* Labelling external content as untrusted is a useful data-boundary mechanism, but it is not a complete prompt-injection defense. Enforcement must remain outside the LLM.



\### 3.7 Observability and Distributed Tracing



Relevant modules:



\- `src/aegisops/observability.py`

\- `src/aegisops/otel.py`



AegisOps implements a vendor-neutral correlation model with an OpenTelemetry adapter.



The observability design includes:



\- Trace IDs and span IDs

\- Parent-child span relationships

\- Incident and action correlation

\- Agent and tool correlation

\- W3C trace-context support

\- Structured span attributes

\- Error recording

\- OpenTelemetry SDK integration

\- OTLP export configuration



The project models agent workflows and tool execution as distinct observable operations.



```text

invoke\_workflow incident-response

&#x20;   |

&#x20;   +-- execute\_tool prometheus.query

&#x20;   |

&#x20;   +-- execute\_tool github.read\_issue

&#x20;   |

&#x20;   +-- execute\_tool kubernetes.deploy

```



The diagram illustrates the target trace structure; individual spans depend on which operations are actually invoked.



The adapter is designed for integration with OpenTelemetry-compatible backends such as Jaeger and Grafana Tempo.



No hosted observability deployment is required to run the unit tests.



\### 3.8 Auditability



Relevant module:



\- `src/aegisops/audit.py`



The audit layer records operational and security-relevant events.



Examples include:



\- Proposed actions

\- Policy decisions

\- Approval events

\- Tool execution attempts

\- Tool failures

\- Incident state transitions



The project includes audit-integrity verification in its test suite.



An application-level integrity check should not be confused with independently secured, tamper-proof storage. Production-grade audit assurance requires appropriate access controls, retention, and external integrity protection.



\## 4. Technology Stack



| Area | Technology |

|---|---|

| Language | Python |

| Data validation | Pydantic |

| Agent ecosystem | OpenAI Agents SDK, LangGraph |

| Tool interoperability | Model Context Protocol |

| Schema validation | JSON Schema |

| API framework dependency | FastAPI |

| Durable local storage | SQLite |

| Distributed tracing | OpenTelemetry |

| Metrics dependency | Prometheus client |

| Testing | pytest, pytest-asyncio |

| Static analysis tooling | Ruff, mypy |

| Version control | Git and GitHub |



The presence of a framework dependency does not imply that every corresponding integration or deployment path is complete.



\## 5. Repository Structure



```text

AegisOps/

|

|-- src/

|   `-- aegisops/

|       |-- agent\_contracts.py

|       |-- supervisor.py

|       |-- orchestrator.py

|       |-- domain.py

|       |-- policy.py

|       |-- action\_gate.py

|       |-- approval.py

|       |-- tool\_runtime.py

|       |-- resilience.py

|       |-- checkpoint.py

|       |-- execution\_journal.py

|       |-- audit.py

|       |-- observability.py

|       |-- otel.py

|       |-- mcp\_trust.py

|       `-- mcp\_registry.py

|

|-- tests/

|   `-- test\_\*.py

|

|-- pyproject.toml

|-- .gitignore

`-- README.md

```



This is a logical overview of the principal modules, not a claim that every possible deployment or agent implementation is present.



\## 6. Installation



\### Requirements



\- Python 3.11 or newer

\- Git

\- Windows, Linux, or macOS

\- An isolated Python virtual environment



\### Clone the repository



```bash

git clone https://github.com/SwapnilMundhekar/AegisOps.git

cd AegisOps

```



\### Create a virtual environment



Windows CMD:



```bat

py -3.12 -m venv .venv

.venv\\Scripts\\activate

```



Linux/macOS:



```bash

python3 -m venv .venv

source .venv/bin/activate

```



\### Install dependencies



```bash

python -m pip install -e ".\[dev]"

```



\### Verify dependency consistency



```bash

python -m pip check

```



Keep AegisOps isolated from unrelated machine-learning or audio-processing environments to avoid conflicting dependency requirements.



\## 7. Running Tests



Run the complete test suite:



```bash

python -m pytest -q

```



Run individual suites:



```bash

python -m pytest tests/test\_supervisor.py -v

python -m pytest tests/test\_mcp\_trust.py -v

python -m pytest tests/test\_mcp\_integration.py -v

python -m pytest tests/test\_execution\_journal.py -v

python -m pytest tests/test\_durable\_tool\_runtime.py -v

python -m pytest tests/test\_otel.py -v

```



Run coverage analysis:



```bash

python -m pytest --cov=aegisops --cov-report=term-missing

```



Coverage percentage and test results should be measured against the current commit rather than assumed from earlier runs.



\## 8. Reliability and Security Design Principles



\### Fail closed



Unknown tools, unauthorized operations, and ambiguous non-idempotent outcomes should not silently proceed.



\### Separate reasoning from authorization



LLM output can suggest actions, but policy evaluation and approval must be performed by trusted application components.



\### Make retries explicit



Retry safety depends on the external operation's semantics, not merely whether an API request failed.



\### Preserve causal context



Incident, agent, action, and tool execution should remain correlated across observable operations.



\### Treat external content as untrusted



MCP tool responses, retrieved documents, and external API payloads are data, not privileged instructions.



\### Prefer reproducible execution



Typed contracts, deterministic routing, explicit limits, and automated tests reduce hidden behavior.



\### Avoid unsupported guarantees



Durable checkpoints, leases, and idempotency controls improve reliability but do not automatically guarantee exactly-once execution.



\## 9. Current Development Status



AegisOps is an active engineering project, not a certified or independently audited production platform.



| Capability | Status |

|---|---|

| Typed agent contracts | Implemented |

| Deterministic supervisor routing | Implemented |

| Policy and action gating | Implemented |

| Human approval controls | Implemented |

| SQLite incident checkpoints | Implemented |

| Execution journal and leases | Implemented |

| Per-tool resilience | Implemented |

| OpenTelemetry adapter | Implemented |

| MCP trust gateway | Implemented |

| MCP capability registry | Implemented |

| Full specialist-agent execution dispatcher | Planned |

| End-to-end production deployment | Not established |

| External penetration/security audit | Not established |



"Implemented" indicates code exists in the repository; it does not independently establish production readiness.



\## 10. Engineering Roadmap



\### Phase 1 — Control Plane



\- \[x] Typed incident and action models

\- \[x] Policy evaluation

\- \[x] Human approval foundations

\- \[x] Audit recording

\- \[x] Durable checkpointing

\- \[x] Execution journal

\- \[x] Circuit breakers and retries



\### Phase 2 — Agent and MCP Governance



\- \[x] MCP trust gateway

\- \[x] MCP capability registry

\- \[x] Typed agent contracts

\- \[x] Deterministic supervisor routing

\- \[ ] Specialist-agent execution dispatcher

\- \[ ] Bounded concurrent agent execution

\- \[ ] Validated agent handoffs

\- \[ ] Full multi-agent incident-response workflow



\### Phase 3 — Production Engineering



\- \[ ] Automated CI quality gates

\- \[ ] Containerized deployment

\- \[ ] PostgreSQL-backed durable state

\- \[ ] Distributed worker coordination

\- \[ ] Reconciliation of uncertain external outcomes

\- \[ ] OpenTelemetry Collector deployment

\- \[ ] Prometheus/Grafana operational dashboards

\- \[ ] Security hardening and adversarial evaluation



\### Phase 4 — Evaluation and Optimization



\- \[ ] Agent task-success evaluations

\- \[ ] Tool-selection accuracy

\- \[ ] Policy-violation testing

\- \[ ] Prompt-injection resistance evaluations

\- \[ ] Recovery and fault-injection testing

\- \[ ] Token and inference-cost accounting

\- \[ ] Latency and throughput benchmarks

\- \[ ] Regression evaluation datasets



\## 11. Intended Use Cases



AegisOps is designed as a foundation for exploring:



\- AI-assisted site reliability engineering

\- Governed incident investigation

\- Human-approved infrastructure remediation

\- Secure MCP-based tool orchestration

\- Multi-agent operational workflows

\- Durable execution and recovery research

\- Agent observability and evaluation



\## 12. Engineering Scope and Limitations



AegisOps should currently be treated as an experimental engineering platform.



Do not grant it unrestricted production infrastructure permissions.



Before production deployment, additional work is required on authentication, authorization, secret management, transactional execution, external-operation reconciliation, persistent worker coordination, security evaluation, and operational monitoring.



\## 13. Author



\*\*Swapnil Mundhekar\*\*



AI Engineer | Python | Agentic AI | Distributed Systems | LLMOps



GitHub: https://github.com/SwapnilMundhekar



\---



\*\*AegisOps — building agentic systems that are not just capable, but governed, observable, and recoverable.\*\*



