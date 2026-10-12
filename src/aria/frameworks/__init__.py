"""Framework integrations for the ARIA orchestration layer.

Each subpackage adapts one agent framework to the framework-agnostic
`aria.orchestration.InterventionController`; `aria.frameworks.autogen` hooks
a MagenticOne group chat at its orchestrator. Other frameworks plug in their
own runner via `register_framework`.
"""
