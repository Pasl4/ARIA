# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-09

### Added

- Initial release of ARIA as a benchmark-agnostic framework.
- Contact-point controller that decides CONSULT or SILENT at structural
  moments of an episode (before each step, before delivery, before a stall
  replan), with a tabular every-visit Monte Carlo learner over a small
  discrete state.
- Advisor client for any OpenAI-compatible endpoint, with token and cost
  metering per role.
- AutoGen integration on top of the MagenticOne group chat (exact pin on the
  0.7.5 packages), plus a framework and benchmark registry for adding your own.
- Run logging, MAST-style failure annotations, a policy artifact store and the
  CLIs `aria-run`, `aria-train`, `aria-eval`, `aria-policy` and
  `aria-preflight`.
