# Third-party notices

ARIA is MIT licensed (see `LICENSE`). Two source files contain code adapted
from other MIT-licensed projects. Their notices are reproduced below as the
MIT license requires.

## AutoGen (autogen-agentchat 0.7.5)

- Source: https://github.com/microsoft/autogen, package `autogen-agentchat`
  0.7.5, `autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py`.
- Used in: `src/aria/frameworks/autogen/team.py`. `AriaOrchestrator._orchestrate_step`
  and `AriaOrchestrator._aria_prepare_final_answer` mirror the bodies of
  `MagenticOneOrchestrator._orchestrate_step` and `_prepare_final_answer`, with
  ARIA additions in marked blocks. This is why the AutoGen packages are pinned
  exactly in `pyproject.toml`.
- License: MIT.

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Magentic-UI

- Source: https://github.com/microsoft/magentic-ui, module `magentic_ui.agents._coder`.
- Used in: `src/aria/frameworks/autogen/agents/coder.py`. `CoderAgent` is a
  trimmed copy of the Magentic-UI coder agent (write, execute, debug loop and
  markdown code-block extraction). Approval, pause/resume, Docker bootstrap,
  summarization and multimodal handling were removed; the system prompt is
  injected from the team spec instead of hardcoded.
- License: MIT.

```
MIT License

Copyright (c) 2025 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Attributions without copied code

- MAST failure taxonomy. `src/aria/evaluation/mast.py` uses the 14 failure-mode
  and 3 cluster names verbatim as labels, so annotations line up with the
  published taxonomy. Source: Cemri et al., "Why Do Multi-Agent LLM Systems
  Fail?", arXiv:2503.13657. The annotators themselves are original code.
- Anthropic advisor tool documentation. `src/aria/advisor/advisor.py` and
  `src/aria/policies/baselines.py` quote short phrases from Anthropic's public
  advisor tool documentation to show which documented behaviour each rule
  implements. No code is copied.
