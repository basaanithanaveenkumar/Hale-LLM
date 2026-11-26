# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Engineering Spec

**Apply to all code you write, in addition to the guidelines above.**

1. **Production code:** handle real failures, no demo shortcuts.
2. **SOLID and OOP:** small classes with one job, depend on abstractions.
3. **Modular, packagable code:** importable package, no script-style logic in modules.
4. **One class per file.**
5. **Naming:** clear, conventional names (PEP 8 for Python).
6. **Documentation:** proper docs, with Mermaid diagrams where they help.
7. **Docstrings and comments:** docstrings on public code, comments that explain the why.
8. **Tests:** unit tests and integration tests.
9. **Logging and exceptions:** proper logging (use loguru or the language's standard tool) and explicit exception handling.
10. **Efficiency:** use the right data structures and algorithms where it matters.
11. **Repo structure:** follow a standard production layout (`src/`, `tests/`, `configs/`, `docs/`).
12. **Design patterns:** use them where they fit, not for show.
13. **Configs:** YAML or JSON files with a config loader, where a base file can be referenced and overridden.
14. **Registries:** MMDetection-style registries where swappable components need them.
15. **Licenses:** pick MIT or Apache licensed libraries only.
16. **Readable code, no AI slop:** a human can follow it, with comments that explain the steps one by one.

When this spec conflicts with Simplicity First, keep the logic simple and keep the structure rules (one class per file, registries, tests).

## HALE-LLM project rules

- Push to `main` only when the user explicitly asks.
- Code style: one class per file, SOLID, docstrings, step-by-step `forward` with a shape comment on every line.
- New code comes with unit tests (and integration tests where it crosses modules); run `uv run pytest` before reporting done.
- Use loguru for logging, YAML configs with inheritance, registries for swappable parts, and MIT/Apache libraries only.
- Commands, layout and testing notes: see `.claude/skills/hale-llm-dev/SKILL.md`.
