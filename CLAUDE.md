# Working agreement for Claude sessions in this repository

This file is read at the start of every Claude session. The full process is in `docs/SDLC.md`. This is the short version you must follow.

## The rules

1. **No work without an issue.** Before changing anything, find the issue for the work or create one with the matching form fields (epic, user story, bug or task). Use sub-issues to link stories and tasks to their epic. Put the issue number in your plan.
2. **Use the specialised agents** in `.claude/agents/`. You are the coordinator. Hand each stage to the agent that owns it and give it the issue number and acceptance criteria in the brief.
   - `product-owner` for turning requests into issues, acceptance criteria and labels
   - `architect` for design before non-trivial changes (read only)
   - `developer` for implementation and tests
   - `tester` for trying to break the change and running the full checks
   - `code-reviewer` for an independent review of the diff (read only)
   - `devops-release` for CI, workflows, versions and release notes
   Small, obvious fixes (a typo, a one-line bug) may skip the architect, but never skip the issue, the tests or the review.
3. **Keep the issue current.** Comment on the issue when work starts, when the design is settled, and when a pull request opens. Update the checklist in the issue body as items finish.
4. **Branch and pull request.** Use the branch the platform gave you. Otherwise use `type/issue-number-short-slug`. The pull request description must contain `Closes #N` or `Refs #N`, and the template must be filled in. The Issue link check fails without it.
5. **Do not open a pull request or merge unless the maintainer asks.** Push your branch and report. When the maintainer does ask for a pull request, it must follow the template.
6. **Never merge on red.** The **CI Gate** check must pass on the latest commit. A red job is fixed at its cause. Never skip, disable or delete a test to get green.
7. **Labels** come from `.github/labels.yml`. Every issue gets one kind, one priority, one size and at least one area label. The issue tool only accepts labels that already exist, so add a new label to `.github/labels.yml` first and let the Sync labels workflow create it.
8. **Report honestly.** Say what was run and what the result was. If something was not verified, say so.

## Checks to run before every push

```
ruff check aria/ tests/ examples/
mypy aria/
PYTHONPATH=. python -m pytest tests/
PYTHONPATH=. python examples/demo_v11.py
```

Use `python -m pytest`, not the bare `pytest` command, so the right environment and plugins are used.

## Project facts

- Python 3.10 to 3.13. The framework has no mandatory runtime dependencies, so do not add any without an issue that explains why.
- Tests are in `tests/` and use pytest with pytest-asyncio.
- The CI workflow is `.github/workflows/ci.yml`. The single required check is **CI Gate**.
- Permission, security scanner and patcher code is security sensitive. Changes there need extra care and a review from `code-reviewer` that looks at security specifically.
