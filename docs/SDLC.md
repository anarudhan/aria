# ARIA Software Development Life Cycle

This is the one process every change to ARIA follows. It applies to people and to Claude sessions alike.

The short version is this. No work starts without an issue. No code reaches `main` without a pull request that links the issue. No pull request merges while the CI Gate is red.

## 1. Stages

| Stage | What happens | Who leads | Output |
|---|---|---|---|
| 1. Discover | An idea, bug or request is written down | Anyone | A new issue labelled `needs-triage` |
| 2. Define | The issue becomes a user story, bug or task with acceptance criteria | Product owner | Issue meets the Definition of Ready and is labelled `ready` |
| 3. Design | Approach, affected modules and risks are agreed. Skipped only for small tasks | Architect | A short design comment on the issue, or an ADR in `docs/` for big decisions |
| 4. Build | Code and tests are written on a branch | Developer | Commits that reference the issue |
| 5. Verify | Tests, lint, demo and type checks are run, and edge cases are tried | Tester | Passing local checks and new or updated tests |
| 6. Review | An independent read of the diff for bugs, security and fit | Code reviewer | Review findings fixed or answered |
| 7. Integrate | Pull request, CI Gate green, merge into `main` | Developer and maintainer | Merged pull request that closes the issue |
| 8. Release | Version bump, changelog, tag and GitHub release | DevOps and release | A tagged release |
| 9. Operate and learn | Watch for regressions and feed lessons back as new issues | Maintainer | New issues in Discover |

## 2. Issue hierarchy

GitHub sub-issues are used to build the tree.

```
Epic                   a goal that needs several stories (for example "Self-healing v2")
  User story           value for a user, written as "As a ..., I want ..., so that ..."
    Task               a small piece of engineering work that finishes in one pull request
Bug                    something is broken and the fix is described by expected versus actual
Tech debt and spikes   cleanup and time-boxed research, created as tasks and given the tech-debt or spike label at triage
```

Rules of thumb:
- A task should fit in one pull request and about one day of work. If it does not, split it.
- A story is done when every acceptance criterion passes, not when the code is written.
- Every issue also uses a GitHub issue type (Feature, Bug or Task) so it shows up correctly on the board.

## 3. Labels

Labels live in `.github/labels.yml` and are synced to GitHub by the **Sync labels** workflow when that file changes on `main`. Change the file, not the label in the UI, so the set stays consistent. Syncing creates and updates labels but never deletes any.

| Group | Labels | Meaning |
|---|---|---|
| Kind | `epic`, `user-story`, `bug`, `task`, `tech-debt`, `spike`, `security`, `documentation`, `tests`, `process` | What sort of work this is |
| Priority | `priority-p0` to `priority-p3` | p0 drop everything, p1 this iteration, p2 soon, p3 whenever |
| Area | `area-core`, `area-agents`, `area-permissions`, `area-security`, `area-plugins`, `area-ci`, `area-docs` | Which part of the code is touched |
| Size | `size-xs`, `size-s`, `size-m`, `size-l`, `size-xl` | Rough effort. xs under an hour, s half a day, m a day, l a few days, xl needs splitting |
| State | `needs-triage`, `ready`, `blocked` | Where the issue sits before and during work |

Every issue needs one Kind, one Priority, at least one Area and one Size before it can be marked `ready`.

## 4. Definition of Ready

An issue can move to Ready only when:
- It has a clear title and the right issue form was used
- Acceptance criteria are written as checkable statements, ideally Given, When, Then
- Dependencies and open questions are listed
- It has Kind, Priority, Area and Size labels
- It is small enough for one pull request, or it is an epic or story with child issues

## 5. Definition of Done

A pull request is done only when all of these are true:
- The linked issue's acceptance criteria are met
- New behaviour has tests, and bug fixes have a test that failed before the fix
- `ruff check aria/ tests/ examples/` is clean
- `python -m pytest tests/` passes
- `python examples/demo_v11.py` still runs
- Documentation and the README changelog are updated when behaviour changes
- The CI Gate check is green on the latest commit
- Review findings are fixed or answered
- The pull request description links the issue with `Closes #N`

## 6. Branches, commits and pull requests

- Branch names are `type/issue-number-short-slug`, for example `feat/42-trust-decay` or `fix/19-version-sync`. Claude sessions use the branch the platform assigns them.
- Commit messages say what and why, and mention the issue, for example `Fix version drift (#19)`.
- One pull request per task. Keep them small.
- Every pull request description must contain a closing keyword and an issue number from this repository, such as `Closes #19`, or `Refs #19` when the pull request is only part of the work. Text inside HTML comments does not count. The **Issue link check** job fails the CI Gate without it. If you add the link after the first run, edit the description and CI runs again. Re-running an old run from the Actions page reuses the old description, so edit the description instead.
- Nobody pushes directly to `main`. The ruleset on `main` should require a pull request and a green **CI Gate**. Until the ruleset exists (issue #22) this is a team rule, not an enforced one.
- Merging is done by the maintainer, or by a Claude session only when the maintainer asks for it.

## 7. Project board

The GitHub Project is named **ARIA Delivery** and is a Board view with these columns.

| Column | Meaning | Set by |
|---|---|---|
| Backlog | New or not yet ready | Auto-add workflow |
| Ready | Meets the Definition of Ready | Maintainer or product owner |
| In progress | Someone is building it and a branch exists | Developer |
| In review | Pull request is open | Developer |
| Done | Issue closed or pull request merged | Project built-in workflow |

Suggested project fields besides Status are Priority, Size, Area and Iteration. Useful saved views are a Roadmap grouped by epic, a table grouped by priority, and a "Blocked" view filtered on the `blocked` label.

Turn on these built-in project workflows so the board mostly runs itself:
- Auto-add to project, filtered to this repository
- Item added sets Status to Backlog
- Item closed sets Status to Done
- Pull request merged sets Status to Done

Setting up the project and the ruleset needs repository admin rights and is tracked in issue #22.

## 8. Roles and specialised agents

Claude sessions use specialised agents defined in `.claude/agents/`. Each agent owns one stage, so one agent does not mark its own homework.

| Agent | Stage | Job |
|---|---|---|
| `product-owner` | Discover and Define | Turns requests into epics, stories and tasks with acceptance criteria. Triage and labelling. Keeps the board tidy |
| `architect` | Design | Reads the code, proposes an approach, lists risks and affected files. Read only |
| `developer` | Build | Implements one task on one branch with tests |
| `tester` | Verify | Tries to break the change, adds missing tests, runs the full checks. Does not edit production code |
| `code-reviewer` | Review | Independent review of the diff against the acceptance criteria. Read only |
| `devops-release` | Integrate and Release | Owns CI, workflows, versioning, changelog and release notes |

The main Claude session acts as the coordinator. It reads the issue, hands each stage to the right agent, passes the issue number and acceptance criteria in the brief, and reports back to the maintainer in plain language.

## 9. Releases

- Versions follow Semantic Versioning. Breaking changes bump the major number, new features the minor number, fixes the patch number.
- The version lives in one place only and everything else reads it (tracked in issue #19).
- A release is a pull request that bumps the version and updates the changelog, followed by a tag `vX.Y.Z` and a GitHub release whose notes list the closed issues.

## 10. Where to start

- New idea or request: open an issue with the matching form in the Issues tab.
- Picking up work: take an issue labelled `ready`, move it to In progress, and create a branch.
- Questions about the process: change this file by pull request, so the process evolves like the code does.
