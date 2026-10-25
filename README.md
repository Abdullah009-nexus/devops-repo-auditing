# DevOps Repository Auditing Agent

An LLM-driven agent that inspects a **public GitHub repository**, finds DevOps and software-engineering best-practice violations, and writes an **evidence-based JSON report** with actionable fixes. It never modifies the repository and never runs its code.

```bash
python agent.py --repo https://github.com/owner/repository --output report.json
```

> Course project: *MLOps & Production AI Engineering, Assignment 2*. The scope is software engineering and DevOps only; ML training, datasets and experiment tracking are out of scope.

---

## Contents
- [Why an agent?](#why-an-agent)
- [Quick start](#quick-start)
- [Choosing a free LLM](#choosing-a-free-llm)
- [Usage, output and exit codes](#usage-output-and-exit-codes)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [What it checks](#what-it-checks)
- [Avoiding false positives](#avoiding-false-positives)
- [Configuration](#configuration)
- [Safety](#safety)
- [Testing and evaluation](#testing-and-evaluation)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Assignment requirement map](#assignment-requirement-map)

---

## Why an agent?

A static scanner runs the same checks everywhere and can't tell *"no ruff config"* from *"uses flake8 through pre-commit"*. This project runs a loop instead:

**Observe → Reason → Select tool → Execute tool → Interpret result → … → Report**

The LLM decides what to look at next: it reads the workflow it just found, follows a suspicious line, and checks for an equivalent tool before calling something "missing". Python owns everything else: which tools exist, argument validation, budgets, redaction and finding validation. **No agent framework** is used (no LangChain, LangGraph, n8n or similar); the loop is plain Python with the standard library only.

---

## Quick start

**Requirements:** Python 3.11+, `git` on the PATH (recommended; the agent falls back to the GitHub API without it), internet access, and an API key for any **OpenAI-compatible** endpoint that supports tool calling. There are **no third-party Python dependencies**.

```bash
git clone https://github.com/Abdullah009-nexus/devops-repo-auditing.git
cd devops-repo-auditing
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e .        # optional: installs the `devops-audit` command; nothing else to install
```

Configure the provider. The easiest way is a `.env` file in the folder you run from:

```bash
cp .env.example .env        # Windows: copy .env.example .env
# edit .env: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL (and optionally GITHUB_TOKEN)
```

Real environment variables always take precedence over `.env`. Run an audit:

```bash
python agent.py --repo https://github.com/owner/repository --output report.json
```

`.env` is git-ignored. Keys are never written to logs, traces or reports.

---

## Choosing a free LLM

Any OpenAI-compatible endpoint with tool calling works; the model choice doesn't change how the agent works.

| Provider | `OPENAI_BASE_URL` | Notes |
|---|---|---|
| **Google Gemini** (recommended free option) | `https://generativelanguage.googleapis.com/v1beta/openai/` | Free key from Google AI Studio; use a current **Flash** model; large context window |
| **Groq** (backup) | `https://api.groq.com/openai/v1` | Free tier, fast, smaller tokens-per-minute limits |
| **OpenAI** (paid) | `https://api.openai.com/v1` (default) | A tool-capable model that accepts sampling parameters, e.g. `gpt-4.1-mini` |

Some newer models reject the `temperature` setting; the agent detects that, retries once without it and remembers. Free tiers change their models and limits often; check the provider console. Rate-limit (429) and server (5xx) errors are retried automatically with backoff.

Optional: set `GITHUB_TOKEN` (any read-only token) to raise the GitHub API rate limit. It's only sent to `api.github.com` and to the clone, never in URLs or command lines.

---

## Usage, output and exit codes

```
python agent.py --repo <github_repository_url> --output <report.json> [--config FILE] [--trace FILE] [--verbose]
```

| Argument | Required | Meaning |
|---|---|---|
| `--repo` | yes | `https://github.com/<owner>/<repo>` (a trailing `.git` is fine) |
| `--output` | yes | Where to write the JSON report (checked for writability before the audit starts) |
| `--config` | no | A TOML file overriding default settings (see [Configuration](#configuration)) |
| `--trace` | no | Where to write the JSONL run trace (default: `logs/<owner>-<repo>-<timestamp>.jsonl`) |
| `--verbose` | no | Log progress (turns, tool calls) to stderr |

**Report format** (exactly these keys):

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "A credential is assigned as a literal value in source code",
      "evidence": "API_KEY = [REDACTED]",
      "recommendation": "Load the credential from an environment variable or a secrets manager and rotate the exposed value"
    }
  ]
}
```

- `severity` is one of `critical`, `high`, `medium`, `low`.
- `file`/`line` are `null` when they don't apply: `line` alone for file-level findings (a committed `.env`), and both for absence findings ("no CI configured").
- A clean repository gives `"violations": []`. Secret values are always redacted.

| Exit code | Meaning |
|---|---|
| `0` | Inspection completed and the report was written (also when there are no violations) |
| `1` | Runtime failure: repository unreachable, LLM provider failing after retries, output not writable. No report is written |
| `2` | Bad input or configuration: invalid URL, missing `OPENAI_API_KEY`/`OPENAI_MODEL`, invalid `--config`/`AUDITOR_*` value, malformed `.env` |
| `130` | Stopped with Ctrl+C. Nothing is written and the temporary clone is deleted |

Every fatal error prints a one-line `Error: …` message on stderr.

A run takes roughly 30–90 seconds: each model turn can take several seconds, and rate-limit retries add waits. Use `--verbose` to see "asking the model …", the reply time and any retry, so a slow turn doesn't look like a hang.

---

## How it works

1. **Validate** the URL (GitHub owner/name rules) and the output path.
2. **Open the repository read-only.** A hardened shallow `git clone --no-checkout` is the default: files are listed with `git ls-tree` and read with `git cat-file`, so nothing is ever checked out or executed. If git is missing or the clone fails, it falls back to the GitHub API.
3. **Start the loop.** The system prompt gives the scope, an investigation order, the evidence rules, a "do not report" list, the severity rubric and the **44-rule catalog**. Repository content is framed as untrusted data.
4. **The LLM calls tools.** Each call is validated, executed, size-capped and **secret-redacted**, and the result goes back to the model. Old tool results are compacted so long audits fit the context window.
5. **Deterministic checks are a tool too.** `audit_inspected_files` runs 11 conservative detectors on the files already read and returns candidate findings with exact evidence. The model decides which to confirm.
6. **Findings are reported one by one** with `report_finding`. Each is checked immediately and rejected with a reason if it fails:
   - the rule id must be in the catalog, with a matching category and a severity within one level of the default
   - the cited file and line must have been read, and a quoted line must really be there
   - detector-backed rules must match the detector's evidence
   - false-positive guards apply
   - root-cause duplicates are dropped

   A rejection is never fatal: the model can correct the finding or drop it.
7. **Budgets** (turns, tool calls, wall time) end with a short *finalize* step, so a large repository still produces a valid report.
8. The report is checked against the contract and written atomically. A redacted JSONL **trace** records every decision and tool call.

## Architecture

```
agent.py              CLI: arguments, settings, early checks, exit codes
│
├── auditor/loop.py        AuditAgent: the observe → reason → act loop, budgets, finalize
│   ├── llm_client.py      OpenAI-compatible client (stdlib HTTP), retries with backoff, no redirects
│   ├── prompts.py         system/finalize prompts loaded from auditor/prompts/*.txt
│   ├── context.py         compaction of old tool results
│   ├── findings.py        FindingStore: catalog, severity, grounding, guards, dedup
│   └── trace.py           redacted JSONL trace
│
├── tools.py               the inspection tools the model sees, their JSON schemas, strict argument checks
│                           (the report_finding / finish_audit schemas are in auditor/findings.py)
│   ├── auditor/repository.py   read-only sources: hardened clone, GitHub API fallback, local dir
│   ├── auditor/redaction.py    one secret matcher for every output
│   └── auditor/checks.py       11 data-driven detectors (audit_inspected_files)
│
├── auditor/rules.py       44-rule catalog loader and prompt rendering
├── auditor/report.py      contract check and atomic write;  auditor/schema.py  contract constants
├── auditor/settings.py    typed settings: defaults.toml < --config < AUDITOR_* env; .env loader
└── auditor/config/        defaults.toml (tunables), rules.toml (catalog, detectors, guards, redaction)
```

**Tools available to the model**

| Tool | Purpose |
|---|---|
| `list_files(path_prefix?, glob?)` | Tracked files on the default branch (reports truncation) |
| `read_file(path, start_line?, end_line?)` | A text file or a line range, numbered, secrets redacted |
| `search_code(pattern, regex?, path_glob?)` | Literal or regex search over redacted text, bounded; unsafe regexes refused |
| `git_log(max_commits?, path?)` | Recent commits: sha, date, message, changed files, author count (no emails) |
| `get_repo_metadata()` | Default branch, archived/fork/visibility, licence, size |
| `list_workflows()` | GitHub Actions files with their triggers, permissions and jobs located by line |
| `audit_inspected_files(rule_ids?)` | Deterministic detectors over the files already read |
| `report_finding(...)`, `finish_audit(summary)` | Submit one finding; end the audit |

---

## What it checks

The catalog has **44 rules** (35 specific plus a generic `-099` rule per category) across the nine assignment areas. Each rule has a default severity, an actionable recommendation and a list of acceptable alternatives.

| Area | Rule ids | Examples |
|---|---|---|
| Repository hygiene | HYG-001…005 | Missing README or `.gitignore`, `.gitignore` gaps, tracked build output/caches, large binaries |
| Dependency management | DEP-001…004 | Unpinned deps without a lockfile, missing lockfile (npm/yarn/pnpm/cargo/go/bundler/composer), conflicting specs, undeclared imports |
| Code quality | QUA-001…002 | No lint/format setup anywhere, bare `except`, committed `breakpoint()` |
| Testing | TST-001…003 | No tests, tests never run in CI, disabled test suites |
| GitHub Actions / CI | CI-001…009 | `permissions: write-all`, PR code run under `pull_request_target`, no CI, gates that can't fail, wrong triggers, skipped validation, script injection, branch-pinned actions, leaked secrets |
| Security | SEC-001…004 | Hard-coded credentials, private keys and token formats, committed secret files, TLS verification off / debug in production |
| Configuration | CFG-001…003 | Machine-specific paths, committed personal settings, hard-coded production DB hosts |
| Collaboration | COL-001…002 | Unsafe PR auto-merge or auto-approve, no review guidance in multi-contributor repos |
| Build and release | BLD-001…003 | Artefacts never built in CI, risky release/deploy triggers, `FROM :latest` / `curl \| sh` in Dockerfiles |

**Deterministic detectors** (11): SEC-001, SEC-002, SEC-003, CI-001, CI-002, CI-004, CI-007, CI-008, HYG-004, CFG-001, BLD-003. The other rules are judged by the model from what it has read.

## Avoiding false positives

Precision matters as much as recall, so the agent is built not to over-report:

- **Acceptable alternatives:** flake8 instead of ruff, `yarn.lock` instead of `package-lock.json`, tag- or SHA-pinned actions, `.env.example` placeholders, and so on. They are listed per rule and shown to the model.
- **Never inferred:** branch protection, required reviews or any remote setting. Claims about these are rejected.
- **Guarded paths:** test fixtures, vendored code, documentation snippets and template files are guarded for secret and config rules.
- **Healthy-repo traps are tested:** CI service passwords (`POSTGRES_PASSWORD: postgres`), safe `${{ github.event.pull_request.number }}` expressions, public `.env.production` values, i18n strings like `"password": "Password"`, Go `build/` source folders and devcontainer paths all produce zero findings in the evaluation set.
- **Grounding:** every located finding must quote a line the agent actually read.

---

## Configuration

All limits, timeouts and URLs live in [`auditor/config/defaults.toml`](auditor/config/defaults.toml); nothing is hard-coded. Override any value without editing that file:

| Method | Example |
|---|---|
| Environment variable `AUDITOR_<SECTION>_<KEY>` | `AUDITOR_AGENT_MAX_TOOL_CALLS=120`, `AUDITOR_REPOSITORY_BACKEND=api` |
| TOML file passed with `--config` | `[agent]` `max_turns = 40` |

Precedence: environment variable > `--config` file > defaults. Every value is type- and range-checked at startup; a misspelt key or `AUDITOR_*` variable stops the run with exit code 2.

| Section | Controls |
|---|---|
| `[github]` | API/raw/clone URLs, owner and name validation, timeouts, response limits |
| `[repository]` | Backend (`clone`/`api`), clone depth/timeout/size budgets, file and range read limits, git environment allowlist |
| `[search]` | Files scanned, matches, pattern and line caps, regex safety guards, searchable file types |
| `[git_log]` | Default and maximum commits, message length, files per commit |
| `[workflows]` | Workflow directory and scan caps |
| `[agent]` | Turns, tool calls, wall-clock and finalize budgets, tool-output size, finding limits |
| `[context]` | How many recent tool results stay in full, total context budget |
| `[trace]` | Trace on/off, directory, string cap |
| `[llm]` | Default base URL, timeout, temperature, retries and backoff |
| `[eval]` | Line tolerance for the evaluation harness |

The rule catalog, detector patterns, false-positive guards and redaction rules live in [`auditor/config/rules.toml`](auditor/config/rules.toml). They are deliberately **not** overridable by `--config`, so a user file can't weaken secret redaction.

---

## Safety

| Risk | Control |
|---|---|
| Changing the target | Read-only access; the clone lives in a private temp directory and is always deleted; nothing is written to GitHub |
| Running untrusted code | No checkout at all (`--no-checkout` + `cat-file`); hooks disabled, symlinks off, submodules and LFS skipped, HTTPS-only protocol, NTFS/HFS path protection, no credential helpers or prompts, argument lists only (no shell) |
| Resource exhaustion | Repository size, file-count and total-size budgets before reading; timeouts; caps on every read, search and tool output; turn, call and wall-time budgets |
| Path tricks | Repository-relative paths only; `..`, absolute paths, symlinks and junctions refused |
| Secret leakage | One redaction pass on every tool output, finding and trace line (assignments, connection strings, private keys, JWTs, provider token formats); API keys never logged; `GITHUB_TOKEN` only as a header; git gets an allowlisted environment, so the LLM key never reaches it; LLM requests never follow redirects |
| Prompt injection | Repository content is framed as data; tools only read; budgets and the tool set live in code; every finding is validated against what was actually read |
| Regex denial of service | Pattern and line caps; nested or alternating repetition is refused in regex search; all built-in patterns are linear-time |

---

## Testing and evaluation

```bash
python -m unittest discover -s tests -v     # 500+ unit/integration tests; no network, no API key needed
python -m eval.run                          # offline detector baseline over the labelled eval cases
python -m eval.run --live                   # the real agent on every eval case (uses your LLM quota)
```

- **Unit and integration tests** cover settings, redaction, the hardened clone (including a real-git check that the `file://` transport is refused), every tool, the LLM client's retry and redirect behaviour, the loop's budgets and finalize, finding validation, guards, dedup, the trace, the exit-code matrix, and an end-to-end CLI run on a fixture repository with a scripted model. **Guard tests** fail the build on any agent-framework import, numeric tunables in code, unsafe subprocess use, secrets in any output, or fixture names leaking into prompts or config.
- **Evaluation harness** (`eval/`): 18 labelled synthetic repositories, matching how the assignment is graded:
  - **A (obvious defects), 5 cases:** hard-coded DB password, `write-all`, no tests and no CI, tracked artefacts, committed `.env`.
  - **B (subtle CI/security/config defects), 7 cases:** `pull_request_target` checkouts, `continue-on-error` test gates, script injection, branch-pinned actions, `FROM :latest`, machine paths.
  - **C (healthy repositories with traps), 6 cases.**

  It reports precision, recall and F1 per case. The offline mode measures the deterministic detector layer alone: currently 13 of 15 labelled violations found with **0 false positives**. The 2 misses are absence findings (no tests, no CI) that only the model makes. These synthetic numbers are not a measure of accuracy on unseen repositories.

---

## Example output

[`sample_report.json`](sample_report.json) is the unedited result of running the agent on `https://github.com/gotcha/ipdb` with a tool-calling model: 6 model turns, 15 tool calls, 2 low-severity findings (an incomplete `.gitignore` and bare `except:` clauses). A repeatable local check on labelled repositories is `python -m eval.run --live`; in one live run on 8 of the 18 evaluation cases the agent found every labelled defect with no false positives on the four healthy repositories. That is a small synthetic sample, not a measure of accuracy on unseen repositories.

---

## Project structure

```
agent.py                    CLI entry point
tools.py                    inspection tools, schemas, dispatcher
auditor/                    agent package (loop, LLM client, findings, rules, checks, repository, redaction, settings, trace, report)
auditor/config/             defaults.toml, rules.toml
auditor/prompts/            system, finalize, start and nudge prompts
eval/                       evaluation harness and labelled cases (A/B/C)
tests/                      unit, integration and guard tests; fixtures/
pyproject.toml              project metadata (no dependencies); requirements.txt kept for tooling
.env.example                provider settings template
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is required` / `OPENAI_MODEL is required` (exit 2) | Put both in `.env` in the folder you run from, or export them |
| `LLM provider request failed with HTTP 429` (exit 1) | Free-tier rate limit after all retries; wait and rerun, or switch provider |
| `LLM provider redirected the request…` | Check `OPENAI_BASE_URL`; use the exact OpenAI-compatible URL from the table above |
| `GitHub denied the request…` | API rate limit; set `GITHUB_TOKEN`, or make sure `git` is installed so the clone path is used |
| `Repository is about N MiB, above repository.max_repo_size_mb` (exit 1) | Very large repository; raise `AUDITOR_REPOSITORY_MAX_REPO_SIZE_MB` or use `AUDITOR_REPOSITORY_BACKEND=api`. (Exceeding the file-count or total-size budget switches to the API automatically; `list_files` reports `fallback_reason`.) |
| `Unknown setting in environment: AUDITOR_…` (exit 2) | A misspelt `AUDITOR_*` variable; fix or unset it |
| The report has fewer findings than expected | See the trace (`logs/…jsonl`): rejected findings are listed with the reason |

---

## Assignment requirement map

| Requirement | Where |
|---|---|
| `python agent.py --repo … --output …`, unattended | `agent.py` |
| LLM-directed tool selection with follow-ups; no fixed checklist | `auditor/loop.py`, `auditor/prompts/system.txt` |
| Own loop, no agent frameworks | `auditor/loop.py` (stdlib only; guard test) |
| `list_files`, `read_file`, `search_code`, `git_log` (+ optional tools) | `tools.py`, `auditor/repository.py` |
| Nine inspection areas, evidence, severity, recommendations | `auditor/config/rules.toml`, `auditor/checks.py`, `auditor/findings.py` |
| Acceptable alternatives; no remote-setting inference | `rules.toml` alternatives and `[guards]`, `auditor/findings.py` |
| Exact JSON contract, empty array when clean, `null` file/line | `auditor/report.py`, `auditor/schema.py` |
| Non-zero exit with a useful message on fatal errors | `agent.py` (exit codes 1/2) |
| Target unchanged, no code execution, no secret exposure | `auditor/repository.py`, `auditor/redaction.py`, safety tests |
| Graceful handling of missing files, API errors, invalid input, timeouts | `auditor/repository.py`, `auditor/llm_client.py`, `tools.py` |
| `requirements.txt` / `pyproject.toml`, `.env.example`, `tests/` | repository root |
| Authentic `sample_report.json` | `sample_report.json`: unedited output of a real run on `github.com/gotcha/ipdb` |
