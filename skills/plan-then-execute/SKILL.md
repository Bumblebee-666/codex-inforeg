---
name: plan-then-execute
description: Require a plan-first workflow for code edits. Use when the user wants Codex to present a concise plan and wait for approval before making any code changes, while allowing read-only analysis without a plan. If the user explicitly asks to skip planning, allow direct execution.
---

# Plan Then Execute

## Overview

Require explicit approval before any code modifications. Use a short plan (2-6 steps), then wait for confirmation.

## Workflow

1. Determine whether the request requires code changes.
2. If code changes are needed, present a plan that names files, key edits, and any tests.
3. Ask for approval (e.g., "Proceed?"). Do not modify files until approval is explicit.
4. After approval, execute the plan and report results.
5. If the user rejects or revises the plan, update it and ask again.

## Guardrails

- Do not run tools that write files (e.g., apply_patch or editing commands) before approval.
- Read-only actions (searching, reviewing, explaining) can proceed without a plan.
- If the user explicitly asks to skip the plan, allow direct execution for that request.
