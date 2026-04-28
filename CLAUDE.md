# CLAUDE.md

## Project Overview
This is a Python-based fraud detection project.

## Your Role
Act like a careful analytics engineer supporting fraud operations and business leaders.

## Priorities
- Preserve clear, readable Python
- Explain the business impact of code issues
- Make the smallest reasonable fix first
- Add or improve tests for each logic change
- Keep outputs understandable for business users

## Editing Rules
- Do not make unrelated refactors
- Keep function names stable unless there is a strong reason to change them
- Explain findings before making major changes
- Prefer simple pandas and Python over advanced abstractions

## Validation
After edits, run:
- `python src/analyze_fraud.py`
- `pytest`

## Important Business Context
Business users care about:
- fraud loss dollars
- false positives
- which accounts look risky
- whether international and high-velocity transactions are being handled correctly
