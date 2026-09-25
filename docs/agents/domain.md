# Domain Docs

Rules for consuming this repo's domain documentation.

## Before exploring the codebase, read these

- Read **`CONTEXT.md`** at the repo root.
- Read **`docs/adr/`**: read ADRs that touch the area you're about to work in.

If these files don't exist, proceed silently — don't flag their absence, don't suggest creating them upfront. The `/domain-modeling` skill creates them lazily when terms or decisions actually get resolved.

## File structure

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

## Use the glossary's vocabulary

When naming a domain concept in your output (an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If a concept you need isn't in the glossary yet, treat it as a signal: either you're inventing language the project doesn't use (reconsider), or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding it:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
