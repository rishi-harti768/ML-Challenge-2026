# Issue Tracker: Local Markdown

Store issues and specs for this repo as markdown files under `.scratch/`.

## Conventions

- Use one directory per feature: `.scratch/<feature-slug>/`
- Write the spec to `.scratch/<feature-slug>/spec.md`
- Write one file per ticket to `.scratch/<feature-slug>/issues/<NN>-<slug>.md`. Number tickets from `01`. Never combine tickets into a single file.
- Record triage state as a `Status:` line near the top of each issue file. Use the label strings from `docs/agents/triage-labels.md`.
- Append comments and conversation history to the bottom of the file under a `## Comments` heading.

## When told to "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`. Create the directory first if it doesn't exist.

## When told to "fetch the relevant ticket"

Read the file at the referenced path. Expect the user to pass the path or issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md`. Maintain the Notes / Decisions-so-far / Fog sections in its body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`. Put the question in the body. Record ticket type on a `Type:` line (`research`/`prototype`/`grilling`/`task`) and state on a `Status:` line (`claimed`/`resolved`).
- **Blocking**: record blockers on a `Blocked by: NN, NN` line near the top. Treat a ticket as unblocked only when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed. Pick the lowest number first.
- **Claim**: before starting work, set `Status: claimed` and save.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
