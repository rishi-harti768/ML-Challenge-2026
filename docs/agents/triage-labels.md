# Triage Labels

Use these five canonical triage roles as the values in each issue's `Status:` line.

| Role               | Label string      | Meaning                                  |
| ------------------ | ------------------ | ----------------------------------------- |
| `needs-triage`      | `needs-triage`      | Maintainer needs to evaluate this issue   |
| `needs-info`        | `needs-info`        | Waiting on reporter for more information  |
| `ready-for-agent`   | `ready-for-agent`   | Fully specified, ready for an AFK agent   |
| `ready-for-human`   | `ready-for-human`   | Requires human implementation             |
| `wontfix`           | `wontfix`           | Will not be actioned                      |

When a skill tells you to apply a triage role (e.g. "apply the AFK-ready triage label"), set the issue's `Status:` line to the matching string from this table.
