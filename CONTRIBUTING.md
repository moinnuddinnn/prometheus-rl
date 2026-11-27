# Contributing

Use Python 3.11+ and the project virtual environment. Start every block from the updated `main` branch:

```sh
git switch main
git pull --ff-only
git switch -c feature/01-mdp-core
# Implement one cohesive change with tests and mathematical documentation.
python -m pytest -q
ruff check .
git add src tests docs README.md
git diff --cached
git commit -m 'feat(mdp): implement finite MDP and Bellman operators'
git push -u origin HEAD
gh pr create --base main --fill
```

Review the diff and CI. Merge only after review and green checks, preferably using squash merge. Delete merged feature branches. Never force-push main. Do not use a later block before its prerequisites merge.

Use conventional commit prefixes (`feat`, `fix`, `test`, `docs`, `chore`, `ci`). Test terminal vs truncated transitions, shapes, reward signs and reproducibility whenever relevant. Record actual training metrics only; distinguish planned results from observations. Do not commit model weights or local experiment outputs. Exact experiment dependency versions will be recorded alongside each run once training is introduced.

Suggested GitHub ruleset (configure under Settings): target main, require pull requests and CI status checks; block force pushes and branch deletion. For a solo repository, do not require approval from another person unless one is available. Ruleset availability depends on repository visibility and plan.
