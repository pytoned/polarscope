#!/usr/bin/env bash
# Full release. Usage: ./release.sh patch|minor|major

set -euo pipefail

# Refuse to run with uncommitted work, so the bump commit stays clean.
if [ -n "$(git status --porcelain)" ]; then
  echo "Uncommitted changes. Commit your work first, then release."
  git status --short
  exit 1
fi

uv run --frozen pytest -q                      # stop here if anything is broken

./bump.sh "${1:-patch}"               # edits files, commits, tags
TAG=$(git describe --tags --abbrev=0)  # read back the tag just made

git push --follow-tags                # commits + tag -> GitHub
gh release create "$TAG" --generate-notes

echo ""
echo "$TAG is on GitHub. Actions is now publishing to PyPI."
echo "Follow it with:  gh run watch"
