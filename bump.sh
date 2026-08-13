#!/usr/bin/env bash
# Bump polarscope's version, commit, and tag.
# Usage:  ./bump.sh patch | minor | major

set -euo pipefail

PART="${1:-patch}"
INIT="polarscope/__init__.py"

CUR=$(grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' "$INIT" | head -1)
IFS='.' read -r MAJOR MINOR PATCH <<< "$CUR"

case "$PART" in
  major) MAJOR=$((MAJOR+1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR+1)); PATCH=0 ;;
  patch) PATCH=$((PATCH+1)) ;;
  *) echo "usage: ./bump.sh [patch|minor|major]"; exit 1 ;;
esac

NEW="$MAJOR.$MINOR.$PATCH"
ESC="${CUR//./\\.}"
echo "$CUR  ->  $NEW"

sed -i '' "s/$ESC/$NEW/g" "$INIT" README.md

git add -A
git commit -m "release: v$NEW"
git tag -a "v$NEW" -m "v$NEW"

echo "tagged v$NEW"
