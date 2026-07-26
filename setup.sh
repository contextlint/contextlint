#!/usr/bin/env bash
# One-command setup. Usage:  ./setup.sh <github-org-or-user> [your-legal-name]
#
# Substitutes your org name everywhere, commits, and pushes. Idempotent —
# safe to re-run. Does not touch anything outside this directory.
set -euo pipefail

ORG="${1:-}"
LEGAL_NAME="${2:-Rayan El Fayoumi}"

if [[ -z "$ORG" ]]; then
  echo "usage: ./setup.sh <github-org-or-user> [your-legal-name]" >&2
  echo "example: ./setup.sh contextlint" >&2
  exit 2
fi

REPO="https://github.com/${ORG}/contextlint"
PAGES="https://${ORG}.github.io/contextlint/"

echo "→ org:   $ORG"
echo "→ repo:  $REPO"
echo "→ pages: $PAGES"
echo "→ name:  $LEGAL_NAME"
echo

# 1. Substitute placeholders
FILES=$(git ls-files '*.md' '*.html' '*.py' '*.toml' '*.yml' 2>/dev/null || true)
for f in $FILES; do
  [[ -f "$f" ]] || continue
  sed -i.bak \
    -e "s#https://github.com/REPO_PLACEHOLDER.git#${REPO}.git#g" \
    -e "s#https://github.com/REPO_PLACEHOLDER#${REPO}#g" \
    -e "s#REPO_PLACEHOLDER#${ORG}/contextlint#g" \
    -e "s#<REPO>#${REPO}#g" \
    -e "s#<PAGES>#${PAGES}#g" \
    -e "s#Rayan El Fayoumi#${LEGAL_NAME}#g" \
    "$f"
  rm -f "$f.bak"
done
echo "✓ placeholders substituted"

# 2. Sanity check: nothing left unfilled
LEFT=$(grep -rl "REPO_PLACEHOLDER\|<REPO>\|<PAGES>\|EMAIL_PLACEHOLDER" --include='*.md' --include='*.html' . 2>/dev/null || true)
if [[ -n "$LEFT" ]]; then
  echo "! still contains placeholders:" >&2
  echo "$LEFT" >&2
  exit 1
fi
echo "✓ no placeholders remain"

# 3. Tests must pass before anything is published
if command -v python3 >/dev/null; then
  python3 -m pytest -q >/dev/null 2>&1 && echo "✓ tests pass" \
    || { echo "! tests failed — refusing to push" >&2; exit 1; }
fi

# 4. Commit and push
git add -A
git commit -q -m "configure for ${ORG}" || echo "  (nothing new to commit)"
git branch -M main 2>/dev/null || true

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/${ORG}/contextlint.git"
else
  git remote add origin "https://github.com/${ORG}/contextlint.git"
fi

echo
echo "→ pushing..."
git push -u origin main

cat <<DONE

──────────────────────────────────────────────────────────
Pushed. Two clicks left, both on GitHub:

1. Settings → Pages → Source: "Deploy from a branch"
   Branch: main    Folder: /docs    → Save
   Your page goes live at: ${PAGES}

2. The repo's ⚙ (top right of the code view) → Topics, paste:
   llm tokens prompt-engineering cost-optimization finops cli

Then tell me and I'll queue the first post.
──────────────────────────────────────────────────────────
DONE
