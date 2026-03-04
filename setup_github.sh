#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ARIA — GitHub Push Setup Script
# Run this once from the repo root to initialise and push to your GitHub repo.
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "  ARIA — GitHub Setup"
echo "  ─────────────────────────────────────"
echo ""

# 1. Initialise git if not already done
if [ ! -d ".git" ]; then
    git init
    echo "  ✅ Git repository initialised"
else
    echo "  ✅ Git repository already exists"
fi

# 2. Set up gitignore
if [ ! -f ".gitignore" ]; then
    echo "  ⚠️  .gitignore not found — make sure you have the full package"
    exit 1
fi

# 3. Stage all files
git add .
echo "  ✅ Files staged"

# 4. Initial commit
git commit -m "feat: ARIA v1.1.0 — Zero-Trust Self-Aware Self-Improving Multi-Agent Framework

- Zero-trust permission engine with 17 capability types and audit trail
- Self-consciousness module with health grading (A-F) and source integrity
- Security self-scanner: two-pass analysis across source files
- Self-patcher with automatic rollback on verification failure
- Self-improvement engine: mistake learning, capability gap detection
- Capability synthesis: ARIA generates new agents to fill gaps
- Empirical routing weights blending semantic fingerprinting with real data
- Full test suite: permissions, improvement, scanner, orchestrator
- GitHub Actions CI pipeline
- Architecture documentation" 2>/dev/null || echo "  ✅ Already committed (no changes)"

# 5. Set main branch
git branch -M main 2>/dev/null || true
echo "  ✅ Branch set to main"

echo ""
echo "  Now add your GitHub remote and push:"
echo ""
echo "    git remote add origin https://github.com/anarudhan/aria.git"
echo "    git push -u origin main"
echo ""
echo "  Or with SSH:"
echo "    git remote add origin git@github.com:anarudhan/aria.git"
echo "    git push -u origin main"
echo ""
echo "  ─────────────────────────────────────"
echo "  After pushing, connect your repo to Claude:"
echo "  Claude Project → Settings → Connected Sources → Add GitHub repo"
echo "  ─────────────────────────────────────"
echo ""
