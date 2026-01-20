# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
We're developing a data viewer to analyze and evaluate quality of Robotics datasets:

## Key Robotics Datasets and Official Links

My Huggingface token is REDACTED-HF-TOKEN for accessing these datasets

- **LIBERO**: [github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
    - **LIBERO-PRO**: [github.com/Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)
- **Open X Embodiment**
    - **Fractal**: [github.com/google-research/robotics_transformer](https://github.com/google-research/robotics_transformer)
    - **Bridge**: [github.com/rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2)
- **Ego4D**: [github.com/facebookresearch/Ego4d](https://github.com/facebookresearch/Ego4d)
- **Egocentric-10K**: [huggingface.co/datasets/builddotai/Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K)
- **10Kh RealOmni-Open Dataset**: [huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData](https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData)

---

## Golden rules ##
AI should do:
- This project will use git worktrees for parallel feature developments. Always follow git worktree instructions outlined below
- Always follow TDD instructions as this project will perform test-driven development
- Always use real data from Key Robotics Datasets and Official Links for tests
- When using local servers, always check which ports are available first
- If needing api keys, pause and ask users to provide instead of using a workaround
- When asked to push to git repo, use gh cli, and set up tagging that is consistent with changelog.md for future version control

AI must NOT do:
- Never modify CLAUDE.md (this file). This file is for humans-only 
- Never use mock or simulate data in TDD
- Never declare tests pass while using simulate or mock data
- Never take shortcuts on tests, meaning declare tests pass without meeting the exact success criteria
- Don't use print statements unless it's a testing cell or explitly asked by users

---

### Git Worktree Instructions

This project uses **git worktrees** for parallel feature development:

---

**Worktree 1:** `data_viewer_dataset`  
**Branch:** `feature/dataset-navigation`

- **Scope:** Dataset download, selector, episode display
- **Owns:**
  - `LeftSidebar.tsx`
  - `DatasetBrowser.tsx`
  - `EpisodeViewer.tsx`
  - `DataManager.tsx`
  - `Modals.tsx`
- **Backend:**
  - `datasets.py`
  - `episodes.py`
  - `downloads.py`
  - `downloaders/*`

---

**Worktree 2:** `data_viewer_quality`  
**Branch:** `feature/quality-metrics`

- **Scope:** Quality metrics computation and display
- **Owns:**
  - `RightSidebar.tsx`
  - `QualityPanel.tsx`
  - `EnhancedTimeline.tsx`
  - `useQuality.ts`
- **Backend:**
  - `quality.py`
  - `quality/*`

---

**Shared files (coordinate changes):**

- `page.tsx` — State management only, imports layout components
- `frontend/src/types/api.ts` — Additive only, use section comments
- `MainContent.tsx` — Minimal changes expected

---

**Ports:**  
- Main: `3000/8000`
- Dataset worktree: `3001/8001`
- Quality worktree: `3002/8002`

---

## Git Worktree Workflow

**Before starting work (sync with main):**
```
git fetch origin
git rebase main
```

**After completing a feature (from your worktree):**
```
git add .
git commit -m "feat: description of changes"
git push origin <branch-name>
```

**Integration (from main worktree /Users/kai/Documents/data_viewer):**
```
git checkout main
git merge feature/dataset-navigation
git merge feature/quality-metrics
# Resolve any conflicts in shared files
git push origin main
```

**Useful commands:**
```
git worktree list              # Show all worktrees
git worktree remove <path>     # Remove a worktree when done
git worktree prune             # Clean up stale worktree references
```

**Testing before merge:**  
Always run `npx playwright test e2e/user-flows.spec.ts` from the frontend directory to verify no regressions.

---