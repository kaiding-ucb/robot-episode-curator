# Robot Episode Curator

Easily identify and curate outlier episodes in Lerobot datasets.

Uses [LeRobot](https://github.com/huggingface/lerobot) for dataset format, [Rerun](https://rerun.io/) for native multi-modal playback, and [Gemini](https://aistudio.google.com/) for video analysis enrichment.

[![Rerun 0.28.2](https://img.shields.io/badge/Rerun-0.28.2-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVY5LjMwOTQyVjcuNTAzNEM2LjcyMDkyIDguNjcyNTJWMi45OTY1Nkw0LjM0NzIzIDYuMzIxMDlMNC4zNDcxNyAxMi4wOTJMNS4wMTA5NCAxMi40NjgzTDUuMDExMjkgNi42OTc1NFoJOSA1LjczMzQxTDkuNTQ1ODQgOC4yOTIwNkw3LjA4ODg2IDkuNjg1NjRMNi40MjU0MSA5LjMwOTQyVjcuNTAzNEM2Ljc5MDMyIDcuMjk2NDkgOS41NDU4OCA1LjcyNzE0IDkuNTQ1NzkgNS43MzM0MVoiIGZpbGw9IndoaXRlIi8+CjwvZz4KPC9zdmc+Cg==)](https://rerun.io/)
[![🤗 Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/jacob314159/robot-episode-curator)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

<p align="center">
  <img src="docs/media/viewer.gif" alt="Robot Episode Curator demo" />
</p>

## Demo

Hosted demo on Hugging Face Spaces:

<a href='https://huggingface.co/spaces/jacob314159/robot-episode-curator'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

## Quick Start (5 Minutes)

```bash
git clone https://github.com/kaiding-ucb/robot-episode-curator.git
cd robot-episode-curator
make install
cp .env.example .env    # paste HF_TOKEN; GEMINI_API_KEY only if you want AI analysis
make dev
```

Open the printed frontend URL in your browser. Defaults are `http://localhost:3000` (frontend) and `http://localhost:8000` (backend) — pass `PORT=<backend>` and/or `FRONTEND_PORT=<frontend>` to use any free pair, e.g. `PORT=8765 FRONTEND_PORT=3765 make dev`. The frontend proxies `/api/*` to whatever `PORT` you choose, so no other config changes are needed.

Get tokens: [HuggingFace](https://huggingface.co/settings/tokens) (read scope is enough) · [Gemini](https://aistudio.google.com/apikey)

## End-to-End Analysis Workflow

![End-to-end analysis workflow](docs/images/workflow.png)

The pipeline runs per-episode **phase segmentation**, statistical flag detection (duration, cycle, envelope, shape outliers), and **Bayesian variance clustering**. Representative clips from each cluster are sent to **Gemini** for characterization and flag enrichment. Output is a deck of cluster cards and flagged-episode cards rendered next to the data.


## Supported Datasets

Out-of-the-box adapters for:

| Format                     | Examples                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| LeRobot v3 (parquet + mp4) | `lerobot/libero_*`, `lerobot/aloha_*`, `lerobot/droid_100`, `lerobot/umi_cup_in_the_wild` |

Add any HuggingFace Lerobot dataset via **+ Add Dataset** in the sidebar — the probe step auto-detects format.
