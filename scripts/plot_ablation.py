#!/usr/bin/env python3
"""
从 TensorBoard event 文件中读取消融实验数据，生成出版质量的 PDF 曲线图。

用法:
    python scripts/plot_ablation.py

输出:
    ablation/figures/*.pdf  (共 6 张图)
"""

import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# TensorBoard 读取
# ---------------------------------------------------------------------------

def read_scalar(event_dir: str, tag: str = "eval/avg_snippet_relevance"):
    """从 TensorBoard event 文件中读取标量数据，返回 (steps, values)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(event_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)


# ---------------------------------------------------------------------------
# 每个消融实验组的定义
#   key    → 输出文件名 (不含 .pdf)
#   title  → 图表标题（可留空，用于 caption 中说明）
#   runs   → {相对路径: 图例标签}
#   styles → {标签: line_kwargs} 可选，指定特定曲线的样式
# ---------------------------------------------------------------------------

ABLATION_GROUPS = {
    "new_topk": {
        "runs": {
            "ablation/new_topk/2": "k = 2",
            "ablation/new_topk/3": "k = 3",
            "ablation/new_topk/5": "k = 5",
        },
    },
    "topk": {
        "runs": {
            "ablation/topk/2": "k = 2",
            "ablation/standard": "k = 3 (standard)",
            "ablation/topk/5": "k = 5",
        },
        "styles": {
            "k = 3 (standard)": {"linestyle": "--", "linewidth": 1.5},
        },
    },
    "entropy": {
        "runs": {
            "ablation/entropy/0": "λ = 0",
            "ablation/entropy/0001": "λ = 0.001",
            "ablation/entropy/01": "λ = 0.01",
            "ablation/standard": "λ = 0.01 (standard)",
        },
        "styles": {
            "λ = 0.01 (standard)": {"linestyle": "--", "linewidth": 1.5},
        },
    },
    "refresh_index": {
        "runs": {
            "ablation/refresh_index/50": "50 steps",
            "ablation/refresh_index/200": "200 steps",
            "ablation/refresh_index/never": "never",
            "ablation/standard": "500 steps (standard)",
        },
        "styles": {
            "500 steps (standard)": {"linestyle": "--", "linewidth": 1.5},
        },
    },
    "algorithm": {
        "runs": {
            "ablation/standard": "EMA (standard)",
            "ablation/algorithmn/GRPO": "GRPO",
            "ablation/algorithmn/global_penalty": "Global Penalty",
        },
        "styles": {
            "EMA (standard)": {"linestyle": "--", "linewidth": 1.5},
        },
    },
    "model": {
        "runs": {
            "ablation/model/3b": "3B",
            "ablation/model/7b": "7B",
            "ablation/standard": "14B (standard)",
        },
        "styles": {
            "14B (standard)": {"linestyle": "--", "linewidth": 1.5},
        },
    },
}

# ColorBrewer 调色板 (Set1 和 Dark2 混合使用)
COLOR_CYCLE = [
    "#E41A1C",  # 红
    "#377EB8",  # 蓝
    "#4DAF4A",  # 绿
    "#984EA3",  # 紫
    "#FF7F00",  # 橙
    "#A65628",  # 棕
]


def plot_group(ax, group_cfg: dict, root: str):
    """在给定 ax 上绘制一个消融实验组的所有曲线。"""
    labels = []
    for i, (rel_path, label) in enumerate(group_cfg["runs"].items()):
        event_dir = os.path.join(root, rel_path)
        if not os.path.isdir(event_dir):
            print(f"  [WARN] 目录不存在，跳过: {event_dir}")
            continue

        steps, values = read_scalar(event_dir)
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]

        # 默认样式 vs 自定义样式
        style = {"color": color, "linestyle": "-", "linewidth": 1.2}
        custom_styles = group_cfg.get("styles", {}).get(label, {})
        style.update(custom_styles)

        ax.plot(steps, values, label=label, **style)
        labels.append(label)

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Avg Snippet Relevance", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # x 轴从 0 开始
    ax.set_xlim(left=0)
    # y 轴留 5% 余量
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin - 0.02), ymax + 0.02)

    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")


def main():
    root = str(Path(__file__).parent.parent)  # 项目根目录
    out_dir = os.path.join(root, "ablation", "figures")
    os.makedirs(out_dir, exist_ok=True)

    for group_key, group_cfg in ABLATION_GROUPS.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_group(ax, group_cfg, root)

        out_path = os.path.join(out_dir, f"{group_key}.pdf")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ {out_path}")

        # 同时输出一份 PNG 预览
        preview_path = os.path.join(out_dir, f"{group_key}.png")
        fig.savefig(preview_path, dpi=120)
        plt.close(fig)

    print(f"\n全部完成 → {out_dir}/")


if __name__ == "__main__":
    main()
