import os
import json
import math
import html
from pathlib import Path
from typing import Dict, Any, List, Optional


METHOD_COLORS = {
    "m_full_finetuned": "#d9534f",          # Crimson Red (Memorized baseline)
    "m_unlearned_fisher": "#f0ad4e",        # Amber (Fisher)
    "m_unlearned_tv": "#0284c7",            # Sky Blue (Task Vectors)
    "m_unlearned_generic_tier2": "#16a34a", # Emerald Green (Generic-Label Tier 2)
    "m_unlearned_gd": "#9333ea",            # Purple (Gradient Difference)
}


def escape_xml(text: str) -> str:
    """Escapes XML/SVG special characters."""
    return html.escape(str(text), quote=True)


def create_svg_scatter_chart(
    title: str,
    x_label: str,
    y_label: str,
    points: List[Dict[str, Any]],
    width: int = 820,
    height: int = 500,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    draw_frontier: bool = True,
) -> str:
    """
    Renders an elegant, publication-grade SVG scatter plot with grid lines,
    Pareto boundary lines, legend, and customizable coordinate ranges without external dependencies.
    """
    margin_l = 80
    margin_r = 220
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def scale_x(x):
        ratio = (x - x_min) / max(x_max - x_min, 1e-9)
        return margin_l + ratio * plot_w

    def scale_y(y):
        ratio = (y - y_min) / max(y_max - y_min, 1e-9)
        return margin_t + plot_h - (ratio * plot_h)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        'style="background-color:#ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        # Background card
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        # Title
        f'<text x="{margin_l + plot_w/2}" y="34" text-anchor="middle" font-size="17" font-weight="bold" fill="#1e293b">{escape_xml(title)}</text>',
    ]

    # Grid & Axes
    num_ticks = 5
    for i in range(num_ticks + 1):
        val_x = x_min + i * (x_max - x_min) / num_ticks
        px = scale_x(val_x)
        svg_parts.append(f'<line x1="{px}" y1="{margin_t}" x2="{px}" y2="{margin_t + plot_h}" stroke="#f1f5f9" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{px}" y="{margin_t + plot_h + 20}" text-anchor="middle" font-size="11" fill="#64748b">{val_x:.2f}</text>')

    for i in range(num_ticks + 1):
        val_y = y_min + i * (y_max - y_min) / num_ticks
        py = scale_y(val_y)
        svg_parts.append(f'<line x1="{margin_l}" y1="{py}" x2="{margin_l + plot_w}" y2="{py}" stroke="#f1f5f9" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{margin_l - 12}" y="{py + 4}" text-anchor="end" font-size="11" fill="#64748b">{val_y:.2f}</text>')

    # Axes lines
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')

    # Axis Labels
    svg_parts.append(f'<text x="{margin_l + plot_w/2}" y="{height - 15}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{escape_xml(x_label)}</text>')
    svg_parts.append(f'<text transform="rotate(-90)" x="-{margin_t + plot_h/2}" y="24" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{escape_xml(y_label)}</text>')

    # Draw Pareto boundary line if requested
    if draw_frontier and len(points) >= 2:
        # Sort points by x ascending
        sorted_pts = sorted(points, key=lambda p: p["x"])
        frontier_pts = []
        max_y_so_far = -1e9
        # Compute upper hull from right to left
        for pt in reversed(sorted_pts):
            if pt["y"] >= max_y_so_far:
                frontier_pts.append(pt)
                max_y_so_far = pt["y"]
        frontier_pts = sorted(frontier_pts, key=lambda p: p["x"])

        if len(frontier_pts) >= 2:
            path_d = []
            for i, fp in enumerate(frontier_pts):
                cmd = "M" if i == 0 else "L"
                path_d.append(f"{cmd} {scale_x(fp['x']):.1f} {scale_y(fp['y']):.1f}")
            svg_parts.append(f'<path d="{" ".join(path_d)}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4" fill="none" />')

    # Legend container
    legend_x = margin_l + plot_w + 16
    legend_y = margin_t + 10
    legend_h = max(110, len(points) * 28 + 36)
    svg_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="190" height="{legend_h}" fill="#f8fafc" rx="8" stroke="#e2e8f0" stroke-width="1" />')
    svg_parts.append(f'<text x="{legend_x + 14}" y="{legend_y + 20}" font-size="12" font-weight="bold" fill="#334155">Algorithms</text>')

    # Plot Points and Legend Entries
    for idx, pt in enumerate(points):
        px = scale_x(pt["x"])
        py = scale_y(pt["y"])
        color = pt.get("color", "#3182ce")
        name = pt.get("label", f"Model {idx+1}")

        # Drop shadow / glow
        svg_parts.append(f'<circle cx="{px}" cy="{py}" r="9" fill="{color}" opacity="0.20" />')
        # Main point
        svg_parts.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{color}" stroke="#ffffff" stroke-width="1.8" />')
        # Point text label with subtle offset
        svg_parts.append(f'<text x="{px + 10}" y="{py - 8}" font-size="11" font-weight="600" fill="#1e293b">{escape_xml(name)}</text>')

        # Legend entry
        entry_y = legend_y + 40 + idx * 24
        svg_parts.append(f'<circle cx="{legend_x + 18}" cy="{entry_y - 4}" r="5" fill="{color}" />')
        svg_parts.append(f'<text x="{legend_x + 32}" y="{entry_y}" font-size="11" fill="#475569">{escape_xml(name)}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def plot_pareto_frontier(results: Dict[str, Any], output_path: Path):
    """
    Generates Pareto Frontier Plot:
    X-axis: Unlearning Efficacy = 1.0 - Forget ROUGE-L (higher is cleaner erasure)
    Y-axis: Model Utility = Retain ROUGE-L (higher is better retention)
    """
    points = []
    for model_key, report in results.items():
        eval_res = report["evaluation"]
        forget_rouge = eval_res["forget"]["rougeL"]
        retain_rouge = eval_res["retain"]["rougeL"]

        points.append({
            "label": report.get("display_name", model_key),
            "x": 1.0 - forget_rouge,  # Erasure Efficacy (0 to 1)
            "y": retain_rouge,         # Retain Retention (0 to 1)
            "color": METHOD_COLORS.get(model_key, "#4a5568"),
        })

    svg = create_svg_scatter_chart(
        title="Pareto Frontier: Unlearning Efficacy vs. Knowledge Retention",
        x_label="Erasure Efficacy: 1 - Forget ROUGE-L (Higher = Cleaner Forget)",
        y_label="Model Utility: Retain ROUGE-L (Higher = Better Retention)",
        points=points,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        draw_frontier=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"  [✓] Pareto chart written to: {output_path}")


def plot_privacy_utility_tradeoff(results: Dict[str, Any], output_path: Path):
    """
    Generates Privacy Trade-off Plot:
    X-axis: MIA ROC-AUC (Optimal is 0.50)
    Y-axis: Retain ROUGE-L (Optimal is 1.0)
    """
    points = []
    for model_key, report in results.items():
        eval_res = report["evaluation"]
        audit_res = report["privacy_audit"]

        points.append({
            "label": report.get("display_name", model_key),
            "x": audit_res["loss_threshold_mia_auc"],
            "y": eval_res["retain"]["rougeL"],
            "color": METHOD_COLORS.get(model_key, "#4a5568"),
        })

    svg = create_svg_scatter_chart(
        title="Privacy vs. Utility: Membership Inference AUC vs. Retain ROUGE-L",
        x_label="MIA ROC-AUC Attack Score (0.50 = Ideal Privacy, 1.00 = Complete Leak)",
        y_label="Model Utility: Retain ROUGE-L (Higher = Better Retention)",
        points=points,
        x_min=0.2,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        draw_frontier=False,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"  [✓] Privacy trade-off chart written to: {output_path}")


def plot_loss_distributions(results: Dict[str, Any], output_path: Path):
    """
    Generates cross-entropy loss comparison bars across Forget, Retain, and Unseen splits.
    Includes exact value annotations above each bar.
    """
    width = 860
    height = 440
    margin_l = 80
    margin_r = 40
    margin_t = 60
    margin_b = 60

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    model_keys = list(results.keys())
    if not model_keys:
        return

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        'style="background-color:#ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="17" font-weight="bold" fill="#1e293b">Cross-Entropy Loss Across TOFU Splits (Forget / Retain / Unseen)</text>',
    ]

    max_loss = max(
        max(
            r["evaluation"]["forget"]["loss"],
            r["evaluation"]["retain"]["loss"],
            r["evaluation"]["unseen"]["loss"],
        )
        for r in results.values()
    )
    max_loss = math.ceil(max(max_loss * 1.25, 4.0))

    num_models = len(model_keys)
    group_width = plot_w / num_models
    bar_width = (group_width * 0.72) / 3

    colors = {"forget": "#ef4444", "retain": "#3b82f6", "unseen": "#8b5cf6"}

    # Y Grid
    for i in range(5):
        val = i * (max_loss / 4)
        py = margin_t + plot_h - (val / max_loss) * plot_h
        svg_parts.append(f'<line x1="{margin_l}" y1="{py}" x2="{margin_l + plot_w}" y2="{py}" stroke="#f1f5f9" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{margin_l - 10}" y="{py + 4}" text-anchor="end" font-size="11" fill="#64748b">{val:.1f}</text>')

    # Bars and Value Labels
    for idx, mk in enumerate(model_keys):
        r = results[mk]["evaluation"]
        group_center = margin_l + idx * group_width + group_width / 2

        split_vals = [
            ("forget", r["forget"]["loss"], -bar_width),
            ("retain", r["retain"]["loss"], 0),
            ("unseen", r["unseen"]["loss"], bar_width),
        ]

        for s_name, val, offset in split_vals:
            bx = group_center + offset - bar_width / 2
            bh = max(2.0, (val / max_loss) * plot_h)
            by = margin_t + plot_h - bh
            svg_parts.append(f'<rect x="{bx}" y="{by}" width="{bar_width*0.88:.1f}" height="{bh:.1f}" fill="{colors[s_name]}" rx="3" />')
            # Text label above bar
            svg_parts.append(f'<text x="{bx + bar_width*0.44}" y="{by - 4}" text-anchor="middle" font-size="9" fill="#475569">{val:.2f}</text>')

        # Model Label
        d_name = results[mk].get("display_name", mk).split("(")[0].strip()
        svg_parts.append(f'<text x="{group_center}" y="{margin_t + plot_h + 22}" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">{escape_xml(d_name)}</text>')

    # Axes
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')

    # Legend
    legend_y = margin_t + 10
    svg_parts.append(f'<rect x="{margin_l + 10}" y="{legend_y}" width="12" height="12" fill="{colors["forget"]}" rx="2" />')
    svg_parts.append(f'<text x="{margin_l + 28}" y="{legend_y + 10}" font-size="11" fill="#475569">Forget Loss (Erased)</text>')

    svg_parts.append(f'<rect x="{margin_l + 175}" y="{legend_y}" width="12" height="12" fill="{colors["retain"]}" rx="2" />')
    svg_parts.append(f'<text x="{margin_l + 193}" y="{legend_y + 10}" font-size="11" fill="#475569">Retain Loss (Preserved)</text>')

    svg_parts.append(f'<rect x="{margin_l + 355}" y="{legend_y}" width="12" height="12" fill="{colors["unseen"]}" rx="2" />')
    svg_parts.append(f'<text x="{margin_l + 373}" y="{legend_y + 10}" font-size="11" fill="#475569">Unseen Control Loss</text>')

    svg_parts.append('</svg>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    print(f"  [✓] Loss distributions chart written to: {output_path}")


def generate_html_dashboard(results: Dict[str, Any], output_path: Path):
    """Generates a responsive HTML dashboard displaying leaderboard and charts together."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Aparichit: Machine Unlearning & Privacy Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #0f172a; margin-bottom: 8px; }}
        p.subtitle {{ color: #64748b; margin-top: 0; margin-bottom: 24px; }}
        .card {{ background: #ffffff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 24px; border: 1px solid #e2e8f0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; color: #475569; font-weight: 600; font-size: 13px; }}
        td {{ font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
        @media (min-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        img.chart {{ width: 100%; height: auto; border-radius: 8px; display: block; border: 1px solid #f1f5f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Aparichit: Machine Unlearning & Privacy Dashboard</h1>
        <p class="subtitle">Standardized TOFU Empirical Benchmark Evaluation</p>
        
        <div class="card">
            <h2>Publication Visualizations</h2>
            <div class="grid">
                <div>
                    <h3>1. Pareto Frontier: Erasure vs. Retention</h3>
                    <img class="chart" src="pareto_frontier.svg" alt="Pareto Frontier">
                </div>
                <div>
                    <h3>2. Privacy vs. Utility Tradeoff</h3>
                    <img class="chart" src="privacy_utility_tradeoff.svg" alt="Privacy Utility Tradeoff">
                </div>
                <div>
                    <h3>3. Cross-Entropy Loss Distributions</h3>
                    <img class="chart" src="loss_distributions.svg" alt="Loss Distributions">
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [✓] Interactive HTML dashboard written to: {output_path}")


def generate_all_visualizations(results: Dict[str, Any], output_dir: Path):
    """Generates all publication vector charts and interactive HTML dashboard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating publication-quality charts...")
    plot_pareto_frontier(results, output_dir / "pareto_frontier.svg")
    plot_privacy_utility_tradeoff(results, output_dir / "privacy_utility_tradeoff.svg")
    plot_loss_distributions(results, output_dir / "loss_distributions.svg")
    generate_html_dashboard(results, output_dir / "benchmark_dashboard.html")
    print(f"All visualizations and dashboard generated in: {output_dir}")
