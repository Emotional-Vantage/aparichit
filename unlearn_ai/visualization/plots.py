import os
import json
import math
import html
from pathlib import Path
from typing import Dict, Any, List, Optional


METHOD_COLORS = {
    "m_base": "#64748b",
    "m_full": "#dc2626",
    "m_full_finetuned": "#dc2626",
    "m_retain": "#059669",
    "m_unlearned_gd": "#7c3aed",
    "m_unlearned_fisher": "#d97706",
    "m_unlearned_tv": "#0284c7",
    "m_unlearned_generic_tier1": "#0891b2",
    "m_unlearned_generic_tier2": "#16a34a",
}

SHORT_LABELS = {
    "m_base": "M_base",
    "m_full": "M_full",
    "m_full_finetuned": "M_full",
    "m_retain": "M_retain",
    "m_unlearned_gd": "GD",
    "m_unlearned_fisher": "Fisher",
    "m_unlearned_tv": "Task Vector",
    "m_unlearned_generic_tier1": "Generic FT",
    "m_unlearned_generic_tier2": "Generic FT (T2)",
}

CLEAN_NAMES = {
    "m_base": "Reference: M_base (Pretrained)",
    "m_full": "Baseline: M_full (Memorized)",
    "m_full_finetuned": "Baseline: M_full (Memorized)",
    "m_retain": "Reference: M_retain (Retrained)",
    "m_unlearned_gd": "Method 1: Gradient Difference (GD)",
    "m_unlearned_fisher": "Method 2: Fisher Retain-Anchored",
    "m_unlearned_tv": "Method 3: Task Vector Negation",
    "m_unlearned_generic_tier1": "Method 4: Generic-Label Tier 1",
    "m_unlearned_generic_tier2": "Method 4: Generic-Label Tier 2",
}


def get_model_color(model_key: str) -> str:
    """Returns color mapped to model key (case-insensitive)."""
    norm = model_key.lower().strip()
    return METHOD_COLORS.get(norm, "#475569")


def get_short_label(model_key: str) -> str:
    norm = model_key.lower().strip()
    return SHORT_LABELS.get(norm, model_key)


def get_clean_name(model_key: str) -> str:
    norm = model_key.lower().strip()
    return CLEAN_NAMES.get(norm, model_key)


def escape_xml(text: str) -> str:
    """Escapes XML/SVG special characters."""
    return html.escape(str(text), quote=True)


def create_svg_scatter_chart(
    title: str,
    x_label: str,
    y_label: str,
    points: List[Dict[str, Any]],
    width: int = 960,
    height: int = 540,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    draw_frontier: bool = True,
    v_guideline: Optional[float] = None,
    v_guideline_label: Optional[str] = None,
    h_guideline: Optional[float] = None,
    h_guideline_label: Optional[str] = None,
) -> str:
    """
    Renders an elegant, publication-grade SVG scatter plot with high contrast,
    collision-free labels, non-overlapping legend, and custom annotations.
    """
    margin_l = 80
    margin_r = 250
    margin_t = 65
    margin_b = 65

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
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_l + plot_w/2}" y="36" text-anchor="middle" font-size="17" font-weight="bold" fill="#0f172a">{escape_xml(title)}</text>',
    ]

    # Grid & Axes
    num_ticks = 5
    for i in range(num_ticks + 1):
        val_x = x_min + i * (x_max - x_min) / num_ticks
        px = scale_x(val_x)
        svg_parts.append(f'<line x1="{px:.1f}" y1="{margin_t}" x2="{px:.1f}" y2="{margin_t + plot_h}" stroke="#f1f5f9" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{px:.1f}" y="{margin_t + plot_h + 20}" text-anchor="middle" font-size="11" fill="#64748b">{val_x:.2f}</text>')

    for i in range(num_ticks + 1):
        val_y = y_min + i * (y_max - y_min) / num_ticks
        py = scale_y(val_y)
        svg_parts.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{margin_l + plot_w}" y2="{py:.1f}" stroke="#f1f5f9" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{margin_l - 12}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{val_y:.2f}</text>')

    # Optional Guidelines
    if v_guideline is not None and x_min <= v_guideline <= x_max:
        gx = scale_x(v_guideline)
        svg_parts.append(f'<line x1="{gx:.1f}" y1="{margin_t}" x2="{gx:.1f}" y2="{margin_t + plot_h}" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4,4" />')
        if v_guideline_label:
            svg_parts.append(f'<text x="{gx + 6:.1f}" y="{margin_t + 16}" font-size="10" font-weight="600" fill="#059669">{escape_xml(v_guideline_label)}</text>')

    if h_guideline is not None and y_min <= h_guideline <= y_max:
        gy = scale_y(h_guideline)
        svg_parts.append(f'<line x1="{margin_l}" y1="{gy:.1f}" x2="{margin_l + plot_w}" y2="{gy:.1f}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3,3" />')
        if h_guideline_label:
            svg_parts.append(f'<text x="{margin_l + plot_w - 6:.1f}" y="{gy - 6:.1f}" text-anchor="end" font-size="10" fill="#64748b">{escape_xml(h_guideline_label)}</text>')

    # Axes border
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')

    # Axis Labels
    svg_parts.append(f'<text x="{margin_l + plot_w/2}" y="{height - 18}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{escape_xml(x_label)}</text>')
    svg_parts.append(f'<text transform="rotate(-90)" x="-{margin_t + plot_h/2}" y="24" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">{escape_xml(y_label)}</text>')

    # Pareto boundary curve
    if draw_frontier and len(points) >= 2:
        sorted_pts = sorted(points, key=lambda p: p["x"])
        frontier_pts = []
        max_y_so_far = -1e9
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
            svg_parts.append(f'<path d="{" ".join(path_d)}" stroke="#64748b" stroke-width="2.2" stroke-dasharray="6,4" fill="none" />')

    # Legend container
    legend_x = margin_l + plot_w + 20
    legend_y = margin_t
    legend_w = 230
    legend_h = max(160, len(points) * 28 + 44)
    svg_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" fill="#f8fafc" rx="8" stroke="#cbd5e1" stroke-width="1" />')
    svg_parts.append(f'<text x="{legend_x + 14}" y="{legend_y + 24}" font-size="12" font-weight="bold" fill="#0f172a">Model / Algorithm</text>')

    # Legend entries
    for idx, pt in enumerate(points):
        entry_y = legend_y + 48 + idx * 26
        color = pt.get("color", "#3182ce")
        disp_name = pt.get("clean_name", pt.get("label", f"Model {idx+1}"))
        svg_parts.append(f'<circle cx="{legend_x + 18}" cy="{entry_y - 4}" r="5.5" fill="{color}" />')
        svg_parts.append(f'<text x="{legend_x + 32}" y="{entry_y}" font-size="11" font-weight="500" fill="#334155">{escape_xml(disp_name)}</text>')

    # Plot Points with collision-free labels
    # Pre-calculate positions to handle close neighbors
    positioned = []
    for idx, pt in enumerate(points):
        px = scale_x(pt["x"])
        py = scale_y(pt["y"])
        positioned.append({
            "idx": idx,
            "pt": pt,
            "px": px,
            "py": py,
            "dx": 0,
            "dy": 0,
            "anchor": "start",
        })

    # Detect close clusters and adjust offsets
    for i in range(len(positioned)):
        p1 = positioned[i]
        # Position label to left if near right edge
        if p1["px"] > (margin_l + plot_w - 75):
            p1["anchor"] = "end"
            p1["dx"] = -12
        else:
            p1["anchor"] = "start"
            p1["dx"] = 12

        # Default vertical placement
        p1["dy"] = -4

        # Check neighbor collision
        for j in range(i + 1, len(positioned)):
            p2 = positioned[j]
            dist_x = abs(p1["px"] - p2["px"])
            dist_y = abs(p1["py"] - p2["py"])
            if dist_x < 50 and dist_y < 25:
                # Stagger vertically
                p1["dy"] = -12
                p2["dy"] = 16

    # Specific known near-collisions (Fisher and GD, M_full and Generic FT)
    for p in positioned:
        key = p["pt"].get("key", "").lower()
        if "fisher" in key:
            p["dy"] = -14
        elif "gd" in key:
            p["dy"] = 18
        elif "generic" in key:
            p["dy"] = -12
        elif "m_full" in key:
            p["dy"] = 18

    # Render point elements
    for p in positioned:
        pt = p["pt"]
        px = p["px"]
        py = p["py"]
        color = pt.get("color", "#3182ce")
        short_lbl = pt.get("short_label", pt.get("label", ""))

        # Point shadow/glow
        svg_parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="{color}" opacity="0.22" />')
        # Core point circle
        svg_parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2" />')

        # Clean short callout with white text-halo for guaranteed legibility
        text_x = px + p["dx"]
        text_y = py + p["dy"]
        svg_parts.append(
            f'<text x="{text_x:.1f}" y="{text_y:.1f}" text-anchor="{p["anchor"]}" '
            f'font-size="11" font-weight="700" fill="#0f172a" '
            f'stroke="#ffffff" stroke-width="3.5" paint-order="stroke fill" stroke-linejoin="round">'
            f'{escape_xml(short_lbl)}</text>'
        )

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
        f_rouge = eval_res["forget"].get("rougeL", eval_res["forget"].get("rougeL_recall", 0.0))
        r_rouge = eval_res["retain"].get("rougeL", eval_res["retain"].get("rougeL_recall", 0.0))

        points.append({
            "key": model_key,
            "label": report.get("display_name", model_key),
            "short_label": get_short_label(model_key),
            "clean_name": get_clean_name(model_key),
            "x": 1.0 - f_rouge,
            "y": r_rouge,
            "color": get_model_color(model_key),
        })

    svg = create_svg_scatter_chart(
        title="Pareto Frontier: Unlearning Efficacy vs. Knowledge Retention",
        x_label="Erasure Efficacy: 1.0 - Forget ROUGE-L (Higher = Cleaner Forget)",
        y_label="Model Utility: Retain ROUGE-L (Higher = Better Retention)",
        points=points,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.05,
        draw_frontier=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"  [✓] Pareto chart written to: {output_path}")


def plot_privacy_utility_tradeoff(results: Dict[str, Any], output_path: Path):
    """
    Generates Privacy Trade-off Plot:
    X-axis: Direction-Adjusted MIA ROC-AUC (0.50 = Optimal / Indistinguishable, 1.0 = Complete Leakage)
    Y-axis: Retain ROUGE-L (Higher = Better Retention)
    """
    points = []
    for model_key, report in results.items():
        eval_res = report["evaluation"]
        audit_res = report.get("privacy_audit", {})

        raw_auc = audit_res.get("loss_threshold_mia_auc", audit_res.get("raw_auc", 0.50))
        adj_auc = audit_res.get("adjusted_auc", audit_res.get("direction_adjusted_auc", max(raw_auc, 1.0 - raw_auc)))
        r_rouge = eval_res["retain"].get("rougeL", eval_res["retain"].get("rougeL_recall", 0.0))

        points.append({
            "key": model_key,
            "label": report.get("display_name", model_key),
            "short_label": get_short_label(model_key),
            "clean_name": get_clean_name(model_key),
            "x": adj_auc,
            "y": r_rouge,
            "color": get_model_color(model_key),
        })

    svg = create_svg_scatter_chart(
        title="Privacy vs. Utility: Membership Inference AUC vs. Retain ROUGE-L",
        x_label="Direction-Adjusted MIA ROC-AUC (0.50 = Ideal Privacy, 1.00 = Complete Leakage)",
        y_label="Model Utility: Retain ROUGE-L (Higher = Better Retention)",
        points=points,
        x_min=0.45,
        x_max=1.05,
        y_min=0.0,
        y_max=1.05,
        draw_frontier=False,
        v_guideline=0.50,
        v_guideline_label="Ideal Privacy (AUC = 0.50)",
        h_guideline=0.927,
        h_guideline_label="Target Retain (~0.93)",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"  [✓] Privacy trade-off chart written to: {output_path}")


def plot_loss_distributions(results: Dict[str, Any], output_path: Path):
    """
    Generates publication-quality cross-entropy loss comparison bars across
    Forget, Retain, and Unseen splits with angled non-overlapping category labels.
    """
    width = 980
    height = 520
    margin_l = 80
    margin_r = 40
    margin_t = 70
    margin_b = 135

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    model_keys = list(results.keys())
    if not model_keys:
        return

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" '
        'style="background-color:#ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="{margin_l + plot_w/2}" y="36" text-anchor="middle" font-size="17" font-weight="bold" fill="#0f172a">Cross-Entropy Loss Across TOFU Splits (Forget / Retain / Unseen)</text>',
    ]

    max_loss = max(
        max(
            r["evaluation"]["forget"]["loss"],
            r["evaluation"]["retain"]["loss"],
            r["evaluation"]["unseen"]["loss"],
        )
        for r in results.values()
    )
    max_loss = math.ceil(max(max_loss * 1.20, 6.0))

    num_models = len(model_keys)
    group_width = plot_w / num_models
    bar_width = min(26.0, (group_width * 0.72) / 3)

    colors = {"forget": "#ef4444", "retain": "#2563eb", "unseen": "#7c3aed"}

    # Y Grid
    num_y_ticks = 5
    for i in range(num_y_ticks + 1):
        val = i * (max_loss / num_y_ticks)
        py = margin_t + plot_h - (val / max_loss) * plot_h
        svg_parts.append(f'<line x1="{margin_l}" y1="{py:.1f}" x2="{margin_l + plot_w}" y2="{py:.1f}" stroke="#f1f5f9" stroke-width="1.5" />')
        svg_parts.append(f'<text x="{margin_l - 12}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{val:.1f}</text>')

    # Unseen Control Reference Guideline (~4.11)
    unseen_ref = results.get("m_retain", {}).get("evaluation", {}).get("unseen", {}).get("loss", 4.109)
    if unseen_ref:
        py_unseen = margin_t + plot_h - (unseen_ref / max_loss) * plot_h
        svg_parts.append(f'<line x1="{margin_l}" y1="{py_unseen:.1f}" x2="{margin_l + plot_w}" y2="{py_unseen:.1f}" stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="4,4" />')
        svg_parts.append(f'<text x="{margin_l + plot_w - 6}" y="{py_unseen - 6:.1f}" text-anchor="end" font-size="10" font-weight="600" fill="#7c3aed">Unseen Control Level ({unseen_ref:.2f})</text>')

    # Bars and Value Labels
    for idx, mk in enumerate(model_keys):
        r = results[mk]["evaluation"]
        group_center = margin_l + idx * group_width + group_width / 2

        split_vals = [
            ("forget", r["forget"]["loss"], -bar_width - 2),
            ("retain", r["retain"]["loss"], 0),
            ("unseen", r["unseen"]["loss"], bar_width + 2),
        ]

        for s_name, val, offset in split_vals:
            bx = group_center + offset - bar_width / 2
            bh = max(3.0, (val / max_loss) * plot_h)
            by = margin_t + plot_h - bh
            svg_parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width:.1f}" height="{bh:.1f}" fill="{colors[s_name]}" rx="3" />')
            # Text label above bar with white halo
            svg_parts.append(
                f'<text x="{bx + bar_width/2:.1f}" y="{by - 4:.1f}" text-anchor="middle" '
                f'font-size="9.5" font-weight="600" fill="#0f172a" '
                f'stroke="#ffffff" stroke-width="2.5" paint-order="stroke fill">{val:.2f}</text>'
            )

        # Angled X-axis label to eliminate horizontal collisions completely
        clean_lbl = get_clean_name(mk)
        label_y = margin_t + plot_h + 16
        svg_parts.append(
            f'<text transform="rotate(-30, {group_center:.1f}, {label_y:.1f})" '
            f'x="{group_center:.1f}" y="{label_y:.1f}" text-anchor="end" '
            f'font-size="11.5" font-weight="600" fill="#1e293b">{escape_xml(clean_lbl)}</text>'
        )

    # Axes
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')
    svg_parts.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#334155" stroke-width="2" />')

    # Legend at the top
    legend_y = margin_t - 22
    svg_parts.append(f'<rect x="{margin_l + 10}" y="{legend_y - 11}" width="12" height="12" fill="{colors["forget"]}" rx="2" />')
    svg_parts.append(f'<text x="{margin_l + 28}" y="{legend_y}" font-size="11" font-weight="500" fill="#334155">Forget Loss (Erased, Target ≈ Unseen)</text>')

    svg_parts.append(f'<rect x="{margin_l + 265}" y="{legend_y - 11}" width="12" height="12" fill="{colors["retain"]}" rx="2" />')
    svg_parts.append(f'<text x="{margin_l + 283}" y="{legend_y}" font-size="11" font-weight="500" fill="#334155">Retain Loss (Preserved, Target ≈ 0.03)</text>')

    svg_parts.append(f'<rect x="{margin_l + 535}" y="{legend_y - 11}" width="12" height="12" fill="{colors["unseen"]}" rx="2" />')
    svg_parts.append(f'<text x="{margin_l + 553}" y="{legend_y}" font-size="11" font-weight="500" fill="#334155">Unseen Control Loss (Baseline)</text>')

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
        .grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
        img.chart {{ width: 100%; height: auto; border-radius: 8px; display: block; border: 1px solid #e2e8f0; }}
        h3 {{ margin-top: 0; color: #334155; }}
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
                    <h3>1. Pareto Frontier: Erasure Efficacy vs. Knowledge Retention</h3>
                    <img class="chart" src="pareto_frontier.svg" alt="Pareto Frontier">
                </div>
                <div>
                    <h3>2. Privacy vs. Utility: Direction-Adjusted MIA ROC-AUC vs. Retain ROUGE-L</h3>
                    <img class="chart" src="privacy_utility_tradeoff.svg" alt="Privacy Utility Tradeoff">
                </div>
                <div>
                    <h3>3. Cross-Entropy Loss Distributions across TOFU Splits</h3>
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
