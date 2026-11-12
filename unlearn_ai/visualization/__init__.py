"""
Aparichit Visualization Suite:
Publication-ready vector charts (SVG) and Pareto trade-off curves.
"""

from .plots import generate_all_visualizations, plot_pareto_frontier, plot_loss_distributions

__all__ = [
    "generate_all_visualizations",
    "plot_pareto_frontier",
    "plot_loss_distributions",
]

