import unittest
from unlearn_ai.visualization.plots import create_svg_scatter_chart, escape_xml


class TestVisualization(unittest.TestCase):

    def test_escape_xml(self):
        self.assertEqual(escape_xml("A < B & C > D"), "A &lt; B &amp; C &gt; D")
        self.assertEqual(escape_xml('Quotes "here"'), "Quotes &quot;here&quot;")

    def test_create_svg_scatter_chart(self):
        sample_points = [
            {"label": "Model A", "x": 0.8, "y": 0.9, "color": "#ff0000"},
            {"label": "Model B", "x": 0.3, "y": 0.5, "color": "#00ff00"},
        ]
        svg = create_svg_scatter_chart(
            title="Test Chart",
            x_label="X Axis",
            y_label="Y Axis",
            points=sample_points,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
        )
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn("Test Chart", svg)
        self.assertIn("Model A", svg)
        self.assertIn("Model B", svg)
        self.assertIn("stroke=", svg)


if __name__ == "__main__":
    unittest.main()

