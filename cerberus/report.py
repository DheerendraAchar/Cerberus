"""Simple report generator (HTML) using Jinja2."""
from typing import Dict, Any


def generate_html_report(metrics: Dict[str, Any], out_path: str = "outputs/report.html") -> None:
    """Render a small HTML report with provided metrics."""
    try:
        from jinja2 import Template
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Jinja2 is required to render reports") from exc

    template = Template(
        """
        <html>
        <head><title>Project Cerberus - Report</title></head>
        <body>
        <h1>Project Cerberus - Run Report</h1>
        <ul>
        {% for k, v in metrics.items() %}
          <li><strong>{{ k }}:</strong> {{ v }}</li>
        {% endfor %}
        </ul>
        </body>
        </html>
        """
    )

    content = template.render(metrics=metrics)
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
