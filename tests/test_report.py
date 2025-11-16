"""Tests for report generator."""
import tempfile
import os


def test_generate_html_report():
    """Test HTML report generation."""
    from cerberus.report import generate_html_report

    metrics = {"baseline_accuracy": 0.95, "adversarial_accuracy": 0.45, "attack": "fgsm"}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "report.html")
        generate_html_report(metrics, out_path=out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "baseline_accuracy" in content
            assert "0.95" in content
            assert "adversarial_accuracy" in content
            assert "0.45" in content
