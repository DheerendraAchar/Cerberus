#!/usr/bin/env python3
"""
Extract SVG visualizations from finals_ieee.html and convert to PDFs
"""

import re
import subprocess
import os
from pathlib import Path

# Read the HTML file
with open('finals_ieee.html', 'r') as f:
    html_content = f.read()

# Define the figures to extract (Figure 5-10)
figures = {
    'fig5': {
        'title': 'Attack Success Rate vs Epsilon',
        'id': 'fig5',
        'output': 'figure5_attack_epsilon.pdf'
    },
    'fig6': {
        'title': 'Robustness-Accuracy Trade-off',
        'id': 'fig6',
        'output': 'figure6_tradeoff.pdf'
    },
    'fig7': {
        'title': 'Model Hardening Progress',
        'id': 'fig7',
        'output': 'figure7_hardening.pdf'
    },
    'fig8': {
        'title': 'Cross-Model Transferability',
        'id': 'fig8',
        'output': 'figure8_transferability.pdf'
    },
    'fig9': {
        'title': 'API Response Time Analysis',
        'id': 'fig9',
        'output': 'figure9_response_time.pdf'
    },
    'fig10': {
        'title': 'Robustness Certification',
        'id': 'fig10',
        'output': 'figure10_certification.pdf'
    }
}

print("Extracting SVG visualizations from finals_ieee.html...")
print("=" * 70)

# Extract SVG content from each figure using the id attribute
extracted_count = 0
for fig_id, fig_info in figures.items():
    print(f"\nProcessing {fig_id}: {fig_info['title']}...")
    
    # Find the figure section using the id
    pattern = f'id="{fig_info["id"]}".*?<svg(.*?)</svg>'
    match = re.search(pattern, html_content, re.DOTALL)
    
    if not match:
        print(f"  ✗ Could not find SVG in {fig_id}")
        continue
    
    # Reconstruct the full SVG tag
    svg_content = f'<svg{match.group(1)}</svg>'
    
    # Add XML declaration and namespace if missing
    if '<?xml' not in svg_content:
        svg_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_content
    
    # Save as temporary SVG file
    svg_filename = f'temp_{fig_id}.svg'
    with open(svg_filename, 'w') as f:
        f.write(svg_content)
    
    print(f"  ✓ Extracted SVG ({len(svg_content)} bytes)")
    
    # Convert SVG to PDF using cairosvg
    pdf_filename = fig_info['output']
    
    try:
        # Use cairosvg to convert
        import cairosvg
        cairosvg.svg2pdf(url=svg_filename, write_to=pdf_filename, dpi=300)
        
        if os.path.exists(pdf_filename):
            size = os.path.getsize(pdf_filename) / 1024
            print(f"  ✓ Converted to PDF: {pdf_filename} ({size:.1f} KB)")
            extracted_count += 1
        else:
            raise Exception("PDF not created")
    
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")
    
    # Clean up temporary SVG
    try:
        os.remove(svg_filename)
    except:
        pass

print("\n" + "=" * 70)

# Copy PDFs to figures folder
print("\nCopying PDFs to figures folder...")
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

for fig_id, fig_info in figures.items():
    pdf_file = fig_info['output']
    if os.path.exists(pdf_file):
        dest = figures_dir / pdf_file
        subprocess.run(['cp', pdf_file, str(dest)], capture_output=True)
        print(f"  ✓ Copied {pdf_file} → figures/{pdf_file}")

print("\n" + "=" * 70)
print(f"✓ Successfully extracted {extracted_count}/6 SVG visualizations")
print("\nGenerated files in figures/ folder:")
for fig_id, fig_info in figures.items():
    pdf_file = fig_info['output']
    if os.path.exists(f'figures/{pdf_file}'):
        size = os.path.getsize(f'figures/{pdf_file}') / 1024
        print(f"  • {pdf_file} ({size:.1f} KB)")

print("\n✓ Ready to add to LaTeX paper!")
