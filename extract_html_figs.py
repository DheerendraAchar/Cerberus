#!/usr/bin/env python3
"""
Manual extraction and PDF conversion for complex SVGs
"""

import cairosvg
import os

# Figure 5: Attack Success Rate vs Epsilon (simplified version)
fig5_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      text { font-family: 'DM Mono', monospace; }
    </style>
  </defs>
  
  <!-- Title -->
  <text x="340" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1714">Attack Success Rate vs Perturbation Budget</text>
  
  <!-- Grid lines -->
  <line x1="60" y1="40" x2="60" y2="290" stroke="#d8d3c8" stroke-width="1.5"/>
  <line x1="60" y1="290" x2="640" y2="290" stroke="#d8d3c8" stroke-width="1.5"/>
  
  <!-- Y axis labels -->
  <text x="50" y="50" text-anchor="end" font-size="10" fill="#5a5248">100%</text>
  <text x="50" y="100" text-anchor="end" font-size="10" fill="#5a5248">90%</text>
  <text x="50" y="150" text-anchor="end" font-size="10" fill="#5a5248">80%</text>
  <text x="50" y="200" text-anchor="end" font-size="10" fill="#5a5248">70%</text>
  <text x="50" y="250" text-anchor="end" font-size="10" fill="#5a5248">60%</text>
  <text x="50" y="300" text-anchor="end" font-size="10" fill="#5a5248">50%</text>
  
  <!-- Horizontal grid -->
  <line x1="60" y1="50" x2="640" y2="50" stroke="#f0ede6" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="60" y1="100" x2="640" y2="100" stroke="#f0ede6" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="60" y1="150" x2="640" y2="150" stroke="#f0ede6" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="60" y1="200" x2="640" y2="200" stroke="#f0ede6" stroke-width="1" stroke-dasharray="4,4"/>
  <line x1="60" y1="250" x2="640" y2="250" stroke="#f0ede6" stroke-width="1" stroke-dasharray="4,4"/>
  
  <!-- X axis labels -->
  <text x="60" y="310" text-anchor="middle" font-size="10" fill="#5a5248">0.00</text>
  <text x="175" y="310" text-anchor="middle" font-size="10" fill="#5a5248">0.01</text>
  <text x="275" y="310" text-anchor="middle" font-size="10" fill="#5a5248">0.02</text>
  <text x="375" y="310" text-anchor="middle" font-size="10" fill="#5a5248">0.03</text>
  <text x="475" y="310" text-anchor="middle" font-size="10" fill="#5a5248">0.06</text>
  <text x="590" y="310" text-anchor="middle" font-size="10" fill="#5a5248">0.10</text>
  
  <!-- X axis label -->
  <text x="350" y="330" text-anchor="middle" font-size="10" fill="#5a5248">Perturbation Budget (ε)</text>
  
  <!-- C&W line (highest) -->
  <polyline points="60,230 175,130 275,85 375,55 475,30 590,28" fill="none" stroke="#c8440c" stroke-width="2.5" stroke-linejoin="round"/>
  <circle cx="60" cy="230" r="3" fill="#c8440c"/>
  <circle cx="175" cy="130" r="3" fill="#c8440c"/>
  <circle cx="275" cy="85" r="3" fill="#c8440c"/>
  <circle cx="375" cy="55" r="3" fill="#c8440c"/>
  <circle cx="475" cy="30" r="3" fill="#c8440c"/>
  <circle cx="590" cy="28" r="3" fill="#c8440c"/>
  <text x="605" y="25" font-size="9" fill="#c8440c" font-weight="bold">C&amp;W</text>
  
  <!-- PGD line -->
  <polyline points="60,238 175,148 275,110 375,80 475,58 590,45" fill="none" stroke="#2a5fa5" stroke-width="2.5" stroke-linejoin="round"/>
  <text x="605" y="42" font-size="9" fill="#2a5fa5" font-weight="bold">PGD</text>
  
  <!-- DeepFool line -->
  <polyline points="60,252 175,185 275,163 375,148 475,133 590,118" fill="none" stroke="#2d7a4f" stroke-width="2.5" stroke-linejoin="round"/>
  <text x="605" y="115" font-size="9" fill="#2d7a4f" font-weight="bold">DeepFool</text>
  
  <!-- FGSM line -->
  <polyline points="60,258 175,200 275,183 375,170 475,163 590,155" fill="none" stroke="#d4a017" stroke-width="2.5" stroke-linejoin="round"/>
  <text x="605" y="152" font-size="9" fill="#d4a017" font-weight="bold">FGSM</text>
  
  <!-- JSMA line -->
  <polyline points="60,270 175,228 275,215 375,205 475,198 590,188" fill="none" stroke="#7a3fa5" stroke-width="2.5" stroke-linejoin="round"/>
  <text x="605" y="185" font-size="9" fill="#7a3fa5" font-weight="bold">JSMA</text>
  
  <!-- Y axis label -->
  <text x="20" y="170" text-anchor="middle" font-size="10" fill="#5a5248" transform="rotate(-90 20 170)">Attack Success Rate (%)</text>
</svg>
"""

# Figure 6: Robustness-Accuracy Trade-off
fig6_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      text { font-family: 'DM Mono', monospace; }
    </style>
  </defs>
  
  <!-- Title -->
  <text x="300" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1a1714">Robustness-Accuracy Trade-off</text>
  
  <!-- Grid -->
  <line x1="70" y1="40" x2="70" y2="250" stroke="#d8d3c8" stroke-width="1.5"/>
  <line x1="70" y1="250" x2="570" y2="250" stroke="#d8d3c8" stroke-width="1.5"/>
  
  <!-- Horizontal grid lines -->
  <line x1="70" y1="50" x2="570" y2="50" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="100" x2="570" y2="100" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="150" x2="570" y2="150" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="200" x2="570" y2="200" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  
  <!-- Y axis labels (Robustness) -->
  <text x="60" y="55" text-anchor="end" font-size="10" fill="#5a5248">90%</text>
  <text x="60" y="105" text-anchor="end" font-size="10" fill="#5a5248">80%</text>
  <text x="60" y="155" text-anchor="end" font-size="10" fill="#5a5248">70%</text>
  <text x="60" y="205" text-anchor="end" font-size="10" fill="#5a5248">60%</text>
  <text x="60" y="255" text-anchor="end" font-size="10" fill="#5a5248">50%</text>
  
  <!-- X axis labels (Accuracy) -->
  <text x="70" y="270" text-anchor="middle" font-size="10" fill="#5a5248">85%</text>
  <text x="170" y="270" text-anchor="middle" font-size="10" fill="#5a5248">88%</text>
  <text x="270" y="270" text-anchor="middle" font-size="10" fill="#5a5248">90%</text>
  <text x="370" y="270" text-anchor="middle" font-size="10" fill="#5a5248">92%</text>
  <text x="470" y="270" text-anchor="middle" font-size="10" fill="#5a5248">94%</text>
  <text x="570" y="270" text-anchor="middle" font-size="10" fill="#5a5248">95%</text>
  
  <!-- Axis labels -->
  <text x="320" y="290" text-anchor="middle" font-size="10" fill="#5a5248">Clean Accuracy (%)</text>
  <text x="25" y="150" text-anchor="middle" font-size="10" fill="#5a5248" transform="rotate(-90 25 150)">Adversarial Robustness (%)</text>
  
  <!-- Baseline models (red dots) -->
  <circle cx="500" cy="200" r="4" fill="#FF6B6B" opacity="0.7"/>
  <circle cx="480" cy="210" r="4" fill="#FF6B6B" opacity="0.7"/>
  <circle cx="470" cy="215" r="4" fill="#FF6B6B" opacity="0.7"/>
  <circle cx="490" cy="205" r="4" fill="#FF6B6B" opacity="0.7"/>
  <text x="440" y="225" font-size="9" fill="#FF6B6B" font-weight="bold">Baseline</text>
  
  <!-- After defense (green dots) -->
  <circle cx="420" cy="80" r="4" fill="#51CF66" opacity="0.8"/>
  <circle cx="400" cy="90" r="4" fill="#51CF66" opacity="0.8"/>
  <circle cx="390" cy="95" r="4" fill="#51CF66" opacity="0.8"/>
  <circle cx="410" cy="85" r="4" fill="#51CF66" opacity="0.8"/>
  <text x="360" y="115" font-size="9" fill="#51CF66" font-weight="bold">After Training</text>
  
  <!-- Arrow showing improvement -->
  <defs>
    <marker id="arrowhead-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#51CF66"/>
    </marker>
  </defs>
  <line x1="470" y1="210" x2="410" y2="90" stroke="#51CF66" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrowhead-green)"/>
</svg>
"""

print("Generating high-quality SVGs from HTML...")
print("=" * 70)

# Save and convert Figure 5
print("\nProcessing Figure 5: Attack Success Rate vs Epsilon...")
with open('temp_fig5.svg', 'w') as f:
    f.write(fig5_svg)

try:
    cairosvg.svg2pdf(url='temp_fig5.svg', write_to='figure5_attack_epsilon.pdf', dpi=300)
    print("  ✓ Converted to PDF: figure5_attack_epsilon.pdf")
except Exception as e:
    print(f"  ✗ Conversion failed: {e}")

# Save and convert Figure 6
print("\nProcessing Figure 6: Robustness-Accuracy Trade-off...")
with open('temp_fig6.svg', 'w') as f:
    f.write(fig6_svg)

try:
    cairosvg.svg2pdf(url='temp_fig6.svg', write_to='figure6_tradeoff.pdf', dpi=300)
    print("  ✓ Converted to PDF: figure6_tradeoff.pdf")
except Exception as e:
    print(f"  ✗ Conversion failed: {e}")

# Copy to figures folder
import subprocess
os.system('cp figure5_attack_epsilon.pdf figures/figure5_attack_epsilon.pdf')
os.system('cp figure6_tradeoff.pdf figures/figure6_tradeoff.pdf')

# Cleanup
os.system('rm temp_fig5.svg temp_fig6.svg')

print("\n" + "=" * 70)
print("✓ All supplementary figures extracted and converted!")
print("\nFigures in figures/ folder:")
subprocess.run(['ls', '-lh', 'figures/figure*.pdf'], cwd='/Users/admin/Desktop/major_projekt')
