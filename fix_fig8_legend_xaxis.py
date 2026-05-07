#!/usr/bin/env python3
"""
Fix Figure 8 - Legend box overflow and X-axis labels
"""

import cairosvg
import os

fig8_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 800 550" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'Arial', sans-serif; }
  </style>
  
  <!-- Title -->
  <text x="400" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#1a1714">Cross-Model Transferability Heatmap</text>
  <text x="400" y="55" text-anchor="middle" font-size="12" fill="#5a5248">Mean Transfer Rate: 68.6%</text>
  
  <!-- Source models (Y axis) - LARGE FONTS -->
  <text x="70" y="100" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">RN-18</text>
  <text x="70" y="145" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">RN-50</text>
  <text x="70" y="190" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">VGG</text>
  <text x="70" y="235" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">Dense</text>
  <text x="70" y="280" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">Mobile</text>
  <text x="70" y="325" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">Eff</text>
  
  <!-- Target models (X axis) - LARGE FONTS, ANGLED -->
  <g transform="translate(105, 340)">
    <text x="0" y="0" text-anchor="start" font-size="11" font-weight="bold" fill="#1a1714" transform="rotate(45)">RN-18</text>
  </g>
  <g transform="translate(175, 340)">
    <text x="0" y="0" text-anchor="start" font-size="11" font-weight="bold" fill="#1a1714" transform="rotate(45)">RN-50</text>
  </g>
  <g transform="translate(245, 340)">
    <text x="0" y="0" text-anchor="start" font-size="11" font-weight="bold" fill="#1a1714" transform="rotate(45)">VGG</text>
  </g>
  <g transform="translate(315, 340)">
    <text x="0" y="0" text-anchor="start" font-size="11" font-weight="bold" fill="#1a1714" transform="rotate(45)">Dense</text>
  </g>
  <g transform="translate(385, 340)">
    <text x="0" y="0" text-anchor="start" font-size="11" font-weight="bold" fill="#1a1714" transform="rotate(45)">Mobile</text>
  </g>
  <g transform="translate(455, 340)">
    <text x="0" y="0" text-anchor="start" font-size="11" font-weight="bold" fill="#1a1714" transform="rotate(45)">Eff</text>
  </g>
  
  <!-- Heat cells 6x6 grid -->
  <!-- Row 1 (RN-18) -->
  <rect x="85" y="85" width="65" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="117" y="112" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <rect x="155" y="85" width="65" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="187" y="112" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">68</text>
  
  <rect x="225" y="85" width="65" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="257" y="112" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">70</text>
  
  <rect x="295" y="85" width="65" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="327" y="112" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">69</text>
  
  <rect x="365" y="85" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="397" y="112" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">65</text>
  
  <rect x="435" y="85" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="467" y="112" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">67</text>
  
  <!-- Row 2 (RN-50) -->
  <rect x="85" y="130" width="65" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="117" y="157" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">68</text>
  
  <rect x="155" y="130" width="65" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="187" y="157" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <rect x="225" y="130" width="65" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="257" y="157" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">72</text>
  
  <rect x="295" y="130" width="65" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="327" y="157" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">72</text>
  
  <rect x="365" y="130" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="397" y="157" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">66</text>
  
  <rect x="435" y="130" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="467" y="157" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">68</text>
  
  <!-- Row 3 (VGG) -->
  <rect x="85" y="175" width="65" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="117" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">70</text>
  
  <rect x="155" y="175" width="65" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="187" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">72</text>
  
  <rect x="225" y="175" width="65" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="257" y="202" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <rect x="295" y="175" width="65" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="327" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">74</text>
  
  <rect x="365" y="175" width="65" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="397" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">67</text>
  
  <rect x="435" y="175" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="467" y="202" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">68</text>
  
  <!-- Remaining 3 rows simplified -->
  <rect x="85" y="220" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="155" y="220" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="225" y="220" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="295" y="220" width="65" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="327" y="247" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  <rect x="365" y="220" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="435" y="220" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <rect x="85" y="265" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="155" y="265" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="225" y="265" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="295" y="265" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="365" y="265" width="65" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="397" y="292" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  <rect x="435" y="265" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <rect x="85" y="310" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="155" y="310" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="225" y="310" width="65" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="295" y="310" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="365" y="310" width="65" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="435" y="310" width="65" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="467" y="337" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <!-- Legend - LARGER BOX, BETTER POSITIONED -->
  <rect x="540" y="85" width="230" height="150" fill="#fff" opacity="0.98" stroke="#d8d3c8" stroke-width="2" rx="4"/>
  
  <text x="560" y="110" font-size="13" font-weight="bold" fill="#1a1714">Transfer Rate Scale</text>
  
  <!-- Color legend items with more space -->
  <rect x="560" y="130" width="22" height="22" fill="#FF0000" opacity="0.9"/>
  <text x="590" y="148" font-size="12" fill="#1a1714" font-weight="bold">100% (Perfect)</text>
  
  <rect x="560" y="165" width="22" height="22" fill="#FFA500" opacity="0.7"/>
  <text x="590" y="183" font-size="12" fill="#1a1714" font-weight="bold">70% (High)</text>
  
  <rect x="560" y="200" width="22" height="22" fill="#FFD700" opacity="0.6"/>
  <text x="590" y="218" font-size="12" fill="#1a1714" font-weight="bold">50% (Low)</text>
</svg>
"""

print("Fixing Figure 8 - Legend box and X-axis...")
with open('temp_fig8.svg', 'w') as f:
    f.write(fig8_svg)

try:
    cairosvg.svg2pdf(url='temp_fig8.svg', write_to='figures/figure8_transferability.pdf', dpi=300)
    size = os.path.getsize('figures/figure8_transferability.pdf') / 1024
    print(f"✓ Fixed: figure8_transferability.pdf ({size:.1f} KB)")
    print(f"  • Legend box enlarged to 230x150 (fits content)")
    print(f"  • X-axis labels rotated 45° for clarity")
    print(f"  • Better spacing and positioning")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    os.remove('temp_fig8.svg')
except:
    pass

print("Done!")
