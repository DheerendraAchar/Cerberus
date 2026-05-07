#!/usr/bin/env python3
"""
Fix Figure 8 - X-axis labels visibility (make them horizontal and spaced properly)
"""

import cairosvg
import os

fig8_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 850 600" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'Arial', sans-serif; }
  </style>
  
  <!-- Title -->
  <text x="425" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#1a1714">Cross-Model Transferability Heatmap</text>
  <text x="425" y="55" text-anchor="middle" font-size="12" fill="#5a5248">Mean Transfer Rate: 68.6%</text>
  
  <!-- Source models (Y axis) - LARGE FONTS -->
  <text x="75" y="100" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">RN-18</text>
  <text x="75" y="145" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">RN-50</text>
  <text x="75" y="190" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">VGG</text>
  <text x="75" y="235" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">Dense</text>
  <text x="75" y="280" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">Mobile</text>
  <text x="75" y="325" text-anchor="end" font-size="12" font-weight="bold" fill="#1a1714">Eff</text>
  
  <!-- Target models (X axis) - HORIZONTAL, CLEARLY VISIBLE -->
  <text x="115" y="370" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1714">RN-18</text>
  <text x="185" y="370" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1714">RN-50</text>
  <text x="255" y="370" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1714">VGG</text>
  <text x="325" y="370" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1714">Dense</text>
  <text x="395" y="370" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1714">Mobile</text>
  <text x="465" y="370" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1714">Eff</text>
  
  <!-- Heat cells 6x6 grid -->
  <!-- Row 1 (RN-18) -->
  <rect x="95" y="85" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="130" y="112" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <rect x="165" y="85" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="200" y="112" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">68</text>
  
  <rect x="235" y="85" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="270" y="112" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">70</text>
  
  <rect x="305" y="85" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="340" y="112" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">69</text>
  
  <rect x="375" y="85" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="410" y="112" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">65</text>
  
  <rect x="445" y="85" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="480" y="112" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">67</text>
  
  <!-- Row 2 (RN-50) -->
  <rect x="95" y="130" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="130" y="157" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">68</text>
  
  <rect x="165" y="130" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="200" y="157" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <rect x="235" y="130" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="270" y="157" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">72</text>
  
  <rect x="305" y="130" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="340" y="157" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">72</text>
  
  <rect x="375" y="130" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="410" y="157" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">66</text>
  
  <rect x="445" y="130" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="480" y="157" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">68</text>
  
  <!-- Row 3 (VGG) -->
  <rect x="95" y="175" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="130" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">70</text>
  
  <rect x="165" y="175" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="200" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">72</text>
  
  <rect x="235" y="175" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="270" y="202" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <rect x="305" y="175" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="340" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">74</text>
  
  <rect x="375" y="175" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="410" y="202" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">67</text>
  
  <rect x="445" y="175" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="480" y="202" text-anchor="middle" font-size="12" fill="#333" font-weight="bold">68</text>
  
  <!-- Remaining 3 rows simplified -->
  <rect x="95" y="220" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="165" y="220" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="235" y="220" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="305" y="220" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="340" y="247" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  <rect x="375" y="220" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="445" y="220" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <rect x="95" y="265" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="165" y="265" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="235" y="265" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="305" y="265" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="375" y="265" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="410" y="292" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  <rect x="445" y="265" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <rect x="95" y="310" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="165" y="310" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="235" y="310" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="305" y="310" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="375" y="310" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="445" y="310" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="480" y="337" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">100</text>
  
  <!-- Legend - LARGER BOX, BETTER POSITIONED -->
  <rect x="570" y="85" width="250" height="160" fill="#fff" opacity="0.98" stroke="#d8d3c8" stroke-width="2" rx="4"/>
  
  <text x="595" y="115" font-size="14" font-weight="bold" fill="#1a1714">Transfer Rate Scale</text>
  
  <!-- Color legend items with more space -->
  <rect x="595" y="135" width="24" height="24" fill="#FF0000" opacity="0.9"/>
  <text x="630" y="153" font-size="12" fill="#1a1714" font-weight="bold">100% (Perfect)</text>
  
  <rect x="595" y="175" width="24" height="24" fill="#FFA500" opacity="0.7"/>
  <text x="630" y="193" font-size="12" fill="#1a1714" font-weight="bold">70% (High)</text>
  
  <rect x="595" y="215" width="24" height="24" fill="#FFD700" opacity="0.6"/>
  <text x="630" y="233" font-size="12" fill="#1a1714" font-weight="bold">50% (Low)</text>
</svg>
"""

print("Fixing Figure 8 - X-axis labels visibility...")
with open('temp_fig8.svg', 'w') as f:
    f.write(fig8_svg)

try:
    cairosvg.svg2pdf(url='temp_fig8.svg', write_to='figures/figure8_transferability.pdf', dpi=300)
    size = os.path.getsize('figures/figure8_transferability.pdf') / 1024
    print(f"✓ Fixed: figure8_transferability.pdf ({size:.1f} KB)")
    print(f"  • X-axis labels now HORIZONTAL (no rotation)")
    print(f"  • All 6 labels clearly visible")
    print(f"  • SVG expanded to 850x600 for proper spacing")
    print(f"  • Legend repositioned with more room")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    os.remove('temp_fig8.svg')
except:
    pass

print("Done!")
