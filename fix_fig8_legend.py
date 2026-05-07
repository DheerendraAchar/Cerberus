#!/usr/bin/env python3
"""
Fix Figure 8 legend positioning
"""

import cairosvg
import os

fig8_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 650 650" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'DM Mono', monospace; }
  </style>
  
  <!-- Title -->
  <text x="325" y="30" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1714">Transferability: Source → Target Models</text>
  <text x="325" y="50" text-anchor="middle" font-size="11" fill="#5a5248">Mean Transfer Rate: 68.6%</text>
  
  <!-- Source models (Y axis) -->
  <text x="55" y="95" text-anchor="end" font-size="9" fill="#5a5248">ResNet-18</text>
  <text x="55" y="135" text-anchor="end" font-size="9" fill="#5a5248">ResNet-50</text>
  <text x="55" y="175" text-anchor="end" font-size="9" fill="#5a5248">VGG-16</text>
  <text x="55" y="215" text-anchor="end" font-size="9" fill="#5a5248">DenseNet</text>
  <text x="55" y="255" text-anchor="end" font-size="9" fill="#5a5248">MobileV2</text>
  <text x="55" y="295" text-anchor="end" font-size="9" fill="#5a5248">EfficientNet</text>
  
  <!-- Target models (X axis) -->
  <text x="95" y="610" text-anchor="middle" font-size="9" fill="#5a5248">RN-18</text>
  <text x="165" y="610" text-anchor="middle" font-size="9" fill="#5a5248">RN-50</text>
  <text x="235" y="610" text-anchor="middle" font-size="9" fill="#5a5248">VGG</text>
  <text x="305" y="610" text-anchor="middle" font-size="9" fill="#5a5248">Dense</text>
  <text x="375" y="610" text-anchor="middle" font-size="9" fill="#5a5248">Mobile</text>
  <text x="445" y="610" text-anchor="middle" font-size="9" fill="#5a5248">Eff</text>
  
  <!-- Heat cells with color gradient -->
  <!-- Row 1 (ResNet-18) -->
  <rect x="70" y="80" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="105" y="107" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  
  <rect x="140" y="80" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="175" y="107" text-anchor="middle" font-size="9" fill="#fff">68.5</text>
  
  <rect x="210" y="80" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="245" y="107" text-anchor="middle" font-size="9" fill="#fff">70.2</text>
  
  <rect x="280" y="80" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="315" y="107" text-anchor="middle" font-size="9" fill="#fff">69.1</text>
  
  <rect x="350" y="80" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="385" y="107" text-anchor="middle" font-size="9" fill="#333">65.3</text>
  
  <rect x="420" y="80" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="455" y="107" text-anchor="middle" font-size="9" fill="#333">66.8</text>
  
  <!-- Row 2 (ResNet-50) -->
  <rect x="70" y="120" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="105" y="147" text-anchor="middle" font-size="9" fill="#fff">68.5</text>
  
  <rect x="140" y="120" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="175" y="147" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  
  <rect x="210" y="120" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="245" y="147" text-anchor="middle" font-size="9" fill="#fff">71.8</text>
  
  <rect x="280" y="120" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="315" y="147" text-anchor="middle" font-size="9" fill="#fff">72.1</text>
  
  <rect x="350" y="120" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="385" y="147" text-anchor="middle" font-size="9" fill="#333">66.2</text>
  
  <rect x="420" y="120" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="455" y="147" text-anchor="middle" font-size="9" fill="#333">67.9</text>
  
  <!-- Row 3 (VGG-16) - Highest transferability -->
  <rect x="70" y="160" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="105" y="187" text-anchor="middle" font-size="9" fill="#fff">70.2</text>
  
  <rect x="140" y="160" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="175" y="187" text-anchor="middle" font-size="9" fill="#fff">71.8</text>
  
  <rect x="210" y="160" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="245" y="187" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  
  <rect x="280" y="160" width="70" height="40" fill="#FF6B00" opacity="0.8"/>
  <text x="315" y="187" text-anchor="middle" font-size="9" fill="#fff">73.6</text>
  
  <rect x="350" y="160" width="70" height="40" fill="#FFA500" opacity="0.7"/>
  <text x="385" y="187" text-anchor="middle" font-size="9" fill="#fff">67.3</text>
  
  <rect x="420" y="160" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <text x="455" y="187" text-anchor="middle" font-size="9" fill="#333">68.4</text>
  
  <!-- Row 4 (DenseNet) -->
  <rect x="70" y="200" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="140" y="200" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="210" y="200" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="280" y="200" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="315" y="227" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  <rect x="350" y="200" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="420" y="200" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <!-- Row 5 (MobileV2) -->
  <rect x="70" y="240" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="140" y="240" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="210" y="240" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="280" y="240" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="350" y="240" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="385" y="267" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  <rect x="420" y="240" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <!-- Row 6 (EfficientNet) -->
  <rect x="70" y="280" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="140" y="280" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="210" y="280" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="280" y="280" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="350" y="280" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="420" y="280" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="455" y="307" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  
  <!-- Color scale legend - MOVED TO TOP RIGHT -->
  <rect x="480" y="65" width="155" height="110" fill="#fff" opacity="0.95" stroke="#d8d3c8" stroke-width="1" rx="3"/>
  
  <text x="495" y="82" font-size="11" font-weight="bold" fill="#1a1714">Transfer Rate</text>
  
  <rect x="495" y="95" width="18" height="18" fill="#FF0000" opacity="0.9"/>
  <text x="520" y="108" font-size="10" fill="#1a1714" font-weight="bold">100%</text>
  
  <rect x="495" y="120" width="18" height="18" fill="#FFA500" opacity="0.7"/>
  <text x="520" y="133" font-size="10" fill="#1a1714">70%</text>
  
  <rect x="495" y="145" width="18" height="18" fill="#FFD700" opacity="0.6"/>
  <text x="520" y="158" font-size="10" fill="#1a1714">50%</text>
</svg>
"""

print("Fixing Figure 8 legend...")
with open('temp_fig8.svg', 'w') as f:
    f.write(fig8_svg)

try:
    cairosvg.svg2pdf(url='temp_fig8.svg', write_to='figures/figure8_transferability.pdf', dpi=300)
    size = os.path.getsize('figures/figure8_transferability.pdf') / 1024
    print(f"✓ Fixed: figure8_transferability.pdf ({size:.1f} KB)")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    os.remove('temp_fig8.svg')
except:
    pass

print("Done!")
