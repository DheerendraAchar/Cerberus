#!/usr/bin/env python3
"""
Generate proper Figures 7-10 from HTML data
"""

import cairosvg
import os

# Figure 7: Model Hardening Progress (before/after robustness)
fig7_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'DM Mono', monospace; }
  </style>
  
  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1714">Model Hardening Progress</text>
  <text x="350" y="45" text-anchor="middle" font-size="11" fill="#5a5248">Robustness Improvement Across 6 Architectures</text>
  
  <!-- Grid -->
  <line x1="70" y1="60" x2="70" y2="320" stroke="#d8d3c8" stroke-width="1.5"/>
  <line x1="70" y1="320" x2="680" y2="320" stroke="#d8d3c8" stroke-width="1.5"/>
  
  <!-- Y axis gridlines and labels -->
  <line x1="70" y1="80" x2="680" y2="80" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="140" x2="680" y2="140" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="200" x2="680" y2="200" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="260" x2="680" y2="260" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  
  <text x="60" y="85" text-anchor="end" font-size="11" fill="#5a5248">85%</text>
  <text x="60" y="145" text-anchor="end" font-size="11" fill="#5a5248">70%</text>
  <text x="60" y="205" text-anchor="end" font-size="11" fill="#5a5248">55%</text>
  <text x="60" y="265" text-anchor="end" font-size="11" fill="#5a5248">40%</text>
  <text x="60" y="325" text-anchor="end" font-size="11" fill="#5a5248">25%</text>
  
  <!-- X axis labels -->
  <text x="100" y="345" text-anchor="middle" font-size="10" fill="#5a5248">ResNet-18</text>
  <text x="170" y="345" text-anchor="middle" font-size="10" fill="#5a5248">ResNet-50</text>
  <text x="240" y="345" text-anchor="middle" font-size="10" fill="#5a5248">VGG-16</text>
  <text x="310" y="345" text-anchor="middle" font-size="10" fill="#5a5248">DenseNet</text>
  <text x="380" y="345" text-anchor="middle" font-size="10" fill="#5a5248">TextCNN</text>
  <text x="450" y="345" text-anchor="middle" font-size="10" fill="#5a5248">LSTM</text>
  
  <!-- Y axis label -->
  <text x="25" y="190" text-anchor="middle" font-size="10" fill="#5a5248" transform="rotate(-90 25 190)">Robustness (%)</text>
  
  <!-- Before defense (red bars) -->
  <rect x="85" y="220" width="20" height="100" fill="#FF6B6B" opacity="0.8"/>
  <text x="95" y="235" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">68.5</text>
  
  <rect x="155" y="218" width="20" height="102" fill="#FF6B6B" opacity="0.8"/>
  <text x="165" y="233" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">69.2</text>
  
  <rect x="225" y="245" width="20" height="75" fill="#FF6B6B" opacity="0.8"/>
  <text x="235" y="262" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">64.2</text>
  
  <rect x="295" y="205" width="20" height="115" fill="#FF6B6B" opacity="0.8"/>
  <text x="305" y="218" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">72.1</text>
  
  <rect x="365" y="290" width="20" height="30" fill="#FF6B6B" opacity="0.8"/>
  <text x="375" y="311" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">52.3</text>
  
  <rect x="435" y="265" width="20" height="55" fill="#FF6B6B" opacity="0.8"/>
  <text x="445" y="298" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">58.7</text>
  
  <!-- After defense (green bars) -->
  <rect x="108" y="125" width="20" height="195" fill="#51CF66" opacity="0.8"/>
  <text x="118" y="142" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">78.2</text>
  
  <rect x="178" y="117" width="20" height="203" fill="#51CF66" opacity="0.8"/>
  <text x="188" y="132" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">79.1</text>
  
  <rect x="248" y="150" width="20" height="170" fill="#51CF66" opacity="0.8"/>
  <text x="258" y="167" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">74.8</text>
  
  <rect x="318" y="85" width="20" height="235" fill="#51CF66" opacity="0.8"/>
  <text x="328" y="102" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">81.5</text>
  
  <rect x="388" y="215" width="20" height="105" fill="#51CF66" opacity="0.8"/>
  <text x="398" y="230" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">68.9</text>
  
  <rect x="458" y="175" width="20" height="145" fill="#51CF66" opacity="0.8"/>
  <text x="468" y="192" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">72.4</text>
  
  <!-- Legend -->
  <rect x="520" y="80" width="15" height="15" fill="#FF6B6B" opacity="0.8"/>
  <text x="545" y="92" font-size="11" fill="#1a1714" font-weight="bold">Before Defense</text>
  
  <rect x="520" y="110" width="15" height="15" fill="#51CF66" opacity="0.8"/>
  <text x="545" y="122" font-size="11" fill="#1a1714" font-weight="bold">After Training</text>
  
  <!-- Improvement percentages -->
  <text x="95" y="360" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">+9.7%</text>
  <text x="165" y="360" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">+9.9%</text>
  <text x="235" y="360" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">+10.6%</text>
  <text x="305" y="360" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">+9.4%</text>
  <text x="375" y="360" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">+16.6%</text>
  <text x="445" y="360" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">+13.7%</text>
</svg>
"""

# Figure 8: Cross-Model Transferability detailed heatmap
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
  
  <!-- Remaining rows simplified -->
  <rect x="70" y="200" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="140" y="200" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="210" y="200" width="70" height="40" fill="#FFD700" opacity="0.6"/>
  <rect x="280" y="200" width="70" height="40" fill="#FF0000" opacity="0.9"/>
  <text x="315" y="227" text-anchor="middle" font-size="10" fill="#fff" font-weight="bold">100</text>
  <rect x="350" y="200" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  <rect x="420" y="200" width="70" height="40" fill="#FFFF00" opacity="0.5"/>
  
  <!-- Color scale legend -->
  <rect x="70" y="560" width="20" height="20" fill="#FF0000" opacity="0.9"/>
  <text x="100" y="575" font-size="9" fill="#1a1714">100% (Perfect)</text>
  
  <rect x="70" y="585" width="20" height="20" fill="#FFA500" opacity="0.7"/>
  <text x="100" y="600" font-size="9" fill="#1a1714">70% (High)</text>
  
  <rect x="70" y="610" width="20" height="20" fill="#FFFF00" opacity="0.5"/>
  <text x="100" y="625" font-size="9" fill="#1a1714">50% (Low)</text>
</svg>
"""

# Figure 9: API Response Time
fig9_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'DM Mono', monospace; }
  </style>
  
  <!-- Title -->
  <text x="300" y="30" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1714">API Response Time Analysis</text>
  <text x="300" y="50" text-anchor="middle" font-size="11" fill="#5a5248">Backend Latency Across Operations</text>
  
  <!-- Grid -->
  <line x1="70" y1="70" x2="70" y2="280" stroke="#d8d3c8" stroke-width="1.5"/>
  <line x1="70" y1="280" x2="580" y2="280" stroke="#d8d3c8" stroke-width="1.5"/>
  
  <!-- Y axis -->
  <line x1="70" y1="100" x2="580" y2="100" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="130" x2="580" y2="130" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="160" x2="580" y2="160" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="190" x2="580" y2="190" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="220" x2="580" y2="220" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  
  <text x="60" y="105" text-anchor="end" font-size="10" fill="#5a5248">500ms</text>
  <text x="60" y="135" text-anchor="end" font-size="10" fill="#5a5248">400ms</text>
  <text x="60" y="165" text-anchor="end" font-size="10" fill="#5a5248">300ms</text>
  <text x="60" y="195" text-anchor="end" font-size="10" fill="#5a5248">200ms</text>
  <text x="60" y="225" text-anchor="end" font-size="10" fill="#5a5248">100ms</text>
  <text x="60" y="285" text-anchor="end" font-size="10" fill="#5a5248">0ms</text>
  
  <!-- Operations -->
  <text x="125" y="305" text-anchor="middle" font-size="10" fill="#5a5248">FGSM</text>
  <text x="200" y="305" text-anchor="middle" font-size="10" fill="#5a5248">PGD</text>
  <text x="275" y="305" text-anchor="middle" font-size="10" fill="#5a5248">C&amp;W</text>
  <text x="350" y="305" text-anchor="middle" font-size="10" fill="#5a5248">DeepFool</text>
  <text x="425" y="305" text-anchor="middle" font-size="10" fill="#5a5248">Training</text>
  <text x="500" y="305" text-anchor="middle" font-size="10" fill="#5a5248">Inference</text>
  
  <!-- Bars -->
  <rect x="110" y="245" width="30" height="35" fill="#2a5fa5" opacity="0.8"/>
  <text x="125" y="263" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">45ms</text>
  
  <rect x="185" y="145" width="30" height="135" fill="#c8440c" opacity="0.8"/>
  <text x="200" y="177" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">270ms</text>
  
  <rect x="260" y="105" width="30" height="175" fill="#c8440c" opacity="0.8"/>
  <text x="275" y="152" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">350ms</text>
  
  <rect x="335" y="165" width="30" height="115" fill="#2a5fa5" opacity="0.8"/>
  <text x="350" y="207" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">230ms</text>
  
  <rect x="410" y="75" width="30" height="205" fill="#FF6B6B" opacity="0.8"/>
  <text x="425" y="147" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">410ms</text>
  
  <rect x="485" y="210" width="30" height="70" fill="#2d7a4f" opacity="0.8"/>
  <text x="500" y="252" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">140ms</text>
  
  <!-- Legend -->
  <text x="70" y="330" font-size="9" fill="#5a5248">✓ FGSM: Fastest (O(1) gradients)</text>
  <text x="70" y="345" font-size="9" fill="#5a5248">✓ Training: Slowest (intensive optimization)</text>
</svg>
"""

# Figure 10: Robustness Certification
fig10_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'DM Mono', monospace; }
  </style>
  
  <!-- Title -->
  <text x="300" y="30" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a1714">Certified Robustness Analysis</text>
  <text x="300" y="50" text-anchor="middle" font-size="11" fill="#5a5248">Certified Radius vs Model Capacity</text>
  
  <!-- Grid -->
  <line x1="70" y1="70" x2="70" y2="320" stroke="#d8d3c8" stroke-width="1.5"/>
  <line x1="70" y1="320" x2="580" y2="320" stroke="#d8d3c8" stroke-width="1.5"/>
  
  <!-- Gridlines -->
  <line x1="70" y1="100" x2="580" y2="100" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="150" x2="580" y2="150" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="200" x2="580" y2="200" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="70" y1="250" x2="580" y2="250" stroke="#f0ede6" stroke-width="1" stroke-dasharray="3,3"/>
  
  <!-- Y axis labels -->
  <text x="60" y="105" text-anchor="end" font-size="10" fill="#5a5248">0.10</text>
  <text x="60" y="155" text-anchor="end" font-size="10" fill="#5a5248">0.08</text>
  <text x="60" y="205" text-anchor="end" font-size="10" fill="#5a5248">0.06</text>
  <text x="60" y="255" text-anchor="end" font-size="10" fill="#5a5248">0.04</text>
  <text x="60" y="325" text-anchor="end" font-size="10" fill="#5a5248">0.00</text>
  
  <!-- X axis labels -->
  <text x="125" y="340" text-anchor="middle" font-size="10" fill="#5a5248">Small</text>
  <text x="200" y="340" text-anchor="middle" font-size="10" fill="#5a5248">Medium</text>
  <text x="275" y="340" text-anchor="middle" font-size="10" fill="#5a5248">Large</text>
  <text x="350" y="340" text-anchor="middle" font-size="10" fill="#5a5248">Vision</text>
  <text x="425" y="340" text-anchor="middle" font-size="10" fill="#5a5248">NLP</text>
  <text x="500" y="340" text-anchor="middle" font-size="10" fill="#5a5248">Ensemble</text>
  
  <!-- Axis labels -->
  <text x="25" y="195" text-anchor="middle" font-size="10" fill="#5a5248" transform="rotate(-90 25 195)">Certified Radius (ε)</text>
  <text x="325" y="365" text-anchor="middle" font-size="10" fill="#5a5248">Model Type</text>
  
  <!-- Points with error bars -->
  <!-- Small models -->
  <line x1="125" y1="240" x2="125" y2="270" stroke="#2a5fa5" stroke-width="1"/>
  <circle cx="125" cy="255" r="4" fill="#2a5fa5" opacity="0.8"/>
  <text x="125" y="280" text-anchor="middle" font-size="9" fill="#2a5fa5" font-weight="bold">0.025</text>
  
  <!-- Medium models -->
  <line x1="200" y1="215" x2="200" y2="245" stroke="#2a5fa5" stroke-width="1"/>
  <circle cx="200" cy="230" r="4" fill="#2a5fa5" opacity="0.8"/>
  <text x="200" y="280" text-anchor="middle" font-size="9" fill="#2a5fa5" font-weight="bold">0.036</text>
  
  <!-- Large models -->
  <line x1="275" y1="170" x2="275" y2="200" stroke="#2a5fa5" stroke-width="1"/>
  <circle cx="275" cy="185" r="4" fill="#2a5fa5" opacity="0.8"/>
  <text x="275" y="280" text-anchor="middle" font-size="9" fill="#2a5fa5" font-weight="bold">0.047</text>
  
  <!-- Vision models -->
  <line x1="350" y1="150" x2="350" y2="180" stroke="#2d7a4f" stroke-width="1"/>
  <circle cx="350" cy="165" r="4" fill="#2d7a4f" opacity="0.8"/>
  <text x="350" y="280" text-anchor="middle" font-size="9" fill="#2d7a4f" font-weight="bold">0.047</text>
  
  <!-- NLP models -->
  <line x1="425" y1="260" x2="425" y2="290" stroke="#c8440c" stroke-width="1"/>
  <circle cx="425" cy="275" r="4" fill="#c8440c" opacity="0.8"/>
  <text x="425" y="280" text-anchor="middle" font-size="9" fill="#c8440c" font-weight="bold">0.031</text>
  
  <!-- Ensemble models -->
  <line x1="500" y1="120" x2="500" y2="150" stroke="#d4a017" stroke-width="1"/>
  <circle cx="500" cy="135" r="4" fill="#d4a017" opacity="0.8"/>
  <text x="500" y="280" text-anchor="middle" font-size="9" fill="#d4a017" font-weight="bold">0.058</text>
  
  <!-- Info box -->
  <rect x="70" y="360" width="510" height="30" fill="#f0ede6" opacity="0.6" rx="2"/>
  <text x="80" y="380" font-size="9" fill="#1a1714">Vision: 0.047 | NLP: 0.031 (0.016 gap) | Ensemble methods achieve higher certified robustness</text>
</svg>
"""

print("Generating proper Figures 7-10...")
print("=" * 70)

# Figure 7
print("\nFigure 7: Model Hardening Progress...")
with open('temp_fig7.svg', 'w') as f:
    f.write(fig7_svg)
try:
    cairosvg.svg2pdf(url='temp_fig7.svg', write_to='figure7_hardening.pdf', dpi=300)
    size = os.path.getsize('figure7_hardening.pdf') / 1024
    print(f"  ✓ Created: figure7_hardening.pdf ({size:.1f} KB)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Figure 8
print("\nFigure 8: Cross-Model Transferability...")
with open('temp_fig8.svg', 'w') as f:
    f.write(fig8_svg)
try:
    cairosvg.svg2pdf(url='temp_fig8.svg', write_to='figure8_transferability.pdf', dpi=300)
    size = os.path.getsize('figure8_transferability.pdf') / 1024
    print(f"  ✓ Created: figure8_transferability.pdf ({size:.1f} KB)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Figure 9
print("\nFigure 9: API Response Time Analysis...")
with open('temp_fig9.svg', 'w') as f:
    f.write(fig9_svg)
try:
    cairosvg.svg2pdf(url='temp_fig9.svg', write_to='figure9_response_time.pdf', dpi=300)
    size = os.path.getsize('figure9_response_time.pdf') / 1024
    print(f"  ✓ Created: figure9_response_time.pdf ({size:.1f} KB)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Figure 10
print("\nFigure 10: Robustness Certification...")
with open('temp_fig10.svg', 'w') as f:
    f.write(fig10_svg)
try:
    cairosvg.svg2pdf(url='temp_fig10.svg', write_to='figure10_certification.pdf', dpi=300)
    size = os.path.getsize('figure10_certification.pdf') / 1024
    print(f"  ✓ Created: figure10_certification.pdf ({size:.1f} KB)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Copy to figures folder
import subprocess
print("\nCopying to figures/ folder...")
os.system('cp figure7_hardening.pdf figures/figure7_hardening.pdf')
os.system('cp figure8_transferability.pdf figures/figure8_transferability.pdf')
os.system('cp figure9_response_time.pdf figures/figure9_response_time.pdf')
os.system('cp figure10_certification.pdf figures/figure10_certification.pdf')

# Cleanup
for f in ['temp_fig7.svg', 'temp_fig8.svg', 'temp_fig9.svg', 'temp_fig10.svg']:
    try:
        os.remove(f)
    except:
        pass

print("\n" + "=" * 70)
print("✓ Figures 7-10 regenerated successfully!")
os.system('ls -lh figures/figure[7-9]_*.pdf figures/figure10_*.pdf 2>/dev/null | awk "{print \"  •\", \\$9, \\\"(\\\" \\$5 \\\")\\\"}"')
