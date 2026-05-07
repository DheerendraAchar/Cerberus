import React from 'react';
import { ArrowRight, Shield, Brain, LineChart, LayoutDashboard } from 'lucide-react';

const steps = [
  { icon: LayoutDashboard, title: 'Benchmark Input', text: 'Choose dataset, architecture, and attack settings.' },
  { icon: Shield, title: 'Attack Run', text: 'Measure clean accuracy, adversarial accuracy, and drop.' },
  { icon: Brain, title: 'Retraining Trigger', text: 'If the drop crosses the threshold, retraining starts.' },
  { icon: LineChart, title: 'Defense Review', text: 'Compare before/after performance and transferability.' }
];

export default function PipelineDiagram() {
  return (
    <div className="insight-box" style={{ marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
        <LayoutDashboard size={18} />
        <strong>Attack → Retrain → Defend Pipeline</strong>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '12px', alignItems: 'stretch' }}>
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={step.title}>
              <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                  <Icon size={18} />
                  <strong>{step.title}</strong>
                </div>
                <p style={{ margin: 0, lineHeight: 1.6, opacity: 0.8 }}>{step.text}</p>
              </div>
              {index < steps.length - 1 && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.7 }}>
                  <ArrowRight size={22} />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
