import React, { useState } from 'react';
import { X, HelpCircle, Keyboard } from 'lucide-react';

export default function HelpModal({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('help');

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'rgba(0,0,0,0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: '#111318',
        borderRadius: '12px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        maxWidth: '600px',
        width: '90%',
        maxHeight: '80vh',
        overflow: 'auto',
        border: '1px solid #1e2430'
      }}>
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '20px',
          borderBottom: '1px solid #1e2430'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <HelpCircle size={24} style={{ color: '#00e5ff' }} />
            <h2 style={{ margin: 0, color: '#e2e8f0' }}>Help & Shortcuts</h2>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#64748b',
              padding: '8px'
            }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Tabs */}
        <div style={{
          display: 'flex',
          gap: '0',
          borderBottom: '1px solid #1e2430',
          backgroundColor: '#0a0c10'
        }}>
          <button
            onClick={() => setActiveTab('help')}
            style={{
              flex: 1,
              padding: '12px',
              background: activeTab === 'help' ? '#00e5ff' : 'transparent',
              color: activeTab === 'help' ? '#0a0c10' : '#64748b',
              border: 'none',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.2s'
            }}
          >
            <HelpCircle size={16} style={{ display: 'inline', marginRight: '8px' }} />
            Getting Started
          </button>
          <button
            onClick={() => setActiveTab('shortcuts')}
            style={{
              flex: 1,
              padding: '12px',
              background: activeTab === 'shortcuts' ? '#00e5ff' : 'transparent',
              color: activeTab === 'shortcuts' ? '#0a0c10' : '#64748b',
              border: 'none',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.2s'
            }}
          >
            <Keyboard size={16} style={{ display: 'inline', marginRight: '8px' }} />
            Shortcuts
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: '20px', color: '#e2e8f0' }}>
          {activeTab === 'help' && (
            <div>
              <h3 style={{ marginTop: 0, color: '#00e5ff' }}>Three Core Features</h3>
              
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ color: '#a8ff78' }}>Attack</h4>
                <p style={{ margin: '8px 0', lineHeight: '1.6' }}>
                  Generate adversarial examples on clean images using various attack algorithms (FGSM, PGD, C&W, DeepFool, JSMA). Lower epsilon = subtle perturbations, higher epsilon = stronger attacks.
                </p>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ color: '#a8ff78' }}>Defense</h4>
                <p style={{ margin: '8px 0', lineHeight: '1.6' }}>
                  Compare standard vs adversarially trained (robust) models. Robust models are trained to resist attacks, showing significantly higher accuracy under adversarial perturbations.
                </p>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ color: '#a8ff78' }}>Transfer Analysis</h4>
                <p style={{ margin: '8px 0', lineHeight: '1.6' }}>
                  Test if attacks generated on one architecture succeed on others. Lower transfer rates indicate architectural diversity provides natural defense.
                </p>
              </div>

              <h3 style={{ marginTop: '20px', color: '#00e5ff' }}>Model Architectures</h3>
              <p style={{ margin: '8px 0', fontSize: '0.9em', color: '#64748b' }}>
                4 architectures available: ResNet-18/50, VGG-16, DenseNet-121
              </p>
            </div>
          )}

          {activeTab === 'shortcuts' && (
            <div>
              <h3 style={{ marginTop: 0, color: '#00e5ff' }}>Keyboard Shortcuts</h3>
              
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <tbody>
                  {[
                    ['1', 'Go to Attack tab'],
                    ['2', 'Go to Defense tab'],
                    ['3', 'Go to Transfer tab'],
                    ['4', 'Go to History tab'],
                    ['?', 'Open this help'],
                    ['Escape', 'Close modals'],
                    ['Enter', 'Submit form (when focused)']
                  ].map(([key, desc], idx) => (
                    <tr key={idx} style={{
                      borderBottom: '1px solid #1e2430',
                      backgroundColor: idx % 2 === 0 ? '#181c24' : 'transparent'
                    }}>
                      <td style={{ padding: '12px', fontFamily: 'monospace', color: '#00e5ff', fontWeight: '600' }}>
                        {key}
                      </td>
                      <td style={{ padding: '12px', color: '#e2e8f0' }}>
                        {desc}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <p style={{ marginTop: '20px', fontSize: '0.9em', color: '#64748b' }}>
                Tip: Focus on input fields and press Enter to run experiments faster.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
