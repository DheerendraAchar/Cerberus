import React, { useState } from 'react';
import { Shield, AlertTriangle, Cpu, Sliders } from 'lucide-react';
import StyledSelect from './StyledSelect';

export default function DefensePanel({ onRun, loading, dataset = 'cifar10' }) {
  const [params, setParams] = useState({
    attack: 'fgsm',
    architecture: dataset === 'cifar10' ? 'ResNet-18' : 'TextCNN',
    epsilon: 0.031,
    num_samples: 128,
    dataset: dataset
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onRun({ ...params, dataset });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams(prev => ({
      ...prev,
      [name]: name === 'epsilon' || name === 'num_samples' ? parseFloat(value) : value
    }));
  };

  // Update architecture when dataset changes
  React.useEffect(() => {
    setParams(prev => ({
      ...prev,
      dataset: dataset,
      attack: 'fgsm',
      architecture: dataset === 'cifar10' ? 'ResNet-18' : 'TextCNN'
    }));
  }, [dataset]);

  const epsilonDisplay = (params.epsilon * 255).toFixed(0);

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <Shield size={24} style={{ color: '#667eea' }} />
        <h2 style={{ margin: 0 }}>Compare Defenses</h2>
        <span style={{ marginLeft: 'auto', padding: '4px 8px', background: '#667eea', color: 'white', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold' }}>
          {dataset === 'cifar10' ? 'CIFAR-10 (Images)' : 'AG News (Text)'}
        </span>
      </div>

      <form onSubmit={handleSubmit}>
        <StyledSelect
          label="Attack Type"
          name="attack"
          value={params.attack}
          onChange={handleChange}
          disabled={loading}
          icon={AlertTriangle}
          groupedOptions={
            dataset === 'cifar10' ? [
              {
                label: 'Gradient-Based (Basic)',
                options: [{ value: 'fgsm', label: 'FGSM - Fast Gradient Sign' }]
              },
              {
                label: 'Gradient-Based (Iterative)',
                options: [{ value: 'pgd', label: 'PGD - Projected Gradient Descent' }]
              },
              {
                label: 'Optimization-Based',
                options: [{ value: 'cw', label: 'C&W - Carlini & Wagner' }]
              },
              {
                label: 'Geometric',
                options: [{ value: 'deepfool', label: 'DeepFool - Minimal Perturbation' }]
              },
              {
                label: 'Feature-Based',
                options: [{ value: 'jsma', label: 'JSMA - Jacobian Saliency Map' }]
              },
              {
                label: 'Advanced Ensemble',
                options: [
                  { value: 'autoattack', label: 'AutoAttack - Ensemble Attack' },
                  { value: 'square', label: 'Square - Query-Based Black-Box' },
                  { value: 'fab', label: 'FAB - Fast Adaptive Boundary' },
                  { value: 'rays', label: 'RayS - Ray Search Boundary' },
                  { value: 'trades', label: 'TRADES - KL-Robustness Attack' }
                ]
              }
            ] : [
              {
                label: 'Gradient-Based',
                options: [
                  { value: 'fgsm', label: 'FGSM - Token Perturbation' },
                  { value: 'pgd', label: 'PGD - Iterative Token Perturbation' }
                ]
              },
              {
                label: 'Structural',
                options: [
                  { value: 'tokenswap', label: 'Token Swap - Adjacent Token Swapping' },
                  { value: 'tokennoise', label: 'Token Noise - Random Perturbation' }
                ]
              },
              {
                label: 'Semantic',
                options: [
                  { value: 'tokensubstitution', label: 'Token Substitution - Vocabulary Replacement' }
                ]
              }
            ]
          }
        />

        <div style={{ marginTop: '20px' }}>
          <StyledSelect
            label="Architecture"
            name="architecture"
            value={params.architecture}
            onChange={handleChange}
            disabled={loading}
            icon={Cpu}
            groupedOptions={
              dataset === 'cifar10' ? [
                {
                  label: 'Traditional CNNs',
                  options: [
                    { value: 'ResNet-18', label: 'ResNet-18' },
                    { value: 'ResNet-50', label: 'ResNet-50' },
                    { value: 'VGG-16', label: 'VGG-16' },
                    { value: 'DenseNet-121', label: 'DenseNet-121' }
                  ]
                }
              ] : [
                {
                  label: 'CNN-based',
                  options: [{ value: 'TextCNN', label: 'TextCNN' }]
                },
                {
                  label: 'Recurrent',
                  options: [
                    { value: 'LSTM', label: 'LSTM' },
                    { value: 'BiLSTM', label: 'BiLSTM' }
                  ]
                },
                {
                  label: 'Attention-based',
                  options: [{ value: 'Transformer', label: 'Transformer' }]
                }
              ]
            }
          />
        </div>

        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={16} style={{ color: '#667eea' }} />
            Perturbation Strength (ε)
          </label>
          <input
            type="range"
            name="epsilon"
            min="0"
            max="0.3"
            step="0.001"
            value={params.epsilon}
            onChange={handleChange}
            disabled={loading}
          />
          <div className="range-display">
            <span>Weak</span>
            <span>{params.epsilon.toFixed(4)} ({epsilonDisplay}/255)</span>
            <span>Strong</span>
          </div>
        </div>

        <div className="form-group">
          <label>Number of Examples</label>
          <input
            type="number"
            name="num_samples"
            min="16"
            max="1000"
            step="16"
            value={params.num_samples}
            onChange={handleChange}
            disabled={loading}
          />
        </div>

        <div className="demo-info">
          {dataset === 'cifar10' ? (
            <>
              <strong>What You'll See:</strong> A baseline-vs-robust comparison. Red shows the standard model, green shows the robust checkpoint, so you can see the defense gap directly.
            </>
          ) : (
            <>
              <strong>AG News Defense:</strong> Compares the standard text model against the robust checkpoint under token-level adversarial perturbations.
            </>
          )}
        </div>

        <button type="submit" className="btn btn-success" disabled={loading}>
          {loading && <span className="spinner"></span>}
          {loading ? 'Comparing...' : 'Compare Defenses'}
        </button>
      </form>
    </div>
  );
}
