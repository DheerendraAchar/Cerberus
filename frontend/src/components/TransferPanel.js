import React, { useState } from 'react';
import { ArrowRight, Cpu, Sliders, AlertTriangle } from 'lucide-react';
import StyledSelect from './StyledSelect';

export default function TransferPanel({ onRun, loading, dataset = 'cifar10' }) {
  const [params, setParams] = useState({
    attack: 'fgsm',
    source_architecture: dataset === 'cifar10' ? 'ResNet-18' : 'TextCNN',
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
      source_architecture: dataset === 'cifar10' ? 'ResNet-18' : 'TextCNN'
    }));
  }, [dataset]);

  const epsilonDisplay = (params.epsilon * 255).toFixed(0);

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <ArrowRight size={24} style={{ color: '#667eea' }} />
        <h2 style={{ margin: 0 }}>Transfer Analysis</h2>
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

        <StyledSelect
          label="Source Architecture (attacker trains on)"
          name="source_architecture"
          value={params.source_architecture}
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
                  { value: 'VGG-16', label: 'VGG-16' }
                ]
              },
              {
                label: 'Other Supported Models',
                options: [
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

        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={16} style={{ color: '#667eea' }} />
            Attack Strength (ε)
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
              <strong>Transfer Attack:</strong> Generate attack on one model, test on all others. Shows architectural diversity as defense!
              <br /><br />
              <strong>Key Finding:</strong> If attacks from ResNet only succeed 83.7% on ResNet but only 70.9% on average on other architectures,
              that 12.8pp gap shows architectural diversity is a natural defense!
            </>
          ) : (
            <>
              <strong>AG News Transfer:</strong> Generates perturbations on one text architecture and measures cross-model transferability across text classifiers.
            </>
          )}
        </div>

        <button type="submit" className="btn btn-info" disabled={loading}>
          {loading && <span className="spinner"></span>}
          {loading ? 'Analyzing...' : 'Analyze Transfer'}
        </button>
      </form>
    </div>
  );
}
