import React from 'react';
import { Zap, BarChart3, Clock, Database, AlertTriangle, Zap as ZapIcon, ShieldCheck, Activity } from 'lucide-react';

const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;

export default function ResultsPanel({ results, retrainingStatus }) {
  if (!results) return null;

  const resolvedAttack = (
    results.attack_type ||
    results.attack ||
    results.source_attack ||
    'unknown'
  ).toString();
  const resolvedArchitecture =
    results.architecture ||
    results.source_architecture ||
    results.model ||
    'selected model';
  const resolvedEpsilon =
    results.epsilon != null
      ? Number(results.epsilon)
      : (results.eps != null ? Number(results.eps) : null);
  const analysisMode = (results.analysis_mode || results.result_mode || results.view_mode || '').toString().toLowerCase();
  const isDefenseRun = analysisMode === 'defense' || results.standard_adversarial_accuracy != null || results.robust_adversarial_accuracy != null;
  const isTransferRun = analysisMode === 'transfer' || results.transfer_results != null || results.insight != null;

  const cleanAccuracy = Number(results.clean_accuracy || 0);
  const standardCleanAccuracy = Number(results.standard_clean_accuracy ?? results.clean_accuracy ?? 0);
  const adversarialAccuracy = Number(results.adversarial_accuracy || 0);
  const standardAdversarialAccuracy = results.standard_adversarial_accuracy != null ? Number(results.standard_adversarial_accuracy) : null;
  const robustAdversarialAccuracy = results.robust_adversarial_accuracy != null ? Number(results.robust_adversarial_accuracy) : (isDefenseRun ? adversarialAccuracy : null);
  const defenseGain = standardAdversarialAccuracy != null && robustAdversarialAccuracy != null
    ? robustAdversarialAccuracy - standardAdversarialAccuracy
    : null;
  const accuracyDropRaw = results.accuracy_drop != null ? results.accuracy_drop : (cleanAccuracy - adversarialAccuracy);
  const accuracyDrop = Number(accuracyDropRaw || 0);
  const retrainingImprovement = Number(retrainingStatus?.improvement || 0);
  const transferGap = results.statistics?.defense_gap != null
    ? Number(results.statistics.defense_gap) / 100
    : cleanAccuracy - adversarialAccuracy;
  const modelSource = (results.model_source || '').toString();
  const usingFallback = modelSource === 'fallback_random_head';

  const benchmarkMetrics = isDefenseRun
    ? [
        { label: 'Baseline Clean', value: formatPercent(standardCleanAccuracy), tone: 'positive' },
        { label: 'Baseline Under Attack', value: standardAdversarialAccuracy != null ? formatPercent(standardAdversarialAccuracy) : '—', tone: 'negative' },
        { label: 'Robust Under Attack', value: robustAdversarialAccuracy != null ? formatPercent(robustAdversarialAccuracy) : '—', tone: 'positive' },
        { label: 'Defense Gain', value: defenseGain != null ? `${(defenseGain * 100).toFixed(1)}%` : '—', tone: defenseGain != null && defenseGain >= 0 ? 'positive' : 'negative' },
        { label: 'Attack / Epsilon', value: `${resolvedAttack.toUpperCase()} @ ${resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—'}`, tone: 'warning' }
      ]
    : isTransferRun
      ? [
          { label: 'Source Clean', value: formatPercent(cleanAccuracy), tone: 'positive' },
          { label: 'Average Transfer Success', value: `${((Number(results.adversarial_accuracy || 0)) * 100).toFixed(1)}%`, tone: 'warning' },
          { label: 'Target Models', value: results.transfer_results ? String(Object.keys(results.transfer_results).length) : '—', tone: 'positive' },
          { label: 'Attack / Epsilon', value: `${resolvedAttack.toUpperCase()} @ ${resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—'}`, tone: 'warning' }
      ]
    : [
        { label: 'Clean Accuracy', value: formatPercent(cleanAccuracy), tone: 'positive' },
        { label: 'Adversarial Accuracy', value: formatPercent(adversarialAccuracy), tone: 'negative' },
        { label: 'Accuracy Drop', value: `${(accuracyDrop * 100).toFixed(1)}%`, tone: accuracyDrop > 0.25 ? 'negative' : '' },
        { label: 'Retraining Improvement', value: retrainingStatus ? `${(retrainingImprovement * 100).toFixed(1)}%` : '—', tone: retrainingImprovement > 0 ? 'positive' : '' },
        { label: 'Transferability Gap', value: `${Math.abs(transferGap * 100).toFixed(1)}%`, tone: 'warning' }
      ];

  const summaryRows = isDefenseRun
    ? [
        { metric: 'Mode', value: 'Standard vs Robust Defense' },
        { metric: 'Attack', value: resolvedAttack.toUpperCase() },
        { metric: 'Architecture', value: resolvedArchitecture },
        { metric: 'Epsilon', value: resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—' },
        { metric: 'Evaluated Samples', value: results.num_samples != null ? String(results.num_samples) : '—' },
        { metric: 'Baseline Clean Accuracy', value: formatPercent(standardCleanAccuracy) },
        { metric: 'Baseline Adversarial Accuracy', value: standardAdversarialAccuracy != null ? formatPercent(standardAdversarialAccuracy) : '—' },
        { metric: 'Robust Adversarial Accuracy', value: robustAdversarialAccuracy != null ? formatPercent(robustAdversarialAccuracy) : '—' },
        { metric: 'Defense Gain', value: defenseGain != null ? `${(defenseGain * 100).toFixed(1)}%` : '—' }
      ]
    : isTransferRun
      ? [
          { metric: 'Mode', value: 'Cross-Model Transfer Study' },
          { metric: 'Source Architecture', value: results.source_architecture || resolvedArchitecture },
          { metric: 'Attack', value: resolvedAttack.toUpperCase() },
          { metric: 'Epsilon', value: resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—' },
          { metric: 'Evaluated Samples', value: results.num_samples != null ? String(results.num_samples) : '—' },
          { metric: 'Source Clean Accuracy', value: formatPercent(cleanAccuracy) },
          { metric: 'Average Transfer Success', value: `${(Number(results.adversarial_accuracy || 0) * 100).toFixed(1)}%` },
          { metric: 'Target Models', value: results.transfer_results ? String(Object.keys(results.transfer_results).length) : '—' }
        ]
    : [
        { metric: 'Attack', value: resolvedAttack.toUpperCase() },
        { metric: 'Architecture', value: resolvedArchitecture },
        { metric: 'Epsilon', value: resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—' },
        { metric: 'Evaluated Samples', value: results.num_samples != null ? String(results.num_samples) : '—' },
        { metric: 'Retrain Trigger', value: results.retrain_triggered ? 'Triggered' : 'Not triggered' },
        { metric: 'Clean Accuracy', value: formatPercent(cleanAccuracy) },
        { metric: 'Adversarial Accuracy', value: formatPercent(adversarialAccuracy) },
        { metric: 'Accuracy Drop', value: `${(accuracyDrop * 100).toFixed(1)}%` }
      ];

  const transferRows = results.transfer_results
    ? Object.entries(results.transfer_results).map(([architecture, successRate]) => ({
        architecture,
        successRate: `${(Number(successRate) * 100).toFixed(1)}%`
      }))
    : [];
  const missingArchitectures = Array.isArray(results.missing_architectures) ? results.missing_architectures : [];

  const latestRun = retrainingStatus?.before_acc != null ? retrainingStatus : null;
  const shellClassName = [
    'results-panel-shell',
    isDefenseRun ? 'results-panel--defense' : isTransferRun ? 'results-panel--transfer' : 'results-panel--attack'
  ].join(' ');

  return (
    <div className={shellClassName}>
      <div className="results-mode-banner">
        <span className="results-mode-kicker">
          {isDefenseRun ? 'Defense Mode' : isTransferRun ? 'Transfer Mode' : 'Attack Mode'}
        </span>
        <span className="results-mode-copy">
          {isDefenseRun
            ? 'Baseline vs robust checkpoint'
            : isTransferRun
              ? 'Cross-model transfer behavior'
              : 'Single-model adversarial benchmark'}
        </span>
      </div>

      <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {isDefenseRun ? <ShieldCheck size={20} style={{ display: 'inline-block', verticalAlign: 'middle' }} /> : isTransferRun ? <Activity size={20} style={{ display: 'inline-block', verticalAlign: 'middle' }} /> : <Zap size={20} style={{ display: 'inline-block', verticalAlign: 'middle' }} />}
        {isDefenseRun ? 'Defense Comparison Report' : isTransferRun ? 'Transferability Report' : 'Attack Results Report'}
      </h2>

      {isDefenseRun && (
        <div className="insight-box" style={{ marginTop: '14px', border: '1px solid rgba(102, 126, 234, 0.35)', background: 'linear-gradient(135deg, rgba(102,126,234,0.14), rgba(46,204,113,0.10))' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <ShieldCheck size={18} />
            <strong>What this tab is showing</strong>
          </div>
          <p style={{ margin: 0, lineHeight: 1.6 }}>
            This view compares the attacked baseline model against the robust checkpoint on the same samples, so the result is intentionally different from a normal attack benchmark.
          </p>
        </div>
      )}

      {isTransferRun && (
        <div className="insight-box" style={{ marginTop: '14px', border: '1px solid rgba(255, 193, 7, 0.35)', background: 'linear-gradient(135deg, rgba(255,193,7,0.14), rgba(102,126,234,0.08))' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Activity size={18} />
            <strong>What this tab is showing</strong>
          </div>
          <p style={{ margin: 0, lineHeight: 1.6 }}>
            This view shows whether an attack crafted on one source model transfers to the other models, so the important signal is cross-model success rather than single-model accuracy.
          </p>
        </div>
      )}

      {isTransferRun && missingArchitectures.length > 0 && (
        <div className="insight-box" style={{ marginTop: '20px', border: '1px solid rgba(230, 126, 34, 0.35)', background: 'rgba(230,126,34,0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <AlertTriangle size={18} style={{ color: '#e67e22' }} />
            <strong>Skipped Checkpoints</strong>
          </div>
          <p style={{ margin: 0 }}>
            The transfer comparison skipped these architectures because their checkpoints are missing: <strong>{missingArchitectures.join(', ')}</strong>.
          </p>
        </div>
      )}

      <div className="metrics-grid" style={isDefenseRun ? { marginTop: '18px' } : undefined}>
        {benchmarkMetrics.map((item) => (
          <div key={item.label} className="metric-card result-metric-card">
            <div className="metric-label">{item.label}</div>
            <div className={`metric-value ${item.tone === 'positive' ? 'positive' : item.tone === 'negative' ? 'negative' : ''}`}>
              {item.value}
            </div>
          </div>
        ))}
      </div>

      <div className="insight-box" style={{ marginTop: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <ShieldCheck size={18} />
          <strong>{isDefenseRun ? 'Defense Snapshot' : isTransferRun ? 'Transfer Snapshot' : 'Explainability Snapshot'}</strong>
        </div>
        <p style={{ margin: 0, lineHeight: 1.6 }}>
          {isDefenseRun ? (
            <>
              The baseline model and the robust model were attacked with <strong>{resolvedAttack.toUpperCase()}</strong> on{' '}
              <strong>{resolvedArchitecture}</strong> at ε = <strong>{resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—'}</strong>. The key point is the gap between baseline and robust accuracy.
            </>
          ) : isTransferRun ? (
            <>
              An attack was generated on <strong>{(results.source_architecture || resolvedArchitecture).toString()}</strong> and then evaluated across the other models. The important signal is how much the attack transfers, not just the source model’s own drop.
            </>
          ) : (
            <>
              Attack <strong>{resolvedAttack.toUpperCase()}</strong> was run on{' '}
              <strong>{resolvedArchitecture}</strong> at ε ={' '}
              <strong>{resolvedEpsilon != null ? resolvedEpsilon.toFixed(4) : '—'}</strong>. The retraining flow is event-driven and starts when the accuracy drop crosses <strong>25%</strong>.
            </>
          )}
        </p>
      </div>

      {usingFallback && (
        <div className="insight-box" style={{ marginTop: '20px', border: '1px solid #e67e22', background: 'rgba(230,126,34,0.08)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <AlertTriangle size={18} style={{ color: '#e67e22' }} />
            <strong>Checkpoint Warning</strong>
          </div>
          <p style={{ margin: 0 }}>
            This run used a fallback classifier head because a fine-tuned checkpoint was not found. Metrics from fallback runs can look flat or unrealistic.
          </p>
        </div>
      )}

      <div className="insight-box" style={{ marginTop: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <Activity size={18} />
          <strong>{isDefenseRun ? 'Defense Comparison Table' : 'Comparison Table'}</strong>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <th style={{ padding: '10px 8px' }}>Metric</th>
                <th style={{ padding: '10px 8px' }}>Value</th>
              </tr>
            </thead>
            <tbody>
              {summaryRows.map((row) => (
                <tr key={row.metric} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '10px 8px', opacity: 0.8 }}>{row.metric}</td>
                  <td style={{ padding: '10px 8px', fontWeight: 600 }}>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {(results.comparison_image || results.visualization) && (
        <div className="visualization-container">
          <img src={results.comparison_image || results.visualization} alt="Attack Comparison" className="visualization" />
        </div>
      )}

      {isDefenseRun ? (
        <div style={{
          padding: '16px',
          marginBottom: '20px',
          borderRadius: '8px',
          backgroundColor: 'rgba(102, 126, 234, 0.10)',
          border: '1px solid rgba(102, 126, 234, 0.35)',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <ShieldCheck size={20} style={{ color: '#667eea', marginTop: '2px', flexShrink: 0 }} />
          <div>
            <strong style={{ display: 'block', marginBottom: '4px' }}>
              {standardAdversarialAccuracy != null && robustAdversarialAccuracy != null
                ? `Baseline: ${(standardAdversarialAccuracy * 100).toFixed(1)}% → Robust: ${(robustAdversarialAccuracy * 100).toFixed(1)}%`
                : 'Defense comparison results'}
            </strong>
            <p style={{ margin: 0, fontSize: '0.9em', opacity: 0.85 }}>
              {defenseGain != null && defenseGain >= 0 ? (
                <>
                  The robust checkpoint holds up better under attack by <strong>{(defenseGain * 100).toFixed(1)} percentage points</strong>.
                </>
              ) : (
                'This report is meant to show how the robust checkpoint compares with the baseline model under the same attack.'
              )}
            </p>
          </div>
        </div>
      ) : isTransferRun ? (
        <div style={{
          padding: '16px',
          marginBottom: '20px',
          borderRadius: '8px',
          backgroundColor: 'rgba(255, 193, 7, 0.10)',
          border: '1px solid rgba(255, 193, 7, 0.35)',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <Activity size={20} style={{ color: '#f1c40f', marginTop: '2px', flexShrink: 0 }} />
          <div>
            <strong style={{ display: 'block', marginBottom: '4px' }}>
              Transfer success: {(Number(results.adversarial_accuracy || 0) * 100).toFixed(1)}%
            </strong>
            <p style={{ margin: 0, fontSize: '0.9em', opacity: 0.85 }}>
              This is the average success rate when the attack is applied to other models; the table below shows the per-model transfer pattern.
            </p>
          </div>
        </div>
      ) : results.accuracy_drop != null && (
        <div style={{
          padding: '16px',
          marginBottom: '20px',
          borderRadius: '8px',
          backgroundColor: accuracyDrop > 0.25 ? 'rgba(230, 126, 34, 0.1)' : 'rgba(46, 204, 113, 0.1)',
          border: `1px solid ${accuracyDrop > 0.25 ? '#e67e22' : '#2ecc71'}`,
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          {accuracyDrop > 0.25 ? (
            <AlertTriangle size={20} style={{ color: '#e67e22', marginTop: '2px', flexShrink: 0 }} />
          ) : (
            <ZapIcon size={20} style={{ color: '#2ecc71', marginTop: '2px', flexShrink: 0 }} />
          )}
          <div>
            <strong style={{ display: 'block', marginBottom: '4px' }}>
              Accuracy Drop: {(accuracyDrop * 100).toFixed(1)}%
            </strong>
            <p style={{ margin: 0, fontSize: '0.9em', opacity: 0.8 }}>
              {results.retrain_triggered ? (
                <>
                  <span style={{ color: '#e67e22' }}>Adversarial retraining initiated</span>
                  <br />
                  The system will retrain because the drop crossed the configured threshold.
                </>
              ) : (
                'Model is performing well against this attack.'
              )}
            </p>
          </div>
        </div>
      )}
      

      {latestRun && (
        <div className="insight-box" style={{ marginTop: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <BarChart3 size={18} />
            <strong>Retraining Story</strong>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '12px' }}>
            <div><small>Before Adv</small><div style={{ fontWeight: 700 }}>{(Number(latestRun.before_acc || 0) * 100).toFixed(1)}%</div></div>
            <div><small>After Adv</small><div style={{ fontWeight: 700, color: '#2ecc71' }}>{(Number(latestRun.after_acc || 0) * 100).toFixed(1)}%</div></div>
            <div><small>Adv Improvement</small><div style={{ fontWeight: 700, color: Number(latestRun.improvement || 0) >= 0 ? '#2ecc71' : '#e74c3c' }}>{(Number(latestRun.improvement || 0) * 100).toFixed(1)}%</div></div>
            <div><small>Status</small><div style={{ fontWeight: 700 }}>{latestRun.in_progress ? 'Running' : 'Ready'}</div></div>
          </div>
        </div>
      )}

      {results.improvement_text && (
        <div className="insight-box">
          <strong>{isDefenseRun ? 'Defense Summary:' : isTransferRun ? 'Transfer Summary:' : 'Key Finding:'}</strong>
          <p>{results.improvement_text}</p>
        </div>
      )}

      {results.insight && (
        <div className="insight-box">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <BarChart3 size={18} />
            <strong>Transfer Insight:</strong>
          </div>
          <p>{results.insight}</p>
          {results.statistics && (
            <>
              <p style={{ marginTop: '10px' }}>
                <strong>Diagonal Average (Self-Attack):</strong> {results.statistics.diagonal_average?.toFixed(1)}%
              </p>
              <p>
                <strong>Off-Diagonal Average (Cross-Arch):</strong> {results.statistics.off_diagonal_average?.toFixed(1)}%
              </p>
              <p>
                <strong style={{ color: '#667eea' }}>Defense Gap:</strong> {results.statistics.defense_gap?.toFixed(2)}pp
              </p>
            </>
          )}
        </div>
      )}

      {transferRows.length > 0 && (
        <div className="insight-box" style={{ marginTop: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Activity size={18} />
            <strong>Transferability Comparison</strong>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                  <th style={{ padding: '10px 8px' }}>Model</th>
                  <th style={{ padding: '10px 8px' }}>Success Rate</th>
                </tr>
              </thead>
              <tbody>
                {transferRows.map((row) => (
                  <tr key={row.architecture} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '10px 8px', textTransform: 'capitalize' }}>{row.architecture}</td>
                    <td style={{ padding: '10px 8px', fontWeight: 600 }}>{row.successRate}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {results.samples && (
        <div className="insight-box" style={{ marginTop: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Database size={18} />
            <strong>Sample Results:</strong>
          </div>
          <p style={{ marginTop: '10px', fontSize: '0.9em' }}>
            Clean predictions: {results.samples.clean_predictions.slice(0, 5).join(', ')}...
            <br />
            Adversarial predictions: {results.samples.adv_predictions.slice(0, 5).join(', ')}...
          </p>
        </div>
      )}

      {results.execution_time && (
        <div className="insight-box" style={{ marginTop: '20px', backgroundColor: 'rgba(46, 204, 113, 0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Clock size={18} />
            <strong>Execution Time:</strong>
          </div>
          <p style={{ marginTop: '8px' }}>{results.execution_time.toFixed(2)} seconds</p>
        </div>
      )}
    </div>
  );
}
