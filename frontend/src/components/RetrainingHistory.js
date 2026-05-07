import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { History, RefreshCw, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';

export default function RetrainingHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchHistory = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await axios.get('/api/retraining-history');
      setHistory(response.data?.runs || []);
    } catch (err) {
      setError('Unable to load retraining history.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
    const interval = setInterval(fetchHistory, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="insight-box" style={{ marginTop: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <History size={18} />
          <strong>Retraining History</strong>
        </div>
        <button
          type="button"
          onClick={fetchHistory}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            border: 'none',
            borderRadius: '999px',
            padding: '8px 12px',
            cursor: 'pointer',
            background: 'rgba(102, 126, 234, 0.14)',
            color: 'inherit'
          }}
        >
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
          Refresh
        </button>
      </div>

      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: '#e67e22' }}>
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {!loading && history.length === 0 && !error && (
        <p style={{ margin: 0, opacity: 0.75 }}>No retraining runs recorded yet. Trigger an attack with a large accuracy drop to populate this view.</p>
      )}

      <div style={{ display: 'grid', gap: '12px' }}>
        {history.map((run) => (
          <div
            key={run.id}
            style={{
              padding: '14px',
              borderRadius: '12px',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.08)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <strong style={{ textTransform: 'capitalize' }}>{run.architecture}</strong>
                {Number(run.demo_mode || 0) === 1 && (
                  <span
                    style={{
                      fontSize: '0.68rem',
                      fontWeight: 800,
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      padding: '3px 8px',
                      borderRadius: '999px',
                      background: 'rgba(255, 167, 38, 0.16)',
                      border: '1px solid rgba(255, 167, 38, 0.35)',
                      color: '#ffa726'
                    }}
                  >
                    Demo
                  </span>
                )}
              </div>
              <span style={{ opacity: 0.75 }}>{new Date(run.timestamp).toLocaleString()}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '10px' }}>
              <div>
                <small>Status</small>
                <div style={{ fontWeight: 700, color: run.status === 'stopped' ? '#f39c12' : run.status === 'completed' ? '#2ecc71' : undefined }}>
                  {run.status === 'completed' ? 'Completed' : run.status === 'stopped' ? 'Stopped' : 'Failed'}
                </div>
              </div>
              <div><small>Before Adv</small><div style={{ fontWeight: 700 }}>{(Number(run.before_acc ?? run.before_accuracy ?? 0) * 100).toFixed(1)}%</div></div>
              <div><small>After Adv</small><div style={{ fontWeight: 700, color: run.status === 'completed' ? '#2ecc71' : run.status === 'stopped' ? '#f39c12' : '#e74c3c' }}>{(Number(run.after_acc ?? run.after_accuracy ?? 0) * 100).toFixed(1)}%</div></div>
              <div><small>Adv Improvement</small><div style={{ fontWeight: 700 }}>{(Number(run.improvement ?? ((run.after_acc ?? run.after_accuracy ?? 0) - (run.before_acc ?? run.before_accuracy ?? 0))) * 100).toFixed(1)}%</div></div>
            </div>
            {run.message && (
              <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '8px', opacity: 0.8 }}>
                <CheckCircle2 size={16} />
                <span>{run.message}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
