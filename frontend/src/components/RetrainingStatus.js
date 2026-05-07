import React, { useState, useEffect, useRef } from 'react';
import { Cpu, TrendingUp, CheckCircle } from 'lucide-react';
import axios from 'axios';

export default function RetrainingStatus() {
  const [status, setStatus] = useState(null);
  const [visible, setVisible] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [stopLoading, setStopLoading] = useState(false);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const cardRef = useRef(null);
  const prevInProgressRef = useRef(false);
  const dismissedCompletionKeyRef = useRef(null);

  const buildCompletionKey = (data) => {
    if (!data) return '';
    return [
      data.arch || '',
      data.before_acc ?? '',
      data.after_acc ?? '',
      data.improvement ?? '',
      data.message || ''
    ].join('|');
  };

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await axios.get('/api/retraining-status');
        const next = response.data;
        setStatus(next);

        if (next.in_progress) {
          // Always show while retraining is active.
          setVisible(true);
          prevInProgressRef.current = true;
          return;
        }

        // Show completion exactly once when transitioning from in-progress -> not in-progress.
        if (prevInProgressRef.current) {
          const completionKey = buildCompletionKey(next);
          if (dismissedCompletionKeyRef.current !== completionKey) {
            setVisible(true);
          }
        }
        prevInProgressRef.current = false;
      } catch (err) {
        console.error('Error fetching retraining status:', err);
      }
    };

    // Check immediately and every 2 seconds while in progress
    checkStatus();
    const interval = setInterval(checkStatus, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleStopRetraining = async () => {
    if (!status?.in_progress || stopLoading) return;

    try {
      setStopLoading(true);
      const response = await axios.post('/api/retraining-stop');
      setStatus((prev) => ({
        ...(prev || {}),
        stop_requested: true,
        message: response?.data?.message || 'Stopping retraining...'
      }));
    } catch (err) {
      console.error('Error stopping retraining:', err);
      setStatus((prev) => ({
        ...(prev || {}),
        message: 'Failed to stop retraining. Please try again.'
      }));
    } finally {
      setStopLoading(false);
    }
  };

  const handleMouseDown = (e) => {
    // Only drag from header
    if (!e.target.closest('.retraining-header') || e.target.closest('button')) {
      return;
    }
    setIsDragging(true);
    const rect = cardRef.current.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e) => {
      setPosition({
        x: e.clientX - dragOffset.x,
        y: e.clientY - dragOffset.y
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  if (!status || !visible) return null;

  return (
    <div 
      className="retraining-status-container"
      style={{
        left: position.x !== 0 ? position.x : 'auto',
        right: position.x === 0 ? '20px' : 'auto',
        top: position.y !== 0 ? position.y : 'auto',
        bottom: position.y === 0 ? '20px' : 'auto',
      }}
    >
      <div 
        className="retraining-card"
        ref={cardRef}
        onMouseDown={handleMouseDown}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        <div className="retraining-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {status.in_progress ? (
              <>
                <div className="spinning-loader">
                  <Cpu size={20} style={{ color: '#667eea' }} />
                </div>
                <div>
                  <h3 style={{ margin: 0, marginBottom: '4px' }}>Adversarial Retraining</h3>
                  <p style={{ margin: 0, fontSize: '0.9em', opacity: 0.7 }}>{status.message}</p>
                </div>
              </>
            ) : status.improvement > 0 ? (
              <>
                <CheckCircle size={20} style={{ color: '#2ecc71' }} />
                <div>
                  <h3 style={{ margin: 0, marginBottom: '4px' }}>Retraining Complete</h3>
                  <p style={{ margin: 0, fontSize: '0.9em', opacity: 0.7 }}>{status.message}</p>
                </div>
              </>
            ) : null}
          </div>
          <button
            onClick={() => {
              if (!status?.in_progress) {
                dismissedCompletionKeyRef.current = buildCompletionKey(status);
              }
              setVisible(false);
            }}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              display: 'flex',
              alignItems: 'center'
            }}
          >
            ✕
          </button>
        </div>

        {status.in_progress && (
          <div className="progress-bar-container">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${status.progress}%` }}
              ></div>
            </div>
            <span style={{ fontSize: '0.85em', marginTop: '8px', display: 'block' }}>
              {status.progress}% Complete
            </span>
            <button
              type="button"
              onClick={() => setShowStopConfirm(true)}
              disabled={stopLoading || status.stop_requested}
              className="stop-retraining-btn"
            >
              {status.stop_requested ? 'Stopping…' : stopLoading ? 'Stopping…' : 'Stop Retraining'}
            </button>
          </div>
        )}

        {status.before_acc > 0 && (
          <div className="improvement-metrics">
            <div className="metric-comparison">
              <div className="metric-item">
                <span className="metric-label">Before (Adv)</span>
                <span className="metric-value">{(status.before_acc * 100).toFixed(1)}%</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <TrendingUp size={16} style={{ color: '#2ecc71' }} />
              </div>
              <div className="metric-item">
                <span className="metric-label">After (Adv)</span>
                <span className="metric-value" style={{ color: '#2ecc71' }}>
                  {(status.after_acc * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            {status.improvement !== 0 && (
              <div className="improvement-text">
                {status.improvement > 0 ? (
                  <>
                    <span style={{ color: '#2ecc71' }}>↑ +{(status.improvement * 100).toFixed(1)}%</span>
                    <span style={{ fontSize: '0.9em', opacity: 0.7 }}>adv improvement</span>
                  </>
                ) : (
                  <>
                    <span style={{ color: '#e74c3c' }}>↓ {(status.improvement * 100).toFixed(1)}%</span>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {showStopConfirm && (
        <div className="stop-modal-backdrop" onClick={() => setShowStopConfirm(false)}>
          <div className="stop-modal" onClick={(e) => e.stopPropagation()}>
            <h4 style={{ margin: 0, marginBottom: '8px' }}>Stop Retraining?</h4>
            <p style={{ margin: 0, marginBottom: '16px', opacity: 0.85, lineHeight: 1.5 }}>
              This will halt the current retraining run and mark it as stopped in history.
            </p>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="stop-modal-cancel"
                onClick={() => setShowStopConfirm(false)}
                disabled={stopLoading}
              >
                Cancel
              </button>
              <button
                type="button"
                className="stop-modal-confirm"
                onClick={async () => {
                  await handleStopRetraining();
                  setShowStopConfirm(false);
                }}
                disabled={stopLoading}
              >
                {stopLoading ? 'Stopping…' : 'Yes, Stop'}
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .retraining-status-container {
          position: fixed;
          z-index: 1000;
          max-width: 400px;
          user-select: none;
        }

        .retraining-card {
          background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
          border: 2px solid #e0e6f2;
          border-radius: 12px;
          padding: 20px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.08);
          animation: slideInUp 0.3s ease-out;
          color: #2c3e50;
        }

        @media (prefers-color-scheme: dark) {
          .retraining-card {
            background: var(--bg-secondary, #2a2a3e);
            border-color: var(--border-color, #444);
            color: #e0e6f2;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
          }
        }

        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .retraining-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 16px;
          cursor: grab;
        }

        .retraining-header:active {
          cursor: grabbing;
        }

        .spinning-loader {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .progress-bar-container {
          margin: 16px 0;
        }

        .stop-retraining-btn {
          margin-top: 12px;
          width: 100%;
          border: 1px solid rgba(231, 76, 60, 0.45);
          background: rgba(231, 76, 60, 0.12);
          color: #e74c3c;
          border-radius: 8px;
          padding: 8px 12px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .stop-retraining-btn:hover:enabled {
          background: rgba(231, 76, 60, 0.18);
          border-color: rgba(231, 76, 60, 0.7);
        }

        .stop-retraining-btn:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .stop-modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(10, 12, 16, 0.55);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1100;
          animation: fadeIn 0.16s ease;
        }

        .stop-modal {
          width: min(380px, calc(100vw - 32px));
          background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
          color: #2c3e50;
          border: 1px solid #e0e6f2;
          border-radius: 12px;
          padding: 16px;
          box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
        }

        .stop-modal-cancel,
        .stop-modal-confirm {
          border: none;
          border-radius: 8px;
          padding: 8px 12px;
          font-weight: 700;
          cursor: pointer;
        }

        .stop-modal-cancel {
          background: rgba(0, 0, 0, 0.06);
          color: #2c3e50;
        }

        .stop-modal-confirm {
          background: rgba(231, 76, 60, 0.14);
          color: #e74c3c;
          border: 1px solid rgba(231, 76, 60, 0.35);
        }

        .stop-modal-cancel:disabled,
        .stop-modal-confirm:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        @media (prefers-color-scheme: dark) {
          .stop-modal {
            background: #1f2430;
            color: #e0e6f2;
            border-color: #3a4257;
          }

          .stop-modal-cancel {
            background: rgba(255, 255, 255, 0.08);
            color: #e0e6f2;
          }
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: #e0e6f2;
          border-radius: 4px;
          overflow: hidden;
        }

        @media (prefers-color-scheme: dark) {
          .progress-bar {
            background: rgba(102, 126, 234, 0.1);
          }
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #667eea, #764ba2);
          transition: width 0.3s ease;
        }

        .improvement-metrics {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid #e0e6f2;
        }

        @media (prefers-color-scheme: dark) {
          .improvement-metrics {
            border-top-color: #444;
          }
        }

        .metric-comparison {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;
        }

        .metric-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          flex: 1;
        }

        .metric-label {
          font-size: 0.85em;
          opacity: 0.7;
          margin-bottom: 4px;
          font-weight: 500;
        }

        .metric-value {
          font-size: 1.4em;
          font-weight: bold;
          color: #667eea;
        }

        .improvement-text {
          display: flex;
          align-items: center;
          gap: 8px;
          justify-content: center;
          padding: 12px;
          background: #f0f3ff;
          border-radius: 8px;
          font-weight: 600;
          color: #667eea;
        }

        @media (prefers-color-scheme: dark) {
          .improvement-text {
            background: rgba(102, 126, 234, 0.1);
            color: #8fa3ff;
          }
        }
      `}</style>
    </div>
  );
}
