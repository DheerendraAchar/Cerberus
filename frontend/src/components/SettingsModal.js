import React, { useContext, useState, useEffect } from 'react';
import { X, Moon, Sun, Bell, Save } from 'lucide-react';
import { ThemeContext } from '../context/ThemeContext';

export default function SettingsModal({ isOpen, onClose }) {
  const { isDark, toggleTheme } = useContext(ThemeContext);
  const [notifications, setNotifications] = useState(true);

  useEffect(() => {
    // Load notification preference from localStorage
    const saved = localStorage.getItem('enableNotifications');
    if (saved !== null) {
      setNotifications(JSON.parse(saved));
    }
  }, [isOpen]);

  const handleNotificationToggle = () => {
    const newValue = !notifications;
    setNotifications(newValue);
    localStorage.setItem('enableNotifications', JSON.stringify(newValue));
  };

  if (!isOpen) return null;

  return (
    <div className="help-modal-overlay" onClick={onClose}>
      <div className="help-modal" onClick={(e) => e.stopPropagation()}>
        <div className="help-modal-header">
          <div className="help-modal-title">
            <Save size={20} />
            Settings
          </div>
          <button
            className="help-modal-close"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>

        <div className="help-modal-content">
          {/* Theme Settings */}
          <div className="settings-section">
            <div className="settings-section-title">Appearance</div>
            
            <div className="settings-option">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {isDark ? <Moon size={16} /> : <Sun size={16} />}
                <span>Dark Mode</span>
              </div>
              <button
                className={`toggle-switch ${isDark ? 'active' : ''}`}
                onClick={toggleTheme}
              >
                <div className="toggle-knob"></div>
              </button>
            </div>
          </div>

          {/* Notification Settings */}
          <div className="settings-section">
            <div className="settings-section-title">Notifications</div>
            
            <div className="settings-option">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Bell size={16} />
                <span>Experiment Results</span>
              </div>
              <button
                className={`toggle-switch ${notifications ? 'active' : ''}`}
                onClick={handleNotificationToggle}
              >
                <div className="toggle-knob"></div>
              </button>
            </div>
          </div>

          {/* System Info */}
          <div className="settings-section">
            <div className="settings-section-title">System Information</div>
            
            <div className="system-info">
              <div className="info-row">
                <span className="info-label">App Version:</span>
                <span className="info-value">1.0.0</span>
              </div>
              <div className="info-row">
                <span className="info-label">Platform:</span>
                <span className="info-value">Cerberus Framework</span>
              </div>
              <div className="info-row">
                <span className="info-label">Status:</span>
                <span className="info-value" style={{ color: '#a8ff78' }}>Ready</span>
              </div>
            </div>
          </div>
        </div>

        <div style={{
          padding: '16px 20px',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '8px'
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              backgroundColor: 'transparent',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              cursor: 'pointer',
              color: 'var(--text)',
              fontSize: '13px',
              fontWeight: '500',
              transition: 'all 0.2s',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
