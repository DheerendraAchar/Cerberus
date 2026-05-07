import React, { useState, useContext } from 'react';
import { Moon, Sun, Settings, HelpCircle } from 'lucide-react';
import SettingsModal from './SettingsModal';
import { ThemeContext } from '../context/ThemeContext';

export default function TopBar({ onHelpClick, stats }) {
  const { isDark, toggleTheme } = useContext(ThemeContext);
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '12px 20px',
      backgroundColor: '#0a0c10',
      borderBottom: '1px solid #1e2430',
      gap: '20px'
    }}>
      {/* Left - Stats */}
      <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
        {stats && (
          <>
            <div style={{
              display: 'flex',
              gap: '8px',
              alignItems: 'center',
              padding: '8px 12px',
              backgroundColor: '#181c24',
              borderRadius: '6px',
              fontSize: '13px',
              color: '#64748b'
            }}>
              <span>Experiments:</span>
              <span style={{ color: '#00e5ff', fontWeight: '600' }}>{stats.total}</span>
            </div>
            <div style={{
              display: 'flex',
              gap: '8px',
              alignItems: 'center',
              padding: '8px 12px',
              backgroundColor: '#181c24',
              borderRadius: '6px',
              fontSize: '13px',
              color: '#64748b'
            }}>
              <span>Avg Success:</span>
              <span style={{ color: '#a8ff78', fontWeight: '600' }}>{stats.avgSuccess?.toFixed(1)}%</span>
            </div>
          </>
        )}
      </div>

      {/* Right - Controls */}
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <button
          onClick={() => setShowSettings(true)}
          style={{
            padding: '8px 12px',
            backgroundColor: 'transparent',
            border: '1px solid #1e2430',
            borderRadius: '6px',
            cursor: 'pointer',
            color: '#64748b',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '13px',
            transition: 'all 0.2s',
            fontWeight: '500'
          }}
          title="Settings"
        >
          <Settings size={16} />
        </button>

        <button
          onClick={onHelpClick}
          style={{
            padding: '8px 12px',
            backgroundColor: 'transparent',
            border: '1px solid #1e2430',
            borderRadius: '6px',
            cursor: 'pointer',
            color: '#64748b',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '13px',
            transition: 'all 0.2s',
            fontWeight: '500'
          }}
          title="Help (Press ?)"
        >
          <HelpCircle size={16} />
        </button>

        <button
          onClick={toggleTheme}
          style={{
            padding: '8px 12px',
            backgroundColor: isDark ? '#1e2430' : '#ddd',
            border: '1px solid #1e2430',
            borderRadius: '6px',
            cursor: 'pointer',
            color: isDark ? '#00e5ff' : '#333',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '13px',
            transition: 'all 0.2s',
            fontWeight: '500'
          }}
          title={isDark ? 'Light Mode' : 'Dark Mode'}
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>

      <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </div>
  );
}
