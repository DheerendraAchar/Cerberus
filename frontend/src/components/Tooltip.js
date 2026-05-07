import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

export default function Tooltip({ children, text, position = 'top' }) {
  const [isVisible, setIsVisible] = useState(false);

  const positionStyles = {
    top: { bottom: '100%', left: '50%', transform: 'translateX(-50%)', marginBottom: '8px' },
    bottom: { top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: '8px' },
    right: { left: '100%', top: '50%', transform: 'translateY(-50%)', marginLeft: '8px' },
    left: { right: '100%', top: '50%', transform: 'translateY(-50%)', marginRight: '8px' }
  };

  return (
    <div
      style={{ position: 'relative', display: 'inline-block', cursor: 'help' }}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      <HelpCircle
        size={14}
        style={{
          display: 'inline-block',
          marginLeft: '4px',
          color: '#667eea',
          opacity: 0.7,
          cursor: 'help'
        }}
      />

      {isVisible && (
        <div
          style={{
            position: 'absolute',
            ...positionStyles[position],
            backgroundColor: '#1e2430',
            color: '#e2e8f0',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            whiteSpace: 'nowrap',
            zIndex: 1000,
            border: '1px solid #667eea',
            boxShadow: '0 4px 12px rgba(102, 126, 234, 0.2)'
          }}
        >
          {text}
        </div>
      )}
    </div>
  );
}
