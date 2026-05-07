import React, { useEffect, useState } from 'react';

export default function SplashScreen({ isVisible, onComplete }) {
  const [isHiding, setIsHiding] = useState(false);

  useEffect(() => {
    if (!isVisible) {
      return;
    }

    // After 3.5 seconds (text + logo animations done), start fading out
    const timer = setTimeout(() => {
      setIsHiding(true);
      // Call onComplete after fade-out completes
      setTimeout(onComplete, 600);
    }, 3500);

    return () => clearTimeout(timer);
  }, [isVisible, onComplete]);

  if (!isVisible && isHiding) {
    return null;
  }

  return (
    <div className={`splash-screen ${isHiding ? 'hide' : ''}`}>
      <div className="splash-content">
        <div className="animated-text">
          <span className="letter">C</span>
          <span className="letter">E</span>
          <span className="letter">R</span>
          <span className="letter">B</span>
          <span className="letter">E</span>
          <span className="letter">R</span>
          <span className="letter">U</span>
          <span className="letter">S</span>
        </div>
        <div className="splash-version-badge">v2</div>
        <div className="logo-container">
          <img 
            src="/cerberuslogo.png" 
            alt="Cerberus Logo" 
            className="splash-logo"
          />
        </div>
      </div>
    </div>
  );
}
