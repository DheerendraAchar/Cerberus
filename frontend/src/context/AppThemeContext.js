import React, { createContext, useState, useContext } from 'react';

const AppThemeContext = createContext();

export const useAppTheme = () => {
  const context = useContext(AppThemeContext);
  if (!context) {
    throw new Error('useAppTheme must be used within AppThemeProvider');
  }
  return context;
};

export const AppThemeProvider = ({ children }) => {
  const [themeMode, setThemeMode] = useState('dark');

  const toggleTheme = () => {
    setThemeMode(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const value = {
    themeMode,
    setThemeMode,
    toggleTheme,
    isDark: themeMode === 'dark'
  };

  return (
    <AppThemeContext.Provider value={value}>
      {children}
    </AppThemeContext.Provider>
  );
};
