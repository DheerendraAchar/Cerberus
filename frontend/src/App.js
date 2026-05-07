import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  AppBar,
  Toolbar,
  Button,
  Paper,
  Container,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CssBaseline,
  ThemeProvider,
  createTheme,
  Divider,
  Tooltip,
  Collapse
} from '@mui/material';
import {
  Security,
  Bolt,
  CompareArrows,
  BarChart,
  Build,
  DatasetLinked,
  DarkMode,
  LightMode,
  Speed,
  Storage,
  KeyboardArrowDown,
  AutoAwesome
} from '@mui/icons-material';
import AttackPanel from './components/AttackPanel';
import DefensePanel from './components/DefensePanel';
import TransferPanel from './components/TransferPanel';
import ResultsPanel from './components/ResultsPanel';
import PipelineDiagram from './components/PipelineDiagram';
import ExperimentsPanel from './components/ExperimentsPanel';
import HelpModal from './components/HelpModal';
import SplashScreen from './components/SplashScreen';
import RetrainingStatus from './components/RetrainingStatus';
import RetrainingHistory from './components/RetrainingHistory';
import { ThemeProvider as CustomThemeProvider } from './context/ThemeContext';
import { AppThemeProvider, useAppTheme } from './context/AppThemeContext';
import './App.css';

// Theme Factory Function
const createAppTheme = (mode) => {
  const isDark = mode === 'dark';
  
  return createTheme({
    palette: {
      mode: mode,
      ...(isDark ? {
        primary: {
          main: '#00d9ff',
          light: '#4df7ff',
          dark: '#0099cc',
        },
        secondary: {
          main: '#ff4081',
          light: '#ff6bb6',
          dark: '#c60055',
        },
        background: {
          default: '#0a0c10',
          paper: '#111318',
        },
        surface: {
          main: '#181c24',
        },
        success: {
          main: '#a8ff78',
        },
        warning: {
          main: '#ffd166',
        },
        error: {
          main: '#ff4d6d',
        },
        text: {
          primary: '#e2e8f0',
          secondary: '#64748b',
        },
      } : {
        primary: {
          main: '#0099cc',
          light: '#4df7ff',
          dark: '#006699',
        },
        secondary: {
          main: '#d81b60',
          light: '#ff6bb6',
          dark: '#a60039',
        },
        background: {
          default: '#f5f7fa',
          paper: '#ffffff',
        },
        surface: {
          main: '#e8eef7',
        },
        success: {
          main: '#66bb6a',
        },
        warning: {
          main: '#ffa726',
        },
        error: {
          main: '#ef5350',
        },
        text: {
          primary: '#1a237e',
          secondary: '#616161',
        },
      }),
    },
    typography: {
      fontFamily: "'Syne', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      h1: {
        fontSize: '2.5rem',
        fontWeight: 800,
        letterSpacing: '-0.02em',
      },
      h2: {
        fontSize: '2rem',
        fontWeight: 700,
      },
      h3: {
        fontSize: '1.5rem',
        fontWeight: 700,
      },
      button: {
        textTransform: 'uppercase',
        fontWeight: 600,
        letterSpacing: '0.05em',
      },
    },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            background: isDark ? 'linear-gradient(135deg, #111318 0%, #181c24 100%)' : 'linear-gradient(135deg, #f5f7fa 0%, #e8eef7 100%)',
            borderBottom: isDark ? '2px solid #00d9ff' : '2px solid #0099cc',
            backdropFilter: 'blur(10px)',
            boxShadow: isDark ? '0 4px 20px rgba(0, 217, 255, 0.1)' : '0 4px 20px rgba(0, 153, 204, 0.15)',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: '8px',
            padding: '10px 24px',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: isDark ? '0 8px 24px rgba(0, 217, 255, 0.2)' : '0 8px 24px rgba(0, 153, 204, 0.2)',
            },
          },
          contained: {
            background: isDark ? 'linear-gradient(135deg, #00d9ff 0%, #0099cc 100%)' : 'linear-gradient(135deg, #0099cc 0%, #006699 100%)',
            color: isDark ? '#0a0c10' : '#ffffff',
            fontWeight: 700,
            '&:hover': {
              background: isDark ? 'linear-gradient(135deg, #4df7ff 0%, #00d9ff 100%)' : 'linear-gradient(135deg, #00d9ff 0%, #0099cc 100%)',
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            background: isDark ? '#111318' : '#ffffff',
            borderRadius: '12px',
            border: isDark ? '1px solid #1e2430' : '1px solid #e0e0e0',
            backdropFilter: 'blur(10px)',
            transition: 'all 0.3s ease',
            '&:hover': {
              borderColor: isDark ? 'rgba(0, 217, 255, 0.4)' : 'rgba(0, 153, 204, 0.4)',
              boxShadow: isDark ? '0 0 30px rgba(0, 217, 255, 0.1)' : '0 0 30px rgba(0, 153, 204, 0.15)',
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontWeight: 600,
            backgroundColor: isDark ? 'rgba(0, 217, 255, 0.1)' : 'rgba(0, 153, 204, 0.1)',
            border: isDark ? '1px solid rgba(0, 217, 255, 0.3)' : '1px solid rgba(0, 153, 204, 0.3)',
            color: isDark ? '#00d9ff' : '#0099cc',
          },
        },
      },
      MuiToggleButton: {
        styleOverrides: {
          root: {
            color: isDark ? '#64748b' : '#616161',
            borderColor: isDark ? '#1e2430' : '#e0e0e0',
            transition: 'all 0.3s ease',
            '&.Mui-selected': {
              color: isDark ? '#00d9ff' : '#0099cc',
              backgroundColor: isDark ? 'rgba(0, 217, 255, 0.15)' : 'rgba(0, 153, 204, 0.15)',
              borderColor: isDark ? '#00d9ff' : '#0099cc',
              '&:hover': {
                backgroundColor: isDark ? 'rgba(0, 217, 255, 0.25)' : 'rgba(0, 153, 204, 0.25)',
              },
            },
          },
        },
      },
    },
  });
};


function AppContent() {
  const { themeMode, setThemeMode } = useAppTheme();
  const [activeTab, setActiveTab] = useState('attack');
  const [dataset, setDataset] = useState('cifar10');
  const [showSplash, setShowSplash] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showHelp, setShowHelp] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [stats, setStats] = useState({ total: 0, avgSuccess: 0 });
  const [retrainingStatus, setRetrainingStatus] = useState(null);
  const [retrainingDemoMode, setRetrainingDemoMode] = useState(false);
  const [updatingDemoMode, setUpdatingDemoMode] = useState(false);
  const [pipelineExpanded, setPipelineExpanded] = useState(false);
  const [pipelineAttention, setPipelineAttention] = useState(true);

  // Apply theme to document body
  useEffect(() => {
    if (themeMode === 'light') {
      document.body.setAttribute('data-theme', 'light');
      document.body.classList.add('light-mode');
    } else {
      document.body.removeAttribute('data-theme');
      document.body.classList.remove('light-mode');
    }
  }, [themeMode]);

  useEffect(() => {
    const initApp = async () => {
      await fetchSystemStatus();
      await fetchExperiments();
      await fetchRetrainingStatus();
    };
    initApp();
    
    const refreshInterval = setInterval(() => {
      fetchExperiments();
      fetchRetrainingStatus();
    }, 10000);
    
    const handleKeyPress = (e) => {
      if (e.key === '1') setActiveTab('attack');
      if (e.key === '2') setActiveTab('defense');
      if (e.key === '3') setActiveTab('transfer');
      if (e.key === '4') setActiveTab('experiments');
      if (e.key === '?' || (e.shiftKey && e.key === '/')) {
        e.preventDefault();
        setShowHelp(!showHelp);
      }
      if (e.key === 'Escape') setShowHelp(false);
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
      clearInterval(refreshInterval);
    };
  }, [showHelp]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setPipelineAttention(false);
    }, 7000);

    return () => clearTimeout(timer);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      setSystemStatus(response.data);
      if (response?.data?.retraining_demo_mode != null) {
        setRetrainingDemoMode(Boolean(response.data.retraining_demo_mode));
      }
    } catch (err) {
      console.error('Error fetching status:', err);
    }
  };

  const handleToggleDemoMode = async () => {
    if (updatingDemoMode) return;
    const nextMode = !retrainingDemoMode;
    setUpdatingDemoMode(true);
    try {
      const response = await axios.post('/api/retraining-config', { demo_mode: nextMode });
      setRetrainingDemoMode(Boolean(response?.data?.demo_mode));
      await fetchSystemStatus();
    } catch (err) {
      console.error('Error updating demo mode:', err);
      setError(err.response?.data?.error || 'Failed to toggle demo retraining mode');
    } finally {
      setUpdatingDemoMode(false);
    }
  };

  const fetchExperiments = async () => {
    try {
      const response = await axios.get('/api/experiments');
      const exps = response.data?.experiments || [];
      
      if (exps.length > 0) {
        const total = exps.length;
        let successSum = 0;
        exps.forEach(exp => {
          const successRate = exp.adversarial_accuracy ?? exp.clean_accuracy ?? exp.success_rate ?? 0;
          successSum += successRate;
        });
        const avgSuccess = Math.round((successSum / total) * 1000) / 10;
        setStats({ total, avgSuccess });
      } else {
        setStats({ total: 0, avgSuccess: 0 });
      }
    } catch (err) {
      console.error('Error fetching experiments:', err);
      setStats({ total: 0, avgSuccess: 0 });
    }
  };

  const fetchRetrainingStatus = async () => {
    try {
      const response = await axios.get('/api/retraining-status');
      setRetrainingStatus(response.data);
    } catch (err) {
      console.error('Error fetching retraining status:', err);
    }
  };

  const handleAttackRun = async (params) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('/api/run-attack', params);
      setResults({
        ...response.data,
        attack_type: response.data?.attack_type || params.attack,
        architecture: response.data?.architecture || params.architecture,
        epsilon: response.data?.epsilon ?? params.epsilon,
        analysis_mode: response.data?.analysis_mode || 'defense',
      });
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDefenseRun = async (params) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('/api/run-defense', params);
      setResults({
        ...response.data,
        attack_type: response.data?.attack_type || params.attack,
        architecture: response.data?.architecture || params.architecture,
        epsilon: response.data?.epsilon ?? params.epsilon,
      });
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTransferAnalysis = async (params) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('/api/transfer-analysis', params);
      setResults({
        ...response.data,
        attack_type: response.data?.attack_type || params.attack || 'fgsm',
        architecture: response.data?.architecture || params.source_architecture || params.architecture,
        epsilon: response.data?.epsilon ?? params.epsilon,
      });
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  const sidebarItems = [
    {
      id: 'attack',
      label: 'Attack',
      icon: <Bolt sx={{ fontSize: 24 }} />,
      description: 'Run adversarial attacks'
    },
    {
      id: 'defense',
      label: 'Defense',
      icon: <Security sx={{ fontSize: 24 }} />,
      description: 'Defense mechanisms'
    },
    {
      id: 'transfer',
      label: 'Transfer',
      icon: <CompareArrows sx={{ fontSize: 24 }} />,
      description: 'Transfer analysis'
    },
    {
      id: 'experiments',
      label: 'History',
      icon: <BarChart sx={{ fontSize: 24 }} />,
      description: 'Experiment history'
    },
  ];

  const sidebarContent = (
    <Box
      sx={{
        width: 280,
        padding: 3,
        backgroundColor: themeMode === 'dark' ? '#111318' : '#f5f7fa',
        borderRight: themeMode === 'dark' ? '1px solid #1e2430' : '1px solid #e0e0e0',
        height: '100%',
      }}
    >
      <Box sx={{ mb: 3, textAlign: 'center' }}>
        <img
          src="/cerberuslogo.png"
          alt="CERBERUS"
          style={{
            width: 60,
            height: 60,
            borderRadius: 8,
            filter: themeMode === 'dark' ? 'drop-shadow(0 0 12px rgba(0, 217, 255, 0.3))' : 'drop-shadow(0 0 12px rgba(0, 153, 204, 0.3))',
            marginBottom: 16,
          }}
        />
        <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <Box sx={{ color: themeMode === 'dark' ? '#00d9ff' : '#0099cc', fontSize: '1.2rem', fontWeight: 800 }}>
            CERBERUS
          </Box>
          <Box
            sx={{
              fontSize: '0.65rem',
              fontWeight: 800,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              px: 1,
              py: 0.25,
              borderRadius: '999px',
              color: themeMode === 'dark' ? '#0a0c10' : '#ffffff',
              background: themeMode === 'dark'
                ? 'linear-gradient(135deg, #00d9ff 0%, #4df7ff 100%)'
                : 'linear-gradient(135deg, #0099cc 0%, #00d9ff 100%)',
              boxShadow: themeMode === 'dark'
                ? '0 0 14px rgba(0, 217, 255, 0.35)'
                : '0 0 14px rgba(0, 153, 204, 0.3)',
            }}
          >
            v2
          </Box>
        </Box>
        <Box sx={{ color: themeMode === 'dark' ? '#64748b' : '#616161', fontSize: '0.75rem' }}>
          Adversarial ML Framework
        </Box>
      </Box>

      <List sx={{ gap: 1, display: 'flex', flexDirection: 'column' }}>
        {sidebarItems.map((item) => (
          <ListItem
            button
            key={item.id}
            onClick={() => {
              setActiveTab(item.id);
              setDrawerOpen(false);
            }}
            selected={activeTab === item.id}
            sx={{
              borderRadius: '8px',
              marginBottom: 1,
              transition: 'all 0.3s ease',
              backgroundColor: activeTab === item.id ? 'rgba(0, 217, 255, 0.15)' : 'transparent',
              border: activeTab === item.id ? '1px solid rgba(0, 217, 255, 0.4)' : '1px solid transparent',
              '&:hover': {
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
              },
              '&.Mui-selected': {
                backgroundColor: 'rgba(0, 217, 255, 0.15)',
                borderColor: 'rgba(0, 217, 255, 0.4)',
              },
            }}
          >
            <ListItemIcon sx={{ color: activeTab === item.id ? (themeMode === 'dark' ? '#00d9ff' : '#0099cc') : (themeMode === 'dark' ? '#64748b' : '#616161'), minWidth: 40 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              secondary={item.description}
              primaryTypographyProps={{
                sx: { fontWeight: 600, color: activeTab === item.id ? (themeMode === 'dark' ? '#00d9ff' : '#0099cc') : (themeMode === 'dark' ? '#e2e8f0' : '#1a237e') }
              }}
              secondaryTypographyProps={{
                sx: { fontSize: '0.7rem', color: themeMode === 'dark' ? '#64748b' : '#616161' }
              }}
            />
          </ListItem>
        ))}
      </List>

      <Box sx={{ mt: 4, p: 2, backgroundColor: themeMode === 'dark' ? 'rgba(0, 217, 255, 0.05)' : 'rgba(0, 153, 204, 0.05)', borderRadius: '8px', border: themeMode === 'dark' ? '1px solid rgba(0, 217, 255, 0.1)' : '1px solid rgba(0, 153, 204, 0.15)' }}>
        <Box sx={{ fontSize: '0.8rem', color: themeMode === 'dark' ? '#64748b' : '#616161', mb: 1 }}>SYSTEM STATUS</Box>
        {systemStatus && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Chip
              icon={<Build sx={{ fontSize: 16 }} />}
              label={systemStatus.device === 'cuda' ? 'GPU Enabled' : 'CPU Mode'}
              size="small"
              sx={{ justifyContent: 'flex-start' }}
            />
            <Chip
              label={`PyTorch ${systemStatus.torch_version || 'v2.x'}`}
              size="small"
              sx={{ justifyContent: 'flex-start' }}
            />
            <Chip
              label="System Ready"
              color="success"
              size="small"
              sx={{ justifyContent: 'flex-start' }}
            />
          </Box>
        )}
      </Box>
    </Box>
  );

  return (
    <>
      <SplashScreen isVisible={showSplash} onComplete={() => setShowSplash(false)} />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', backgroundColor: themeMode === 'dark' ? '#0a0c10' : '#f5f7fa', position: 'relative', overflow: 'hidden' }}>
        <Box
          sx={{
            position: 'absolute',
            top: -120,
            right: -100,
            width: 360,
            height: 360,
            borderRadius: '50%',
            background: themeMode === 'dark'
              ? 'radial-gradient(circle, rgba(0,217,255,0.22) 0%, rgba(0,217,255,0) 70%)'
              : 'radial-gradient(circle, rgba(0,153,204,0.20) 0%, rgba(0,153,204,0) 70%)',
            pointerEvents: 'none',
            zIndex: 0,
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            bottom: -160,
            left: -140,
            width: 420,
            height: 420,
            borderRadius: '50%',
            background: themeMode === 'dark'
              ? 'radial-gradient(circle, rgba(255,77,109,0.16) 0%, rgba(255,77,109,0) 70%)'
              : 'radial-gradient(circle, rgba(216,27,96,0.12) 0%, rgba(216,27,96,0) 70%)',
            pointerEvents: 'none',
            zIndex: 0,
          }}
        />
        {/* AppBar */}
        <AppBar
          position="static"
          sx={{
            background: themeMode === 'dark' ? 'linear-gradient(135deg, #111318 0%, #181c24 100%)' : 'linear-gradient(135deg, #f5f7fa 0%, #e8eef7 100%)',
            borderBottom: themeMode === 'dark' ? '2px solid #00d9ff' : '2px solid #0099cc',
            backdropFilter: 'blur(14px)',
            boxShadow: themeMode === 'dark' ? '0 10px 30px rgba(0, 217, 255, 0.12)' : '0 10px 30px rgba(0, 153, 204, 0.12)',
            zIndex: 2,
          }}
        >
          <Toolbar sx={{ justifyContent: 'space-between', padding: '16px 24px' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <IconButton
                color="inherit"
                onClick={() => setDrawerOpen(true)}
                sx={{ display: { xs: 'flex', md: 'none' } }}
              >
                <Bolt />
              </IconButton>
              <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1.5 }}>
                <img
                  src="/cerberuslogo.png"
                  alt="CERBERUS"
                  style={{
                    width: 34,
                    height: 34,
                    borderRadius: 8,
                    filter: themeMode === 'dark' ? 'drop-shadow(0 0 10px rgba(0, 217, 255, 0.35))' : 'drop-shadow(0 0 10px rgba(0, 153, 204, 0.3))',
                  }}
                />
                <Box sx={{ display: 'flex', flexDirection: 'column', lineHeight: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ color: themeMode === 'dark' ? '#00d9ff' : '#0099cc', fontSize: '1.35rem', fontWeight: 800, letterSpacing: '0.01em' }}>
                      CERBERUS
                    </Box>
                    <Box
                      sx={{
                        fontSize: '0.6rem',
                        fontWeight: 800,
                        letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        px: 1,
                        py: 0.3,
                        borderRadius: '999px',
                        color: themeMode === 'dark' ? '#0a0c10' : '#ffffff',
                        background: themeMode === 'dark'
                          ? 'linear-gradient(135deg, #00d9ff 0%, #4df7ff 100%)'
                          : 'linear-gradient(135deg, #0099cc 0%, #00d9ff 100%)',
                      }}
                    >
                      v2
                    </Box>
                  </Box>
                  <Box sx={{ color: themeMode === 'dark' ? '#7a8aa1' : '#4f5f7a', fontSize: '0.72rem', mt: 0.35 }}>
                    Adversarial AI Command Center
                  </Box>
                </Box>
              </Box>
            </Box>

            {/* Dataset Selector - Professional Toggle */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {/* GPU/CPU Status */}
              {systemStatus && (
                <Tooltip title={systemStatus.device === 'cuda' ? 'GPU Acceleration Enabled' : 'Running on CPU'}>
                  <Chip
                    icon={systemStatus.device === 'cuda' ? <Speed /> : <Storage />}
                    label={systemStatus.device === 'cuda' ? 'GPU' : 'CPU'}
                    size="small"
                    sx={{
                      backgroundColor: systemStatus.device === 'cuda' ? 'rgba(168, 255, 120, 0.15)' : 'rgba(255, 209, 102, 0.15)',
                      color: systemStatus.device === 'cuda' ? '#a8ff78' : '#ffd166',
                      border: systemStatus.device === 'cuda' ? '1px solid rgba(168, 255, 120, 0.3)' : '1px solid rgba(255, 209, 102, 0.3)',
                      fontWeight: 600,
                    }}
                  />
                </Tooltip>
              )}

              <Divider orientation="vertical" flexItem sx={{ borderColor: themeMode === 'dark' ? 'rgba(0, 217, 255, 0.2)' : 'rgba(0, 153, 204, 0.2)' }} />

              <ToggleButtonGroup
                value={dataset}
                exclusive
                onChange={(e, newDataset) => {
                  if (newDataset !== null) setDataset(newDataset);
                }}
                sx={{
                  backgroundColor: 'rgba(0, 0, 0, 0.2)',
                  padding: '4px',
                  borderRadius: '8px',
                  border: '1px solid rgba(0, 217, 255, 0.2)',
                }}
              >
                <Tooltip title="CIFAR-10 Image Classification">
                  <ToggleButton
                    value="cifar10"
                    sx={{
                      padding: '8px 16px',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      transition: 'all 0.3s ease',
                    }}
                  >
                    <DatasetLinked sx={{ mr: 1, fontSize: 18 }} />
                    CIFAR-10
                  </ToggleButton>
                </Tooltip>
                <Tooltip title="AG News Text Classification">
                  <ToggleButton
                    value="agnews"
                    sx={{
                      padding: '8px 16px',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      transition: 'all 0.3s ease',
                    }}
                  >
                    <Build sx={{ mr: 1, fontSize: 18 }} />
                    AG News
                  </ToggleButton>
                </Tooltip>
              </ToggleButtonGroup>

              <Chip
                label={`${stats.total} Benchmarks`}
                size="small"
                icon={<BarChart />}
                sx={{
                  backgroundColor: themeMode === 'dark' ? 'rgba(168, 255, 120, 0.1)' : 'rgba(102, 187, 106, 0.12)',
                  color: themeMode === 'dark' ? '#a8ff78' : '#2e7d32',
                  border: themeMode === 'dark' ? '1px solid rgba(168, 255, 120, 0.3)' : '1px solid rgba(102, 187, 106, 0.35)',
                }}
              />

              <Tooltip title="! Demo retraining mode uses fewer batches/epochs.(Only for demonstration)">
                <Chip
                  label={retrainingDemoMode ? 'Demo Retrain: ON' : 'Demo Retrain: OFF'}
                  size="small"
                  onClick={handleToggleDemoMode}
                  sx={{
                    cursor: updatingDemoMode ? 'wait' : 'pointer',
                    opacity: updatingDemoMode ? 0.7 : 1,
                    backgroundColor: retrainingDemoMode
                      ? (themeMode === 'dark' ? 'rgba(255, 209, 102, 0.14)' : 'rgba(255, 167, 38, 0.16)')
                      : (themeMode === 'dark' ? 'rgba(148, 163, 184, 0.12)' : 'rgba(96, 125, 139, 0.14)'),
                    color: retrainingDemoMode
                      ? (themeMode === 'dark' ? '#ffd166' : '#e65100')
                      : (themeMode === 'dark' ? '#cbd5e1' : '#455a64'),
                    border: retrainingDemoMode
                      ? (themeMode === 'dark' ? '1px solid rgba(255, 209, 102, 0.35)' : '1px solid rgba(255, 167, 38, 0.35)')
                      : (themeMode === 'dark' ? '1px solid rgba(148, 163, 184, 0.25)' : '1px solid rgba(96, 125, 139, 0.30)'),
                    fontWeight: 700,
                  }}
                />
              </Tooltip>

              {/* Theme Toggle */}
              <Tooltip title={themeMode === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
                <IconButton
                  onClick={() => setThemeMode(themeMode === 'dark' ? 'light' : 'dark')}
                  sx={{
                    color: themeMode === 'dark' ? '#00d9ff' : '#0099cc',
                    '&:hover': {
                      backgroundColor: themeMode === 'dark' ? 'rgba(0, 217, 255, 0.1)' : 'rgba(0, 153, 204, 0.1)',
                    },
                  }}
                >
                  {themeMode === 'dark' ? <LightMode /> : <DarkMode />}
                </IconButton>
              </Tooltip>

              <Tooltip title="Help (Press ?)">
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setShowHelp(true)}
                  sx={{
                    borderColor: themeMode === 'dark' ? '#00d9ff' : '#0099cc',
                    color: themeMode === 'dark' ? '#00d9ff' : '#0099cc',
                    '&:hover': {
                      borderColor: themeMode === 'dark' ? '#4df7ff' : '#00d9ff',
                      backgroundColor: themeMode === 'dark' ? 'rgba(0, 217, 255, 0.1)' : 'rgba(0, 153, 204, 0.1)',
                    },
                  }}
                >
                  Help
                </Button>
              </Tooltip>
            </Box>
          </Toolbar>
        </AppBar>

        {/* Main Content Area */}
        <Box sx={{ display: 'flex', flex: 1, position: 'relative', zIndex: 1 }}>
          {/* Desktop Sidebar */}
          <Box sx={{ display: { xs: 'none', md: 'block' }, borderRight: themeMode === 'dark' ? '1px solid #1e2430' : '1px solid #e0e0e0' }}>
            {sidebarContent}
          </Box>

          {/* Mobile Drawer */}
          <Drawer
            anchor="left"
            open={drawerOpen}
            onClose={() => setDrawerOpen(false)}
            PaperProps={{
              sx: {
                backgroundColor: themeMode === 'dark' ? '#111318' : '#f5f7fa',
                borderRight: themeMode === 'dark' ? '1px solid #1e2430' : '1px solid #e0e0e0',
              },
            }}
          >
            {sidebarContent}
          </Drawer>

          {/* Main Content */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'auto' }}>
            <Container maxWidth="xl" sx={{ flex: 1, padding: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Paper
                sx={{
                  p: 2.25,
                  borderRadius: 3,
                  background: themeMode === 'dark'
                    ? 'linear-gradient(135deg, rgba(17,19,24,0.95) 0%, rgba(24,28,36,0.95) 100%)'
                    : 'linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(242,247,255,0.96) 100%)',
                  border: themeMode === 'dark' ? '1px solid rgba(0, 217, 255, 0.25)' : '1px solid rgba(0, 153, 204, 0.25)',
                  boxShadow: themeMode === 'dark' ? '0 14px 40px rgba(0, 217, 255, 0.08)' : '0 14px 40px rgba(0, 153, 204, 0.1)',
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: { xs: 'flex-start', md: 'center' }, gap: 2, flexWrap: 'wrap' }}>
                  <Box>
                    <Box sx={{ color: themeMode === 'dark' ? '#00d9ff' : '#0099cc', fontSize: '0.78rem', letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: 800, mb: 0.5 }}>
                      Panel Showcase Ready
                    </Box>
                    <Box sx={{ color: themeMode === 'dark' ? '#e2e8f0' : '#1a237e', fontSize: '1.1rem', fontWeight: 800 }}>
                      CERBERUS v2 • {dataset === 'cifar10' ? 'CIFAR-10 Vision Security' : 'AG News Text Security'}
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={`${stats.total} Runs`}
                      size="small"
                      sx={{
                        fontWeight: 700,
                        backgroundColor: themeMode === 'dark' ? 'rgba(0, 217, 255, 0.12)' : 'rgba(0, 153, 204, 0.12)',
                        border: themeMode === 'dark' ? '1px solid rgba(0, 217, 255, 0.35)' : '1px solid rgba(0, 153, 204, 0.35)',
                      }}
                    />
                    <Chip
                      label={`${Math.round(stats.avgSuccess || 0)}% Avg Success`}
                      size="small"
                      sx={{
                        fontWeight: 700,
                        backgroundColor: themeMode === 'dark' ? 'rgba(255, 209, 102, 0.12)' : 'rgba(255, 167, 38, 0.13)',
                        color: themeMode === 'dark' ? '#ffd166' : '#e65100',
                        border: themeMode === 'dark' ? '1px solid rgba(255, 209, 102, 0.35)' : '1px solid rgba(255, 167, 38, 0.35)',
                      }}
                    />
                    <Chip
                      label={systemStatus?.device === 'cuda' ? 'Realtime GPU' : 'Realtime CPU'}
                      size="small"
                      sx={{
                        fontWeight: 700,
                        backgroundColor: themeMode === 'dark' ? 'rgba(168, 255, 120, 0.12)' : 'rgba(102, 187, 106, 0.14)',
                        color: themeMode === 'dark' ? '#a8ff78' : '#2e7d32',
                        border: themeMode === 'dark' ? '1px solid rgba(168, 255, 120, 0.35)' : '1px solid rgba(102, 187, 106, 0.35)',
                      }}
                    />
                  </Box>
                </Box>
              </Paper>

              <Paper
                className={!pipelineExpanded && pipelineAttention ? 'pipeline-attention-card' : ''}
                sx={{
                  borderRadius: 3,
                  overflow: 'hidden',
                  border: themeMode === 'dark' ? '1px solid rgba(0, 217, 255, 0.22)' : '1px solid rgba(0, 153, 204, 0.22)',
                  background: themeMode === 'dark'
                    ? 'linear-gradient(135deg, rgba(17,19,24,0.92) 0%, rgba(24,28,36,0.92) 100%)'
                    : 'linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(242,247,255,0.94) 100%)',
                }}
              >
                <Button
                  fullWidth
                  onClick={() => {
                    setPipelineExpanded((prev) => !prev);
                    setPipelineAttention(false);
                  }}
                  sx={{
                    justifyContent: 'space-between',
                    px: 2,
                    py: 1.35,
                    borderRadius: 0,
                    color: themeMode === 'dark' ? '#e2e8f0' : '#1a237e',
                    textTransform: 'none',
                    '&:hover': {
                      backgroundColor: themeMode === 'dark' ? 'rgba(0, 217, 255, 0.08)' : 'rgba(0, 153, 204, 0.08)',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
                    <AutoAwesome sx={{ color: themeMode === 'dark' ? '#00d9ff' : '#0099cc', fontSize: 20 }} />
                    <Box sx={{ textAlign: 'left' }}>
                      <Box sx={{ fontWeight: 800, fontSize: '0.98rem', color: themeMode === 'dark' ? '#00d9ff' : '#0099cc' }}>
                        Attack → Retrain → Defend Pipeline
                      </Box>
                      <Box sx={{ fontSize: '0.76rem', color: themeMode === 'dark' ? '#7a8aa1' : '#5f6f8a' }}>
                        {pipelineExpanded ? 'Click to collapse' : 'Click to expand architecture flow'}
                      </Box>
                    </Box>
                  </Box>

                  <KeyboardArrowDown
                    sx={{
                      color: themeMode === 'dark' ? '#00d9ff' : '#0099cc',
                      transform: pipelineExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.25s ease',
                    }}
                  />
                </Button>

                <Collapse in={pipelineExpanded} timeout={320} unmountOnExit>
                  <Box sx={{ p: 2.2, pt: 1.2 }}>
                    <PipelineDiagram />
                  </Box>
                </Collapse>
              </Paper>

              {/* Error Alert */}
              {error && (
                <Alert
                  severity="error"
                  onClose={() => setError(null)}
                  sx={{
                    backgroundColor: 'rgba(255, 77, 109, 0.1)',
                    border: '1px solid rgba(255, 77, 109, 0.3)',
                    color: '#ff4d6d',
                  }}
                >
                  {error}
                </Alert>
              )}

              {/* Content Panels */}
              <Box sx={{ position: 'relative', flex: 1 }}>
                {loading && (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
                    <CircularProgress sx={{ color: '#00d9ff' }} />
                  </Box>
                )}
                
                {!loading && activeTab === 'attack' && (
                  <AttackPanel onRun={handleAttackRun} loading={loading} dataset={dataset} />
                )}
                {!loading && activeTab === 'defense' && (
                  <DefensePanel onRun={handleDefenseRun} loading={loading} dataset={dataset} />
                )}
                {!loading && activeTab === 'transfer' && (
                  <TransferPanel onRun={handleTransferAnalysis} loading={loading} dataset={dataset} />
                )}
                {!loading && activeTab === 'experiments' && (
                  <ExperimentsPanel />
                )}
              </Box>
            </Container>

            {/* Results Panel */}
            {results && (
              <Paper
                sx={{
                  margin: 3,
                  marginTop: 0,
                  padding: 2,
                  backgroundColor: 'rgba(0, 217, 255, 0.05)',
                  border: '1px solid rgba(0, 217, 255, 0.2)',
                }}
              >
                <ResultsPanel results={results} retrainingStatus={retrainingStatus} />
              </Paper>
            )}

            {results && (
              <Paper
                sx={{
                  margin: 3,
                  marginTop: 0,
                  padding: 2,
                  backgroundColor: themeMode === 'dark' ? 'rgba(168, 255, 120, 0.04)' : 'rgba(168, 255, 120, 0.08)',
                  border: '1px solid rgba(168, 255, 120, 0.2)',
                }}
              >
                <RetrainingHistory />
              </Paper>
            )}
          </Box>
        </Box>

        <RetrainingStatus />
      </Box>

      <HelpModal isOpen={showHelp} onClose={() => setShowHelp(false)} />
    </>
  );
}

function App() {
  return (
    <AppThemeProvider>
      <AppWithTheme />
    </AppThemeProvider>
  );
}

function AppWithTheme() {
  const { themeMode } = useAppTheme();

  return (
    <ThemeProvider theme={createAppTheme(themeMode)}>
      <CustomThemeProvider>
        <CssBaseline />
        <AppContent />
      </CustomThemeProvider>
    </ThemeProvider>
  );
}

export default App;
