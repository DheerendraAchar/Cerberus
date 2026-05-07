import React from 'react';
import {
  FormControl,
  Select,
  MenuItem,
  ListItemText,
  styled,
} from '@mui/material';
import { useAppTheme } from '../context/AppThemeContext';

const StyledSelect = styled(Select)(({ theme }) => ({
  borderRadius: '10px',
  fontSize: '0.95rem',
  fontFamily: "'Space Mono', monospace",
  fontWeight: 600,
  '& .MuiOutlinedInput-notchedOutline': {
    borderWidth: '2px',
    borderColor: theme.palette.mode === 'dark' ? '#2a3f5f' : '#cbd5e1',
  },
  '&:hover .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.palette.mode === 'dark' ? '#00e5ff' : '#0099cc',
  },
  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
    borderColor: theme.palette.mode === 'dark' ? '#00e5ff' : '#0099cc',
    boxShadow: `0 0 0 5px ${theme.palette.mode === 'dark' ? 'rgba(0,229,255,0.25)' : 'rgba(0,153,204,0.25)'}`,
  },
  '& .MuiSelect-select': {
    padding: '14px 18px !important',
  },
  '& .MuiSvgIcon-root': {
    color: theme.palette.mode === 'dark' ? '#00e5ff' : '#0099cc',
  },
}));

export default function CustomFormSelect({
  label,
  name,
  value,
  onChange,
  options,
  disabled = false,
  icon: Icon = null,
  groupedOptions = null,
}) {
  const { isDark } = useAppTheme();

  // Helper function to find the label for a given value
  const getSelectedLabel = (selectedValue) => {
    if (!selectedValue) return null;
    
    if (groupedOptions) {
      for (const group of groupedOptions) {
        const found = group.options.find(opt => opt.value === selectedValue);
        if (found) return found.label;
      }
    } else if (options) {
      const found = options.find(opt => opt.value === selectedValue);
      if (found) return found.label;
    }
    return null;
  };

  const handleSelectChange = (event, child) => {
    console.log('=== SELECT CHANGE ===');
    console.log('event:', event);
    console.log('event.target:', event.target);
    console.log('event.target.value:', event.target.value);
    console.log('event.target.name:', event.target.name);
    console.log('child:', child);
    
    const compatibleEvent = {
      target: {
        name: name,
        value: event.target.value
      }
    };
    console.log('compatibleEvent being passed to parent:', compatibleEvent);
    onChange(compatibleEvent);
  };



  console.log('=== StyledSelect RENDER ===');
  console.log('name:', name, 'value:', value, 'label:', label);

  return (
    <div style={{ position: 'relative' }}>
      <FormControl fullWidth variant="outlined" disabled={disabled}>
        {label && (
          <label style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '8px',
            fontSize: '0.9rem',
            fontWeight: 600,
            color: isDark ? '#a8b8d8' : '#475569',
          }}>
            {Icon && <Icon size={16} style={{ color: isDark ? '#667eea' : '#667eea' }} />}
            {label}
          </label>
        )}
        <StyledSelect
          name={name}
          value={value}
          onChange={handleSelectChange}
          disabled={disabled}
          displayEmpty
          MenuProps={{
            PaperProps: {
              sx: {
                maxHeight: 300,
                zIndex: 1401,
              }
            },
          }}
          renderValue={(selected) => {
            console.log('renderValue - selected:', selected);
            if (selected === '' || selected === undefined || selected === null) {
              return <span style={{ color: isDark ? '#6b7e98' : '#94a3b8' }}>Select {label || 'option'}...</span>;
            }
            const selectedLabel = getSelectedLabel(selected);
            console.log('renderValue - selectedLabel:', selectedLabel);
            return selectedLabel || selected;
          }}
        >
          {groupedOptions
            ? groupedOptions.flatMap((group, groupIdx) => [
                <MenuItem 
                  key={`group-${groupIdx}`}
                  disabled 
                  sx={{ opacity: 1, fontWeight: 700, backgroundColor: isDark ? '#0a0c10' : '#f5f7fa' }}
                >
                  {group.label}
                </MenuItem>,
                ...group.options.map((opt) => (
                  <MenuItem 
                    key={opt.value} 
                    value={opt.value}
                    sx={{
                      fontFamily: "'Space Mono', monospace",
                      fontSize: '0.95rem',
                      padding: '12px 18px',
                      borderRadius: '6px',
                      '&:hover': {
                        backgroundColor: isDark ? 'rgba(0, 229, 255, 0.15)' : 'rgba(0, 153, 204, 0.15)',
                      },
                      '&.Mui-selected': {
                        backgroundColor: isDark ? 'rgba(0, 229, 255, 0.25)' : 'rgba(0, 153, 204, 0.25)',
                        color: isDark ? '#00e5ff' : '#0099cc',
                        fontWeight: 700,
                      },
                    }}
                  >
                    <ListItemText primary={opt.label} />
                  </MenuItem>
                ))
              ])
            : options && options.map((opt) => (
                <MenuItem 
                  key={opt.value} 
                  value={opt.value}
                  sx={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: '0.95rem',
                    padding: '12px 18px',
                    borderRadius: '6px',
                    '&:hover': {
                      backgroundColor: isDark ? 'rgba(0, 229, 255, 0.15)' : 'rgba(0, 153, 204, 0.15)',
                    },
                    '&.Mui-selected': {
                      backgroundColor: isDark ? 'rgba(0, 229, 255, 0.25)' : 'rgba(0, 153, 204, 0.25)',
                      color: isDark ? '#00e5ff' : '#0099cc',
                      fontWeight: 700,
                    },
                  }}
                >
                  <ListItemText primary={opt.label} />
                </MenuItem>
              ))
          }
        </StyledSelect>
      </FormControl>
    </div>
  );
}
