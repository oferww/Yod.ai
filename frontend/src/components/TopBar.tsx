import React, { useState } from 'react';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Info as InfoIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { useChat } from '../contexts/ChatContext';

interface TopBarProps {
  hasMessages: boolean;
}

const TopBar: React.FC<TopBarProps> = ({ hasMessages }) => {
  const { clearChat } = useChat();
  const [infoOpen, setInfoOpen] = useState(false);

  const handleNewChat = () => {
    if (window.confirm('Start a new chat? Your current conversation will be cleared.')) {
      clearChat();
    }
  };

  const handleInfoOpen = () => {
    setInfoOpen(true);
  };

  const handleInfoClose = () => {
    setInfoOpen(false);
  };

  return (
    <>
      <AppBar
        position="static"
        sx={{
          background: 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          zIndex: 100,
        }}
      >
        <Toolbar
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            px: { xs: 2, sm: 3 },
            py: 1,
          }}
        >
          {/* Left section - App name and logo */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1.5,
              flex: 1,
            }}
          >
            <Box
              component="img"
              src="/y.png"
              alt="Yod.ai"
              sx={{
                width: 120,
                height: 120,
                objectFit: 'contain',
              }}
            />
            <Box>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 700,
                  color: '#fff',
                  fontSize: '1.3rem',
                  letterSpacing: '0.5px',
                  margin: 0,
                }}
              >
                Yod.ai
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: 'rgba(255, 255, 255, 0.8)',
                  fontSize: '0.7rem',
                  fontStyle: 'italic',
                }}
              >
                Your Laptop Guide
              </Typography>
            </Box>
          </Box>

          {/* Right section - Action buttons */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            {/* New Chat Button - visible when there are messages */}
            {hasMessages && (
              <Tooltip title="Start a new chat">
                <IconButton
                  onClick={handleNewChat}
                  sx={{
                    color: '#fff',
                    '&:hover': {
                      backgroundColor: 'rgba(255, 255, 255, 0.15)',
                    },
                  }}
                  size="small"
                >
                  <AddIcon />
                </IconButton>
              </Tooltip>
            )}

            {/* Info Button */}
            <Tooltip title="About Yod.ai">
              <IconButton
                onClick={handleInfoOpen}
                sx={{
                  color: '#fff',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                  },
                }}
                size="small"
              >
                <InfoIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Info Dialog */}
      <Dialog
        open={infoOpen}
        onClose={handleInfoClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #f0f9f4 0%, #e8f5e9 100%)',
          },
        }}
      >
        <DialogTitle
          sx={{
            background: 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)',
            color: '#fff',
            fontWeight: 700,
            fontSize: '1.3rem',
          }}
        >
          About Yod.ai
        </DialogTitle>
        <DialogContent
          sx={{
            pt: 3,
          }}
        >
          <Box sx={{ mb: 2 }}>
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 600,
                color: '#2E7D32',
                mb: 1,
              }}
            >
              🤖 What is Yod.ai?
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: '#555',
                lineHeight: 1.6,
              }}
            >
              Yod.ai is your intelligent laptop recommendation assistant. Powered by advanced AI and a comprehensive product database, I help you find the perfect laptop tailored to your needs, budget, and preferences.
            </Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          <Box sx={{ mb: 2 }}>
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 600,
                color: '#2E7D32',
                mb: 1,
              }}
            >
              ✨ Key Features
            </Typography>
            <Box component="ul" sx={{ pl: 2, color: '#555', lineHeight: 1.8 }}>
              <li>Personalized laptop recommendations</li>
              <li>Real-time product database</li>
              <li>Budget-aware suggestions</li>
              <li>Detailed specifications comparison</li>
              <li>Expert guidance on performance metrics</li>
            </Box>
          </Box>

          <Divider sx={{ my: 2 }} />

          <Box sx={{ mb: 2 }}>
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 600,
                color: '#2E7D32',
                mb: 1,
              }}
            >
              👨‍💻 Developer
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: '#555',
                lineHeight: 1.6,
              }}
            >
              Built with ❤️ using React, Python, and cutting-edge AI technology. Yod.ai combines the wisdom of Yoda with modern machine learning to guide your laptop journey.
            </Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          <Box>
            <Typography
              variant="subtitle1"
              sx={{
                fontWeight: 600,
                color: '#2E7D32',
                mb: 1,
              }}
            >
              📊 System Info
            </Typography>
            <Box
              sx={{
                background: 'rgba(76, 175, 80, 0.05)',
                p: 1.5,
                borderRadius: '8px',
                fontSize: '0.85rem',
                color: '#555',
                fontFamily: 'monospace',
                lineHeight: 1.8,
              }}
            >
              <div>Version: 1.0.0</div>
              <div>Status: Active</div>
              <div>Database: Real-time</div>
              <div>AI Model: Advanced LLM</div>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions
          sx={{
            p: 2,
            background: 'rgba(76, 175, 80, 0.05)',
          }}
        >
          <Button
            onClick={handleInfoClose}
            variant="contained"
            sx={{
              background: 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)',
              color: '#fff',
              textTransform: 'none',
              fontWeight: 600,
              '&:hover': {
                background: 'linear-gradient(135deg, #1b5e20 0%, #388e3c 100%)',
              },
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default TopBar;
