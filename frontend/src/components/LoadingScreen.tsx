import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';

const YODA_PUNS = [
  "Ready, the app is not. Wait, you must.",
  "Patience, you have? Good. Loading, it is.",
  "Fast, the backend is not. Meditate, you should.",
  "Complete, the startup is not. Hmm, loading still.",
  "Rush, you must not. Ready soon, it will be.",
  "The Force, it needs time. Load, it does.",
  "Hurry, you should not. Patient, be you.",
  "Wise, you are, to wait. Loading, the app is.",
  "Loaded, it will be. Believe, you must.",
  "The way of the backend, slow it is. Accept, you must.",
  "Ready, almost there it is. Wait more, you must.",
  "Force strong, the loading is. Patience, your ally is.",
  "Startup complete, soon it will be. Trust the process, you must.",
  "Yoda, I am. Wait, you must. Ready, the app will be.",
];

interface LoadingScreenProps {
  isReady: boolean;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ isReady }) => {
  const [currentPun, setCurrentPun] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPun((prev) => (prev + 1) % YODA_PUNS.length);
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  if (isReady) {
    return null;
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #1a472a 0%, #2d5a3d 50%, #1a472a 100%)',
        zIndex: 9999,
        gap: 4,
      }}
    >
      {/* Yoda Image */}
      <Box
        component="img"
        src="/y.png"
        alt="Yoda"
        sx={{
          width: 200,
          height: 200,
          objectFit: 'contain',
          animation: 'float 3s ease-in-out infinite',
          '@keyframes float': {
            '0%, 100%': { transform: 'translateY(0px)' },
            '50%': { transform: 'translateY(-15px)' },
          },
        }}
      />

      {/* Loading Spinner */}
      <CircularProgress
        sx={{
          color: '#4CAF50',
          '& .MuiCircularProgress-circle': {
            strokeLinecap: 'round',
          },
        }}
        size={60}
      />

      {/* Main Loading Text */}
      <Typography
        variant="h4"
        sx={{
          color: '#4CAF50',
          fontWeight: 700,
          textAlign: 'center',
          fontSize: '1.8rem',
        }}
      >
        Initializing the Force...
      </Typography>

      {/* Yoda Pun - Animated */}
      <Box
        sx={{
          minHeight: '80px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          maxWidth: '600px',
          px: 3,
        }}
      >
        <Typography
          key={currentPun}
          variant="body1"
          sx={{
            color: '#81C784',
            fontSize: '1.1rem',
            textAlign: 'center',
            fontStyle: 'italic',
            animation: 'fadeInOut 3s ease-in-out',
            '@keyframes fadeInOut': {
              '0%': { opacity: 0, transform: 'translateY(10px)' },
              '10%': { opacity: 1, transform: 'translateY(0px)' },
              '90%': { opacity: 1, transform: 'translateY(0px)' },
              '100%': { opacity: 0, transform: 'translateY(-10px)' },
            },
          }}
        >
          "{YODA_PUNS[currentPun]}"
        </Typography>
      </Box>

      {/* Status Text */}
      <Typography
        variant="body2"
        sx={{
          color: '#66BB6A',
          fontSize: '0.9rem',
          textAlign: 'center',
        }}
      >
        Waiting for backend to be ready...
      </Typography>

      {/* Loading dots animation */}
      <Box
        sx={{
          display: 'flex',
          gap: 1,
          mt: 2,
        }}
      >
        {[0, 1, 2].map((i) => (
          <Box
            key={i}
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#4CAF50',
              animation: 'pulse 1.4s infinite',
              animationDelay: `${i * 0.2}s`,
              '@keyframes pulse': {
                '0%, 100%': { opacity: 0.3 },
                '50%': { opacity: 1 },
              },
            }}
          />
        ))}
      </Box>
    </Box>
  );
};

export default LoadingScreen;
