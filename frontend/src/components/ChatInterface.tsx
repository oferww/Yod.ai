import React, { useEffect, useMemo, useRef } from 'react';
import { Box, Typography, IconButton, Tooltip } from '@mui/material';
import { Refresh as RefreshIcon, Info as InfoIcon } from '@mui/icons-material';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import TopBar from './TopBar';
import { useChat } from '../contexts/ChatContext';

const SUBTITLE_OPTIONS = [
  'The right laptop for you, I will find. ⭐',
  'Choose wisely, you must. Help you, I will.',
  'Judge laptops by their specs, do not. Judge them by how they fit, you should.',
  'Your perfect laptop, find we will.',
  'Much to learn about laptops, you have. Teach you, I shall.',
  'Do or do not buy. There is no try... before Yod.ai helps. ⭐⭐',
  'May the specs be with you.',
  'Strong with the GPU, this one is.',
  'Size matters not... except for RAM and storage.',
  'In the cloud, your perfect laptop is. Find it, we will.',
  'Wise laptop choices, you shall make.',
  'Your tech guide, I am.',
  'Find your laptop destiny.',
  'Laptop wisdom, seek you must.',
  'Budget concerns, understand I do. ⭐',
  'YODA + AI = YODAI. Smart, this is. ⭐⭐⭐',
  'Confusion about laptops, dispel I will.',
  'The wise choice in laptop shopping.',
  'Recommend laptops, I do. Regret them, you will not.',
  'Powerful you will become... at choosing laptops.',
  'Better than googling for hours, I am.',
  'Hmm. Gaming or work? Both, you want? Show you, I can.',
  '900 years old I am. Know laptops, I do.',
  'Much RAM you need? Or much you THINK you need? Know the difference, I will.',
  'Try not. Do... with Yod.ai.'
];

const ChatInterface: React.FC = () => {
  const { messages, isTyping, sendMessage, clearChat } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const randomSubtitle = useMemo(
    () => SUBTITLE_OPTIONS[Math.floor(Math.random() * SUBTITLE_OPTIONS.length)],
    []
  );

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      clearChat();
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        margin: 0,
        padding: 0,
        background: 'linear-gradient(135deg, #f0f9f4 0%, #e8f5e9 50%, #f1f8f4 100%)',
        overflow: 'hidden',
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '300px',
          background: 'radial-gradient(ellipse at top, rgba(76, 175, 80, 0.15) 0%, transparent 70%)',
          pointerEvents: 'none',
        }
      }}
    >
      {/* Top Bar */}
      <TopBar hasMessages={messages.length > 0} />
      
      {/* Messages container */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          maxWidth: '900px',
          width: '100%',
          margin: '0 auto',
          padding: '30px 20px',
          position: 'relative',
          zIndex: 1,
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(76, 175, 80, 0.05)',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'rgba(76, 175, 80, 0.3)',
            borderRadius: '4px',
            '&:hover': {
              background: 'rgba(76, 175, 80, 0.5)',
            }
          },
        }}
      >
        {messages.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              textAlign: 'center',
              gap: 3,
            }}
          >
            <Box
              component="img"
              src="/y.png"
              alt="Yod.ai Assistant"
              sx={{
                width: 280,
                height: 280,
                // borderRadius: '24px',
                objectFit: 'contain',
                animation: 'float 3s ease-in-out infinite',
                '@keyframes float': {
                  '0%, 100%': { transform: 'translateY(0px)' },
                  '50%': { transform: 'translateY(-10px)' },
                },
              }}
            />
            <Box>
              <Typography 
                variant="h4" 
                gutterBottom 
                sx={{ 
                  color: '#2E7D32',
                  fontWeight: 700,
                  mb: 2,
                }}
              >
                Welcome to Yod.ai, you are.
              </Typography>
              <Typography 
                variant="body1" 
                sx={{ 
                  color: '#4CAF50',
                  fontSize: '1.1rem',
                  mb: 1,
                  justifyContent: 'center',
                }}
              >
                Tell me what you seek, and the perfect laptop for your needs and budget, I will find. Hmmmmm.
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: '#81C784',
                  lineHeight: 1.7,
                  justifyContent: 'center',

                }}
              >
                {randomSubtitle}
              </Typography>
            </Box>
            <Box
              sx={{
                display: 'flex',
                gap: 2,
                mt: 2,
                flexWrap: 'wrap',
                justifyContent: 'center',
              }}
            >
              {['Gaming', 'Work', 'Student', 'Creative'].map((tag) => (
                <Box
                  key={tag}
                  component="button"
                  type="button"
                  onClick={() => void sendMessage(`I need a ${tag.toLowerCase()} laptop.`)}
                  sx={{
                    px: 3,
                    py: 1.5,
                    borderRadius: '20px',
                    background: 'rgba(76, 175, 80, 0.1)',
                    border: '1px solid rgba(76, 175, 80, 0.2)',
                    color: '#4CAF50',
                    fontWeight: 600,
                    fontSize: '0.9rem',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    textTransform: 'none',
                    outline: 'none',
                    '&:hover': {
                      background: 'rgba(76, 175, 80, 0.2)',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 4px 12px rgba(76, 175, 80, 0.2)',
                    }
                  }}
                >
                  {tag}
                </Box>
              ))}
            </Box>
          </Box>
        ) : (
          messages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))
        )}
        <div ref={messagesEndRef} />
        
        {isTyping && messages.length > 0 && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              ml: 2,
            }}
          >
            {[0, 1, 2].map((i) => (
              <Box
                key={i}
                sx={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%)',
                  animation: 'bounce 1.4s infinite',
                  animationDelay: `${i * 0.2}s`,
                  '@keyframes bounce': {
                    '0%, 80%, 100%': { 
                      transform: 'scale(0.8)',
                      opacity: 0.5,
                    },
                    '40%': { 
                      transform: 'scale(1.2)',
                      opacity: 1,
                    },
                  },
                }}
              />
            ))}
          </Box>
        )}
      </Box>

      {/* Input area */}
      <Box
        sx={{
          p: 0,
          m: 0,
          background: 'transparent',
          position: 'sticky',
          bottom: 0,
          width: '100%',
          maxWidth: '900px',
          margin: '0 auto',
          boxSizing: 'border-box',
        }}
      >
        <ChatInput onSendMessage={sendMessage} isTyping={isTyping} />
      </Box>
    </Box>
  );
};

export default ChatInterface;