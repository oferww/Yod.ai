import React from 'react';
import { Message } from '../types';
import { Box, Typography, Paper, Avatar, useTheme } from '@mui/material';
import { 
  Person as PersonIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ReactNode } from 'react';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const theme = useTheme();
  const isUser = message.role === 'user';

  // Custom markdown components for better formatting
  const markdownComponents: any = {
    p: ({ children }: any) => (
      <Typography variant="body2" sx={{ mb: 1.2, lineHeight: 1.6, color: 'inherit' }}>
        {children}
      </Typography>
    ),
    ul: ({ children }: any) => (
      <Box 
        component="ul" 
        sx={{ 
          mb: 1.5, 
          pl: 3,
          listStyleType: 'disc',
          '& li': { 
            mb: 1,
            lineHeight: 1.7,
          },
          '& li::marker': {
            color: '#4caf50',
            fontWeight: 'bold',
          }
        }}
      >
        {children}
      </Box>
    ),
    ol: ({ children }: any) => (
      <Box 
        component="ol" 
        sx={{ 
          mb: 1.5, 
          pl: 3,
          listStyleType: 'decimal',
          '& li': { 
            mb: 1,
            lineHeight: 1.7,
          },
          '& li::marker': {
            color: '#4caf50',
            fontWeight: 'bold',
          }
        }}
      >
        {children}
      </Box>
    ),
    li: ({ children }: any) => (
      <Box component="li" sx={{ mb: 0.8 }}>
        <Typography variant="body2" sx={{ lineHeight: 1.6, color: 'inherit' }}>
          {children}
        </Typography>
      </Box>
    ),
    h1: ({ children }: any) => (
      <Typography variant="h5" sx={{ mb: 1.2, mt: 2, fontWeight: 700, color: '#2e7d32' }}>
        {children}
      </Typography>
    ),
    h2: ({ children }: any) => (
      <Typography variant="h6" sx={{ mb: 1.2, mt: 1.8, fontWeight: 700, color: '#388e3c' }}>
        {children}
      </Typography>
    ),
    h3: ({ children }: any) => (
      <Typography variant="subtitle1" sx={{ mb: 1, mt: 1.5, fontWeight: 600, color: '#4caf50' }}>
        {children}
      </Typography>
    ),
    strong: ({ children }: any) => (
      <Typography component="strong" sx={{ fontWeight: 700, color: '#2e7d32' }}>
        {children}
      </Typography>
    ),
    em: ({ children }: any) => (
      <Typography component="em" sx={{ fontStyle: 'italic', color: 'inherit' }}>
        {children}
      </Typography>
    ),
    code: ({ children }: any) => (
      <Box
        component="code"
        sx={{
          bgcolor: 'rgba(76, 175, 80, 0.1)',
          px: 0.75,
          py: 0.4,
          borderRadius: 0.75,
          fontFamily: 'monospace',
          fontSize: '0.9em',
          color: '#2e7d32',
          fontWeight: 500,
        }}
      >
        {children}
      </Box>
    ),
    hr: () => (
      <Box sx={{ my: 2, borderTop: '2px solid #e0e0e0' }} />
    ),
  };
  
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
        width: '100%',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          alignItems: 'flex-start',
          maxWidth: '80%',
        }}
      >
        {isUser ? (
          <Avatar
            sx={{
              bgcolor: theme.palette.primary.main,
              width: 40,
              height: 40,
              ml: 2,
              mr: 0,
            }}
          >
            <PersonIcon />
          </Avatar>
        ) : (
          <Box
            component="img"
            src="/y.png"
            alt="Yod.ai Bot"
            sx={{
              width: 86,
              height: 86,
              borderRadius: '50%',
              objectFit: 'contain',
              mr: 2,
            }}
          />
        )}
        <Paper
          elevation={2}
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: isUser 
              ? theme.palette.primary.light 
              : theme.palette.grey[100],
            color: isUser 
              ? theme.palette.primary.contrastText 
              : theme.palette.text.primary,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {isUser ? (
            <Typography component="div" variant="body1">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </Typography>
          ) : (
            <Box sx={{ whiteSpace: 'normal' }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {message.content}
              </ReactMarkdown>
            </Box>
          )}
          <Typography 
            variant="caption" 
            sx={{
              display: 'block',
              textAlign: 'right',
              mt: 1,
              opacity: 0.7,
              color: isUser 
                ? theme.palette.primary.contrastText 
                : theme.palette.text.secondary,
            }}
          >
            {new Date(message.timestamp).toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
};

export default ChatMessage;
