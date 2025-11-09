import React, { useState, useRef, useEffect } from 'react';
import { Box, TextField, IconButton, CircularProgress } from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';

interface ChatInputProps {
  onSendMessage: (content: string) => Promise<void>;
  isTyping: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isTyping }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() === '' || isTyping) return;
    
    const msg = message.trim();
    setMessage('');
    
    try {
      await onSendMessage(msg);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [message]);

  return (
    <Box 
      component="form" 
      onSubmit={handleSubmit}
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        p: 0,
        bgcolor: 'transparent',
      }}
    >
      <TextField
        inputRef={textareaRef}
        fullWidth
        multiline
        maxRows={6}
        variant="outlined"
        placeholder="Type your message..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isTyping}
        sx={{
          '& .MuiOutlinedInput-root': {
            borderRadius: 2,
            bgcolor: '#f5f5f5',
            padding: '8px 12px',
            '&.Mui-focused fieldset': {
              borderColor: '#4caf50', // Green border when focused
              borderWidth: '2px',
            },
          },
        }}
      />
      <IconButton
        type="submit"
        color="primary"
        disabled={message.trim() === '' || isTyping}
        sx={{
          height: 48,
          width: 48,
          bgcolor: '#4caf50', // Green background
          color: 'white',
          '&:hover': {
            bgcolor: '#45a049', // Darker green on hover
          },
          '&:disabled': {
            bgcolor: 'action.disabled',
            color: 'action.disabled',
          },
        }}
      >
        {isTyping ? (
          <CircularProgress size={24} color="inherit" />
        ) : (
          <SendIcon />
        )}
      </IconButton>
    </Box>
  );
};

export default ChatInput;