import React, { createContext, useContext, useState, ReactNode, useCallback, useRef, useEffect } from 'react';
import { Message } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface ChatContextType {
  messages: Message[];
  isTyping: boolean;
  sendMessage: (content: string) => Promise<void>;
  clearChat: () => void;
  isBackendReady: boolean;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isBackendReady, setIsBackendReady] = useState(false);
  const streamingTimerRef = useRef<number | null>(null);

  // Check backend readiness on mount and periodically
  useEffect(() => {
    const checkBackendReady = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (response.ok) {
          const data = await response.json();
          // Consider backend ready if products are loaded (semantic RAG can initialize in background)
          if (data.products_loaded) {
            setIsBackendReady(true);
            return true; // Signal that backend is ready
          }
        }
      } catch (error) {
        // Backend not ready yet
        setIsBackendReady(false);
      }
      return false;
    };

    // Check immediately
    checkBackendReady().then((ready) => {
      if (ready) return; // Don't set interval if already ready

      // Check every 1000ms until backend is ready
      const interval = setInterval(async () => {
        const ready = await checkBackendReady();
        if (ready) {
          clearInterval(interval);
        }
      }, 1000);

      return () => clearInterval(interval);
    });
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    if (streamingTimerRef.current !== null) {
      clearTimeout(streamingTimerRef.current);
      streamingTimerRef.current = null;
    }
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (streamingTimerRef.current !== null) {
        clearTimeout(streamingTimerRef.current);
      }
    };
  }, []);

  const sendMessage = useCallback(async (content: string): Promise<void> => {
    // Add user message to chat
    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsTyping(true);

    // Accumulate chunks and display with delay
    let accumulatedContent = '';
    let assistantMessageAdded = false;
    let chunkQueue: string[] = [];
    let isProcessing = false;

    const updateAssistantMessage = (content: string) => {
      setMessages(prevMessages => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        
        if (!assistantMessageAdded || lastMessage?.role !== 'assistant') {
          assistantMessageAdded = true;
          return [
            ...prevMessages,
            {
              role: 'assistant' as const,
              content,
              timestamp: new Date(),
            },
          ];
        }
        
        return [
          ...prevMessages.slice(0, -1),
          {
            ...lastMessage,
            content,
          },
        ];
      });
    };

    const processQueue = () => {
      if (isProcessing || chunkQueue.length === 0) return;
      
      isProcessing = true;
      const chunk = chunkQueue.shift()!;
      
      // Split by spaces but preserve newlines
      const tokens: string[] = [];
      let currentToken = '';
      
      for (let i = 0; i < chunk.length; i++) {
        const char = chunk[i];
        
        if (char === '\n') {
          if (currentToken) {
            tokens.push(currentToken);
            currentToken = '';
          }
          tokens.push('\n');
        } else if (char === ' ' || char === '\t') {
          if (currentToken) {
            tokens.push(currentToken);
            currentToken = '';
          }
          tokens.push(' ');
        } else {
          currentToken += char;
        }
      }
      
      if (currentToken) {
        tokens.push(currentToken);
      }
      
      let tokenIndex = 0;

      const displayNextToken = () => {
        if (tokenIndex < tokens.length) {
          accumulatedContent += tokens[tokenIndex];
          updateAssistantMessage(accumulatedContent);
          tokenIndex++;

          // To make streaming faster or slower change the delay
          streamingTimerRef.current = window.setTimeout(displayNextToken, 15);
        } else {
          isProcessing = false;
          processQueue();
        }
      };

      displayNextToken();
    };

    const handleWebSocketMessage = (event: MessageEvent) => {
      try {
        const response = JSON.parse(event.data);
        
        if (response.type === 'error') {
          console.error('Error from server:', response.content);
          setIsTyping(false);
          return;
        }

        if (response.type === 'chat_chunk' || response.type === 'chat') {
          const chunkContent = response.content || '';
          if (chunkContent) {
            chunkQueue.push(chunkContent);
            processQueue();
          }

          if (response.is_final) {
            // Wait for queue to finish, then stop typing indicator
            const checkComplete = () => {
              if (chunkQueue.length === 0 && !isProcessing) {
                setIsTyping(false);
              } else {
                setTimeout(checkComplete, 100);
              }
            };
            checkComplete();
          }
        }
      } catch (error) {
        console.error('Error processing message:', error);
        setIsTyping(false);
      }
    };

    try {
      // Initialize WebSocket if not already connected
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = API_URL.replace(/^http/, 'ws') + '/ws';

        console.log('🔌 Attempting WebSocket');
        console.log('API_URL:', API_URL);
        console.log('wsUrl:', wsUrl);

        const newSocket = new WebSocket(wsUrl);

        newSocket.onmessage = handleWebSocketMessage;

        newSocket.onopen = () => {
          // Send the message once the connection is open
          newSocket.send(JSON.stringify({
            type: "chat",
            messages: [{ role: 'user', content }]
          }));
        };

        newSocket.onclose = () => {
          setSocket(currentSocket => (currentSocket === newSocket ? null : currentSocket));
          setIsTyping(false);
        };

        newSocket.onerror = (error) => {
          console.error('WebSocket error:', error);
          setSocket(currentSocket => (currentSocket === newSocket ? null : currentSocket));
          setIsTyping(false);
        };

        setSocket(newSocket);
      } else {
        socket.onmessage = handleWebSocketMessage;
        // If socket is already open, send the message
        socket.send(JSON.stringify({
          type: "chat",
          messages: [{ role: 'user', content }]
        }));
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setIsTyping(false);

      // Fallback to HTTP if WebSocket fails
      try {
        const response = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: [...messages, { role: 'user', content }].map(({ role, content }) => ({
              role,
              content,
            })),
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to send message');
        }

        const data = await response.json();

        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.content,
            timestamp: new Date(),
          },
        ]);
      } catch (httpError) {
        console.error('HTTP fallback error:', httpError);
        // Add error message to chat
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please try again.',
            timestamp: new Date(),
          },
        ]);
      } finally {
        setIsTyping(false);
      }
    }
  }, [messages, socket]);

  return (
    <ChatContext.Provider value={{ messages, isTyping, sendMessage, clearChat, isBackendReady }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
