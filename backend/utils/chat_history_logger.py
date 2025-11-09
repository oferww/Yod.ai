"""
Chat history logger for persisting conversations to JSON files.
Saves all chat messages, user preferences, and recommendations in real-time.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading


class ChatHistoryLogger:
    """Logs chat conversations to JSON files with preferences and recommendations."""
    
    def __init__(self, logs_dir: str = "chat_logs"):
        """Initialize the chat history logger.
        
        Args:
            logs_dir: Directory to store chat logs (default: chat_logs)
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()  # Thread-safe file operations
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new chat session structure.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session structure with metadata
        """
        return {
            "session_id": session_id,
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "messages": [],
            "user_preferences": {},
            "final_recommendation": None,
            "recommendation_given_at": None,
            "total_messages": 0
        }
    
    def get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Path to the session JSON file
        """
        return self.logs_dir / f"session_{session_id}.json"
    
    def log_message(self, session_id: str, role: str, content: str, 
                   timestamp: Optional[str] = None) -> None:
        """Log a single message to the chat history.
        
        Args:
            session_id: Session ID
            role: "user" or "assistant"
            content: Message content
            timestamp: Optional ISO timestamp (uses current time if not provided)
        """
        if not timestamp:
            timestamp = datetime.utcnow().isoformat()
        
        with self._lock:
            session_file = self.get_session_file(session_id)
            
            # Load or create session
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
            else:
                session = self.create_session(session_id)
            
            # Add message
            session["messages"].append({
                "role": role,
                "content": content,
                "timestamp": timestamp
            })
            session["total_messages"] = len(session["messages"])
            
            # Save session
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
    
    def update_preferences(self, session_id: str, preferences: Dict[str, Any]) -> None:
        """Update user preferences in the session.
        
        Args:
            session_id: Session ID
            preferences: Dictionary of user preferences (brand, cpu, gpu, ram, storage, max_price)
        """
        with self._lock:
            session_file = self.get_session_file(session_id)
            
            # Load or create session
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
            else:
                session = self.create_session(session_id)
            
            # Update preferences (merge with existing)
            session["user_preferences"].update(preferences)
            
            # Save session
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
    
    def log_recommendation(self, session_id: str, recommendation: Dict[str, Any]) -> None:
        """Log a product recommendation given to the user.
        
        Args:
            session_id: Session ID
            recommendation: Recommendation data containing product info and details
        """
        with self._lock:
            session_file = self.get_session_file(session_id)
            
            # Load or create session
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
            else:
                session = self.create_session(session_id)
            
            # Update recommendation
            session["final_recommendation"] = recommendation
            session["recommendation_given_at"] = datetime.utcnow().isoformat()
            
            # Save session
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
    
    def end_session(self, session_id: str) -> None:
        """Mark a session as ended.
        
        Args:
            session_id: Session ID
        """
        with self._lock:
            session_file = self.get_session_file(session_id)
            
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                
                session["ended_at"] = datetime.utcnow().isoformat()
                
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session, f, indent=2, ensure_ascii=False)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a complete session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None if not found
        """
        session_file = self.get_session_file(session_id)
        
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    
    def list_sessions(self) -> List[str]:
        """List all session IDs.
        
        Returns:
            List of session IDs
        """
        sessions = []
        for file in self.logs_dir.glob("session_*.json"):
            session_id = file.stem.replace("session_", "")
            sessions.append(session_id)
        return sorted(sessions)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve all sessions.
        
        Returns:
            List of all session data
        """
        sessions = []
        for session_id in self.list_sessions():
            session = self.get_session(session_id)
            if session:
                sessions.append(session)
        return sessions


# Global instance
_logger_instance: Optional[ChatHistoryLogger] = None


def get_chat_history_logger(logs_dir: str = "chat_logs") -> ChatHistoryLogger:
    """Get or create the global chat history logger instance.
    
    Args:
        logs_dir: Directory to store chat logs
        
    Returns:
        ChatHistoryLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ChatHistoryLogger(logs_dir)
    return _logger_instance
