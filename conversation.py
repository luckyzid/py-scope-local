"""Conversation memory management for context-aware responses."""
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
import json
from pathlib import Path
from logger import rag_logger


class Message(BaseModel):
    """Single message in conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict] = None


class Conversation(BaseModel):
    """Conversation session."""
    session_id: str
    messages: List[Message] = []
    created_at: str
    updated_at: str
    metadata: Optional[Dict] = None


class ConversationManager:
    """Manage conversation history and context."""
    
    def __init__(self, storage_dir: str = "./conversations", max_history: int = 10):
        """Initialize conversation manager.
        
        Args:
            storage_dir: Directory to store conversations
            max_history: Maximum number of messages to keep in context
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.max_history = max_history
        self.current_session: Optional[Conversation] = None
        
        rag_logger.info(f"Conversation manager initialized (max_history={max_history})")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session.
        
        Args:
            session_id: Optional session ID, auto-generated if not provided
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        now = datetime.now().isoformat()
        self.current_session = Conversation(
            session_id=session_id,
            created_at=now,
            updated_at=now
        )
        
        rag_logger.info(f"Started conversation session: {session_id}")
        return session_id
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to current session.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional metadata
        """
        if self.current_session is None:
            self.start_session()
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self.current_session.messages.append(message)
        self.current_session.updated_at = datetime.now().isoformat()
        
        # Keep only recent messages
        if len(self.current_session.messages) > self.max_history * 2:
            self.current_session.messages = self.current_session.messages[-self.max_history * 2:]
        
        rag_logger.debug(f"Added {role} message ({len(content)} chars)")
    
    def get_context(self, num_messages: Optional[int] = None) -> List[Message]:
        """Get recent conversation context.
        
        Args:
            num_messages: Number of recent messages to retrieve
            
        Returns:
            List of recent messages
        """
        if self.current_session is None or not self.current_session.messages:
            return []
        
        if num_messages is None:
            num_messages = self.max_history
        
        return self.current_session.messages[-num_messages:]
    
    def get_context_string(self, num_messages: Optional[int] = None) -> str:
        """Get conversation context as formatted string.
        
        Args:
            num_messages: Number of recent messages
            
        Returns:
            Formatted context string
        """
        messages = self.get_context(num_messages)
        
        if not messages:
            return ""
        
        context_lines = []
        for msg in messages:
            role_label = "사용자" if msg.role == "user" else "AI"
            context_lines.append(f"{role_label}: {msg.content}")
        
        return "\n".join(context_lines)
    
    def save_session(self, session_id: Optional[str] = None):
        """Save current session to disk.
        
        Args:
            session_id: Session to save, current if not specified
        """
        if self.current_session is None:
            rag_logger.warning("No active session to save")
            return
        
        if session_id is None:
            session_id = self.current_session.session_id
        
        file_path = self.storage_dir / f"{session_id}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                self.current_session.model_dump(),
                f,
                ensure_ascii=False,
                indent=2
            )
        
        rag_logger.info(f"Saved session: {session_id} ({len(self.current_session.messages)} messages)")
    
    def load_session(self, session_id: str) -> bool:
        """Load a saved session.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if loaded successfully
        """
        file_path = self.storage_dir / f"{session_id}.json"
        
        if not file_path.exists():
            rag_logger.warning(f"Session not found: {session_id}")
            return False
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.current_session = Conversation(**data)
        rag_logger.info(f"Loaded session: {session_id} ({len(self.current_session.messages)} messages)")
        return True
    
    def list_sessions(self) -> List[str]:
        """List all saved sessions.
        
        Returns:
            List of session IDs
        """
        sessions = [f.stem for f in self.storage_dir.glob("*.json")]
        return sorted(sessions, reverse=True)
    
    def clear_session(self):
        """Clear current session."""
        self.current_session = None
        rag_logger.info("Cleared current session")
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message.
        
        Returns:
            Last user message content or None
        """
        if self.current_session is None:
            return None
        
        for message in reversed(self.current_session.messages):
            if message.role == "user":
                return message.content
        
        return None
    
    def format_for_llm(self, num_messages: Optional[int] = None) -> str:
        """Format conversation history for LLM context.
        
        Args:
            num_messages: Number of recent messages
            
        Returns:
            Formatted string for LLM
        """
        messages = self.get_context(num_messages)
        
        if not messages:
            return ""
        
        formatted = "이전 대화 내용:\n"
        for msg in messages:
            role = "사용자" if msg.role == "user" else "AI"
            formatted += f"{role}: {msg.content}\n"
        
        return formatted


if __name__ == "__main__":
    # Test conversation manager
    manager = ConversationManager(max_history=5)
    
    # Start session
    session_id = manager.start_session()
    print(f"Session: {session_id}")
    
    # Add messages
    manager.add_message("user", "Galaxy S22의 배터리 용량은?")
    manager.add_message("assistant", "Galaxy S22의 배터리 용량은 3,700mAh입니다.")
    manager.add_message("user", "그것의 가격은?")
    
    # Get context
    context = manager.get_context_string()
    print(f"\nContext:\n{context}")
    
    # Save session
    manager.save_session()
    print(f"\nSaved to: {manager.storage_dir}/{session_id}.json")
