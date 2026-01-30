"""
Conversation Manager for Mental Health Chatbot
Manages conversation history and context
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context for the chatbot.
    """
    
    def __init__(self, max_history: int = 50, context_window: int = 5):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum number of messages to store
            context_window: Number of recent messages to use for context
        """
        self.max_history = max_history
        self.context_window = context_window
        self.conversation_history = []
        self.session_start = datetime.now()
    
    def add_message(
        self,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            emotion: Detected emotion (for user messages)
            confidence: Emotion confidence score
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion,
            'confidence': confidence
        }
        
        self.conversation_history.append(message)
        
        # Trim history if exceeds max length
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context(self, num_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get recent conversation context.
        
        Args:
            num_messages: Number of recent messages (uses context_window if None)
            
        Returns:
            List of recent messages
        """
        if num_messages is None:
            num_messages = self.context_window
        
        return self.conversation_history[-num_messages:]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.session_start = datetime.now()
        logger.info("Conversation history cleared")
    
    def export_conversation(self, format: str = 'txt') -> str:
        """
        Export conversation history.
        
        Args:
            format: Export format ('json', 'txt')
            
        Returns:
            Exported conversation string
        """
        if format == 'json':
            return json.dumps(self.conversation_history, indent=2)
        elif format == 'txt':
            lines = []
            for msg in self.conversation_history:
                role = msg['role'].upper()
                content = msg['content']
                timestamp = msg.get('timestamp', '')
                emotion = msg.get('emotion', '')
                
                if emotion:
                    lines.append(f"[{timestamp}] {role} ({emotion}): {content}")
                else:
                    lines.append(f"[{timestamp}] {role}: {content}")
            
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

