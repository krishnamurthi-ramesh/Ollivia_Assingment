"""
Shared Conversation Memory Manager
Implements sliding-window short-term memory for both assistants.
Author: Krishna Murthi
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
import time


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Sliding-window conversation memory.
    Keeps the last `max_turns` full exchanges in context.
    """

    def __init__(self, max_turns: int = 10, system_prompt: Optional[str] = None):
        self.max_turns = max_turns
        self.system_prompt = system_prompt or (
            "You are a helpful, harmless, and honest AI assistant. "
            "You provide accurate information, acknowledge uncertainty, "
            "and refuse harmful or inappropriate requests politely."
        )
        self.messages: List[Message] = []
        self.turn_count: int = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))
        self.turn_count += 1

    def _trim(self) -> None:
        """Keep only the last max_turns*2 messages (user+assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history(self) -> List[dict]:
        """Return messages as list of dicts for API consumption."""
        return [msg.to_dict() for msg in self.messages]

    def get_full_prompt(self) -> List[dict]:
        """Return full prompt including system message."""
        result = [{"role": "system", "content": self.system_prompt}]
        result.extend(self.get_history())
        return result

    def clear(self) -> None:
        self.messages = []
        self.turn_count = 0

    def get_summary_stats(self) -> dict:
        return {
            "total_turns": self.turn_count,
            "messages_in_context": len(self.messages),
            "max_turns": self.max_turns,
        }

    def to_json(self) -> str:
        return json.dumps({
            "system_prompt": self.system_prompt,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.messages
            ],
            "turn_count": self.turn_count,
        }, indent=2)
