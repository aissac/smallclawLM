"""Common types and utilities for SmallClawLM gateways."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MessageType(Enum):
    TEXT = "text"
    COMMAND = "command"
    ARTIFACT = "artifact"


@dataclass
class GatewayMessage:
    """Normalized message from any gateway (Telegram, Discord, web)."""
    platform: str
    user_id: str
    chat_id: str
    text: str
    message_type: MessageType = MessageType.TEXT
    metadata: dict = field(default_factory=dict)


@dataclass
class GatewayResponse:
    """Normalized response to send back through any gateway."""
    text: str
    notebook_id: Optional[str] = None
    match_level: Optional[str] = None
    created_new: bool = False
    metadata: dict = field(default_factory=dict)


def format_route_info(response: GatewayResponse) -> str:
    """Format routing metadata for display."""
    parts = []
    if response.notebook_id:
        parts.append(f"notebook: {response.notebook_id[:8]}...")
    if response.match_level:
        parts.append(f"match: {response.match_level}")
    if response.created_new:
        parts.append("NEW notebook created")
    return " | ".join(parts) if parts else ""
