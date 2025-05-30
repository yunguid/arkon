"""
OpenAI Realtime API Integration for Arkon Financial
Enables natural speech-to-speech conversations for financial assistance
"""

import asyncio
import json
import base64
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import websockets
import numpy as np
import pyaudio
from datetime import datetime

from backend.utils.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)


class RealtimeEventType(Enum):
    """OpenAI Realtime API event types"""
    # Session events
    SESSION_CREATE = "session.create"
    SESSION_UPDATE = "session.update"
    
    # Input events
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    
    # Response events
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    
    # Conversation events
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    
    # Server events
    ERROR = "error"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RATE_LIMITS_UPDATED = "rate_limits.updated"


@dataclass
class RealtimeConfig:
    """Configuration for OpenAI Realtime API"""
    model: str = "gpt-4o-realtime-preview"
    voice: str = "alloy"  # alloy, echo, shimmer
    instructions: str = """You are Arkon, a helpful and friendly AI financial assistant. 
    You help users manage their finances, track spending, set budgets, and make smart financial decisions.
    Be conversational, empathetic, and proactive in offering financial advice.
    Always prioritize the user's financial wellbeing and privacy."""
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: bool = True
    turn_detection: Dict[str, Any] = None
    tools: List[Dict[str, Any]] = None
    tool_choice: str = "auto"
    temperature: float = 0.7
    max_output_tokens: Optional[int] = None


class OpenAIRealtimeClient:
    """Client for OpenAI Realtime API"""
    
    def __init__(self, config: RealtimeConfig = None):
        self.config = config or RealtimeConfig()
        self.ws_url = "wss://api.openai.com/v1/realtime"
        self.websocket = None
        self.session_id = None
        self.is_connected = False
        self.event_handlers = {}
        self._setup_default_handlers()
        
        # Audio configuration
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 24000  # 24kHz for Realtime API
        self.chunk_size = 1024
        
        # Audio streams
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
        # Function tools for financial operations
        self.config.tools = self._get_financial_tools()
        
    def _setup_default_handlers(self):
        """Set up default event handlers"""
        self.on(RealtimeEventType.SESSION_CREATED, self._handle_session_created)
        self.on(RealtimeEventType.ERROR, self._handle_error)
        self.on(RealtimeEventType.RESPONSE_AUDIO_DELTA, self._handle_audio_delta)
        self.on(RealtimeEventType.RESPONSE_TEXT_DONE, self._handle_text_done)
        self.on(RealtimeEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE, self._handle_function_call)
        
    def _get_financial_tools(self) -> List[Dict[str, Any]]:
        """Define available function tools for financial operations"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "check_balance",
                    "description": "Check user's account balance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "account_type": {
                                "type": "string",
                                "enum": ["checking", "savings", "total"],
                                "description": "Type of account balance to check"
                            }
                        },
                        "required": ["account_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_spending_summary",
                    "description": "Get spending summary for a time period",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "period": {
                                "type": "string",
                                "enum": ["today", "week", "month", "year"],
                                "description": "Time period for spending summary"
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional spending category filter"
                            }
                        },
                        "required": ["period"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_budget",
                    "description": "Set a budget for a category",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Budget category"
                            },
                            "amount": {
                                "type": "number",
                                "description": "Monthly budget amount"
                            }
                        },
                        "required": ["category", "amount"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_transactions",
                    "description": "Search for specific transactions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (merchant name, amount, etc.)"
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)"
                            },
                            "date_to": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
    async def connect(self):
        """Connect to OpenAI Realtime API"""
        try:
            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers
            )
            
            self.is_connected = True
            logger.info("Connected to OpenAI Realtime API")
            
            # Create session with configuration
            await self._create_session()
            
            # Start listening for events
            asyncio.create_task(self._listen_for_events())
            
            # Start audio streams
            self._start_audio_streams()
            
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            raise
            
    async def _create_session(self):
        """Create a new session with configuration"""
        event = {
            "type": RealtimeEventType.SESSION_UPDATE.value,
            "session": {
                "model": self.config.model,
                "voice": self.config.voice,
                "instructions": self.config.instructions,
                "input_audio_format": self.config.input_audio_format,
                "output_audio_format": self.config.output_audio_format,
                "input_audio_transcription": {
                    "enabled": self.config.input_audio_transcription
                },
                "turn_detection": self.config.turn_detection or {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "suffix_padding_ms": 1000
                },
                "tools": self.config.tools,
                "tool_choice": self.config.tool_choice,
                "temperature": self.config.temperature
            }
        }
        
        if self.config.max_output_tokens:
            event["session"]["max_output_tokens"] = self.config.max_output_tokens
            
        await self._send_event(event)
        
    async def _send_event(self, event: Dict[str, Any]):
        """Send event to Realtime API"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
            
        await self.websocket.send(json.dumps(event))
        logger.debug(f"Sent event: {event['type']}")
        
    async def _listen_for_events(self):
        """Listen for events from Realtime API"""
        try:
            while self.is_connected and self.websocket:
                message = await self.websocket.recv()
                event = json.loads(message)
                
                event_type = event.get("type")
                logger.debug(f"Received event: {event_type}")
                
                # Call registered handlers
                await self._handle_event(event)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in event listener: {e}")
            self.is_connected = False
            
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle incoming event"""
        event_type = event.get("type")
        
        # Call specific handlers
        for registered_type, handlers in self.event_handlers.items():
            if registered_type.value == event_type:
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
                        
    def on(self, event_type: RealtimeEventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
    def _start_audio_streams(self):
        """Start audio input/output streams"""
        # Input stream (microphone)
        self.input_stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_input_callback
        )
        
        # Output stream (speaker)
        self.output_stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.input_stream.start_stream()
        logger.info("Audio streams started")
        
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        if self.is_connected:
            # Convert audio to base64 and send to API
            audio_base64 = base64.b64encode(in_data).decode('utf-8')
            
            asyncio.create_task(self._send_audio_chunk(audio_base64))
            
        return (in_data, pyaudio.paContinue)
        
    async def _send_audio_chunk(self, audio_base64: str):
        """Send audio chunk to Realtime API"""
        event = {
            "type": RealtimeEventType.INPUT_AUDIO_BUFFER_APPEND.value,
            "audio": audio_base64
        }
        await self._send_event(event)
        
    async def _handle_session_created(self, event: Dict[str, Any]):
        """Handle session created event"""
        self.session_id = event.get("session", {}).get("id")
        logger.info(f"Session created: {self.session_id}")
        
    async def _handle_error(self, event: Dict[str, Any]):
        """Handle error event"""
        error = event.get("error", {})
        logger.error(f"Realtime API error: {error}")
        
    async def _handle_audio_delta(self, event: Dict[str, Any]):
        """Handle audio delta event (streaming audio output)"""
        audio_base64 = event.get("delta", "")
        if audio_base64:
            # Decode and play audio
            audio_data = base64.b64decode(audio_base64)
            if self.output_stream:
                self.output_stream.write(audio_data)
                
    async def _handle_text_done(self, event: Dict[str, Any]):
        """Handle completed text response"""
        text = event.get("text", "")
        logger.info(f"Assistant response: {text}")
        
    async def _handle_function_call(self, event: Dict[str, Any]):
        """Handle function call from assistant"""
        function_name = event.get("name")
        arguments = json.loads(event.get("arguments", "{}"))
        
        logger.info(f"Function call: {function_name} with args: {arguments}")
        
        # Execute function and send result back
        result = await self._execute_function(function_name, arguments)
        
        # Send function result back to conversation
        await self._send_function_result(event.get("call_id"), result)
        
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute function call"""
        # This would integrate with your actual backend services
        if function_name == "check_balance":
            # Mock implementation
            return {
                "checking": 5432.10,
                "savings": 12000.00,
                "total": 17432.10
            }
        elif function_name == "get_spending_summary":
            # Mock implementation
            return {
                "period": arguments.get("period"),
                "total_spent": 2845.67,
                "top_categories": [
                    {"category": "Food", "amount": 523.45},
                    {"category": "Transport", "amount": 234.56},
                    {"category": "Shopping", "amount": 1234.56}
                ]
            }
        elif function_name == "set_budget":
            # Mock implementation
            return {
                "success": True,
                "message": f"Budget set for {arguments.get('category')}: ${arguments.get('amount')}/month"
            }
        elif function_name == "find_transactions":
            # Mock implementation
            return {
                "transactions": [
                    {
                        "date": "2024-01-15",
                        "merchant": "Amazon",
                        "amount": 45.67,
                        "category": "Shopping"
                    }
                ],
                "total_found": 1
            }
        else:
            return {"error": f"Unknown function: {function_name}"}
            
    async def _send_function_result(self, call_id: str, result: Any):
        """Send function execution result back to API"""
        event = {
            "type": RealtimeEventType.CONVERSATION_ITEM_CREATE.value,
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result)
            }
        }
        await self._send_event(event)
        
    async def send_text(self, text: str):
        """Send text message to assistant"""
        event = {
            "type": RealtimeEventType.CONVERSATION_ITEM_CREATE.value,
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        }
        await self._send_event(event)
        
        # Create response
        await self._create_response()
        
    async def _create_response(self):
        """Request a response from the assistant"""
        event = {
            "type": RealtimeEventType.RESPONSE_CREATE.value,
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Please respond naturally and helpfully."
            }
        }
        await self._send_event(event)
        
    async def disconnect(self):
        """Disconnect from Realtime API"""
        self.is_connected = False
        
        # Stop audio streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        self.audio.terminate()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            
        logger.info("Disconnected from OpenAI Realtime API")


class RealtimeVoiceAssistant:
    """High-level voice assistant using OpenAI Realtime API"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.client = None
        self.conversation_history = []
        
    async def start(self):
        """Start the voice assistant"""
        # Configure for financial assistant
        config = RealtimeConfig(
            voice="alloy",  # Professional voice
            instructions=f"""You are Arkon, a professional and helpful AI financial assistant for user {self.user_id}.
            Help them manage finances, track spending, create budgets, and make smart financial decisions.
            Be concise but thorough. Always confirm important actions before executing them.
            If unsure about something, ask for clarification.""",
            temperature=0.7
        )
        
        self.client = OpenAIRealtimeClient(config)
        
        # Register custom handlers
        self.client.on(RealtimeEventType.RESPONSE_TEXT_DONE, self._log_conversation)
        self.client.on(RealtimeEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED, self._on_speech_started)
        self.client.on(RealtimeEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED, self._on_speech_stopped)
        
        # Connect to API
        await self.client.connect()
        
        # Send welcome message
        await self.client.send_text(
            f"Hello! I'm Arkon, your AI financial assistant. How can I help you manage your finances today?"
        )
        
    async def _log_conversation(self, event: Dict[str, Any]):
        """Log conversation for analysis"""
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "type": "assistant",
            "content": event.get("text", "")
        })
        
    async def _on_speech_started(self, event: Dict[str, Any]):
        """Handle speech started event"""
        logger.info("User started speaking")
        
    async def _on_speech_stopped(self, event: Dict[str, Any]):
        """Handle speech stopped event"""
        logger.info("User stopped speaking")
        
    async def stop(self):
        """Stop the voice assistant"""
        if self.client:
            await self.client.disconnect()
            
    async def send_message(self, message: str):
        """Send a text message to the assistant"""
        if self.client and self.client.is_connected:
            await self.client.send_text(message)
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "type": "user",
                "content": message
            })


# Example usage
async def main():
    """Example of using the Realtime Voice Assistant"""
    assistant = RealtimeVoiceAssistant(user_id="demo_user")
    
    try:
        # Start the assistant
        await assistant.start()
        
        # Keep running
        await asyncio.sleep(60)  # Run for 60 seconds
        
    finally:
        # Clean up
        await assistant.stop()


if __name__ == "__main__":
    asyncio.run(main()) 