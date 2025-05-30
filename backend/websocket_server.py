"""
Real-time WebSocket Server for Financial Analyzer
Provides live updates, notifications, and collaborative features
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any, List
from fastapi import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import redis.asyncio as redis
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

@dataclass
class WebSocketMessage:
    """Structured WebSocket message"""
    type: str
    data: Any
    timestamp: datetime
    sender_id: Optional[str] = None
    room_id: Optional[str] = None

class ConnectionManager:
    """Manages WebSocket connections and messaging"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_rooms: Dict[str, Set[str]] = {}
        self.room_users: Dict[str, Set[str]] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis for distributed messaging"""
        try:
            self.redis_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe("financial_updates")
            logger.info("Redis initialized for WebSocket messaging")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Send welcome message
        await self.send_personal_message(
            WebSocketMessage(
                type="connection",
                data={"status": "connected", "client_id": client_id},
                timestamp=datetime.now()
            ),
            client_id
        )
        
        # Broadcast user joined
        await self.broadcast(
            WebSocketMessage(
                type="user_joined",
                data={"user_id": client_id},
                timestamp=datetime.now(),
                sender_id=client_id
            ),
            exclude=[client_id]
        )
    
    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # Leave all rooms
        if client_id in self.user_rooms:
            for room_id in self.user_rooms[client_id]:
                if room_id in self.room_users:
                    self.room_users[room_id].discard(client_id)
            del self.user_rooms[client_id]
    
    async def join_room(self, client_id: str, room_id: str) -> None:
        """Join a collaborative room"""
        if client_id not in self.user_rooms:
            self.user_rooms[client_id] = set()
        self.user_rooms[client_id].add(room_id)
        
        if room_id not in self.room_users:
            self.room_users[room_id] = set()
        self.room_users[room_id].add(client_id)
        
        # Notify room members
        await self.send_to_room(
            WebSocketMessage(
                type="user_joined_room",
                data={"user_id": client_id, "room_id": room_id},
                timestamp=datetime.now(),
                room_id=room_id
            ),
            room_id
        )
    
    async def leave_room(self, client_id: str, room_id: str) -> None:
        """Leave a collaborative room"""
        if client_id in self.user_rooms:
            self.user_rooms[client_id].discard(room_id)
        
        if room_id in self.room_users:
            self.room_users[room_id].discard(client_id)
            
        # Notify room members
        await self.send_to_room(
            WebSocketMessage(
                type="user_left_room",
                data={"user_id": client_id, "room_id": room_id},
                timestamp=datetime.now(),
                room_id=room_id
            ),
            room_id
        )
    
    async def send_personal_message(self, message: WebSocketMessage, client_id: str) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(self._serialize_message(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def send_to_room(self, message: WebSocketMessage, room_id: str) -> None:
        """Send message to all clients in a room"""
        if room_id in self.room_users:
            tasks = []
            for client_id in self.room_users[room_id]:
                if client_id in self.active_connections:
                    tasks.append(self.send_personal_message(message, client_id))
            await asyncio.gather(*tasks)
    
    async def broadcast(self, message: WebSocketMessage, exclude: List[str] = None) -> None:
        """Broadcast message to all connected clients"""
        exclude = exclude or []
        tasks = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                tasks.append(self.send_personal_message(message, client_id))
        
        await asyncio.gather(*tasks)
        
        # Also publish to Redis for distributed systems
        if self.redis_client:
            await self.redis_client.publish(
                "financial_updates",
                json.dumps(self._serialize_message(message))
            )
    
    def _serialize_message(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Serialize message for transmission"""
        data = asdict(message)
        data['timestamp'] = data['timestamp'].isoformat()
        return data
    
    async def handle_redis_messages(self):
        """Handle messages from Redis pubsub"""
        if not self.pubsub:
            return
            
        async for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    # Broadcast to all local connections
                    await self.broadcast(
                        WebSocketMessage(
                            type=data['type'],
                            data=data['data'],
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            sender_id=data.get('sender_id'),
                            room_id=data.get('room_id')
                        )
                    )
                except Exception as e:
                    logger.error(f"Error handling Redis message: {e}")

class RealtimeAnalytics:
    """Real-time analytics engine"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def track_event(self, user_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Track user analytics event"""
        event = {
            "user_id": user_id,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update session
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = {
                "start_time": datetime.now(),
                "events": []
            }
        self.active_sessions[user_id]["events"].append(event)
        
        # Broadcast analytics update
        await self.connection_manager.broadcast(
            WebSocketMessage(
                type="analytics_event",
                data=event,
                timestamp=datetime.now(),
                sender_id=user_id
            )
        )
    
    async def get_live_stats(self) -> Dict[str, Any]:
        """Get real-time statistics"""
        active_users = len(self.connection_manager.active_connections)
        total_events = sum(len(session["events"]) for session in self.active_sessions.values())
        
        # Event breakdown
        event_types = {}
        for session in self.active_sessions.values():
            for event in session["events"]:
                event_type = event["event_type"]
                event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "active_users": active_users,
            "total_events": total_events,
            "event_breakdown": event_types,
            "timestamp": datetime.now().isoformat()
        }

class NotificationService:
    """Real-time notification service"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        
    async def send_notification(self, user_id: str, notification: Dict[str, Any]) -> None:
        """Send notification to user"""
        notification_id = str(uuid.uuid4())
        
        message = WebSocketMessage(
            type="notification",
            data={
                "id": notification_id,
                "title": notification.get("title", "Notification"),
                "message": notification.get("message", ""),
                "severity": notification.get("severity", "info"),
                "actions": notification.get("actions", []),
                "metadata": notification.get("metadata", {})
            },
            timestamp=datetime.now()
        )
        
        await self.connection_manager.send_personal_message(message, user_id)
        
        # Store notification for persistence
        await self.notification_queue.put({
            "id": notification_id,
            "user_id": user_id,
            "notification": notification,
            "timestamp": datetime.now()
        })
    
    async def send_budget_alert(self, user_id: str, category: str, spent: float, limit: float) -> None:
        """Send budget exceeded alert"""
        percentage = (spent / limit) * 100
        
        await self.send_notification(user_id, {
            "title": "Budget Alert! 💸",
            "message": f"You've spent ${spent:.2f} of your ${limit:.2f} budget for {category} ({percentage:.0f}%)",
            "severity": "warning" if percentage < 100 else "error",
            "actions": [
                {"label": "View Details", "action": "view_budget", "data": {"category": category}},
                {"label": "Adjust Budget", "action": "edit_budget", "data": {"category": category}}
            ],
            "metadata": {
                "category": category,
                "spent": spent,
                "limit": limit,
                "percentage": percentage
            }
        })
    
    async def send_anomaly_alert(self, user_id: str, transaction: Dict[str, Any]) -> None:
        """Send anomaly detection alert"""
        await self.send_notification(user_id, {
            "title": "Unusual Transaction Detected! 🚨",
            "message": f"Transaction of ${transaction['amount']:.2f} at {transaction['description']} seems unusual",
            "severity": "warning",
            "actions": [
                {"label": "Mark as Normal", "action": "mark_normal", "data": {"transaction_id": transaction.get('id')}},
                {"label": "Report Fraud", "action": "report_fraud", "data": {"transaction_id": transaction.get('id')}}
            ],
            "metadata": transaction
        })
    
    async def send_insight(self, user_id: str, insight: Dict[str, Any]) -> None:
        """Send financial insight"""
        await self.send_notification(user_id, {
            "title": f"Financial Insight 💡",
            "message": insight.get("message", ""),
            "severity": "info",
            "actions": insight.get("actions", []),
            "metadata": insight.get("metadata", {})
        })

class CollaborationHub:
    """Real-time collaboration features"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.shared_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def create_shared_session(self, creator_id: str, session_type: str) -> str:
        """Create a new shared session"""
        session_id = str(uuid.uuid4())
        
        self.shared_sessions[session_id] = {
            "id": session_id,
            "type": session_type,
            "creator": creator_id,
            "participants": [creator_id],
            "data": {},
            "created_at": datetime.now(),
            "last_updated": datetime.now()
        }
        
        # Join the room
        await self.connection_manager.join_room(creator_id, session_id)
        
        return session_id
    
    async def update_shared_data(self, session_id: str, user_id: str, data: Dict[str, Any]) -> None:
        """Update shared session data"""
        if session_id not in self.shared_sessions:
            raise ValueError("Session not found")
            
        session = self.shared_sessions[session_id]
        session["data"].update(data)
        session["last_updated"] = datetime.now()
        
        # Broadcast update to room
        await self.connection_manager.send_to_room(
            WebSocketMessage(
                type="shared_data_update",
                data={
                    "session_id": session_id,
                    "updated_by": user_id,
                    "data": data
                },
                timestamp=datetime.now(),
                sender_id=user_id,
                room_id=session_id
            ),
            session_id
        )
    
    async def add_participant(self, session_id: str, user_id: str) -> None:
        """Add participant to shared session"""
        if session_id not in self.shared_sessions:
            raise ValueError("Session not found")
            
        session = self.shared_sessions[session_id]
        if user_id not in session["participants"]:
            session["participants"].append(user_id)
            
        await self.connection_manager.join_room(user_id, session_id)
    
    async def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get shared session data"""
        if session_id not in self.shared_sessions:
            raise ValueError("Session not found")
            
        return self.shared_sessions[session_id]

# WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket, client_id: str, manager: ConnectionManager):
    """Main WebSocket endpoint handler"""
    await manager.connect(websocket, client_id)
    
    # Initialize services
    analytics = RealtimeAnalytics(manager)
    notifications = NotificationService(manager)
    collaboration = CollaborationHub(manager)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            payload = data.get("data", {})
            
            # Handle different message types
            if message_type == "ping":
                await manager.send_personal_message(
                    WebSocketMessage(
                        type="pong",
                        data={"timestamp": datetime.now().isoformat()},
                        timestamp=datetime.now()
                    ),
                    client_id
                )
                
            elif message_type == "track_event":
                await analytics.track_event(
                    client_id,
                    payload.get("event_type"),
                    payload.get("event_data", {})
                )
                
            elif message_type == "join_room":
                room_id = payload.get("room_id")
                if room_id:
                    await manager.join_room(client_id, room_id)
                    
            elif message_type == "leave_room":
                room_id = payload.get("room_id")
                if room_id:
                    await manager.leave_room(client_id, room_id)
                    
            elif message_type == "create_session":
                session_id = await collaboration.create_shared_session(
                    client_id,
                    payload.get("session_type", "analysis")
                )
                await manager.send_personal_message(
                    WebSocketMessage(
                        type="session_created",
                        data={"session_id": session_id},
                        timestamp=datetime.now()
                    ),
                    client_id
                )
                
            elif message_type == "update_shared_data":
                await collaboration.update_shared_data(
                    payload.get("session_id"),
                    client_id,
                    payload.get("data", {})
                )
                
            elif message_type == "request_notification":
                # Send test notification
                await notifications.send_insight(client_id, {
                    "message": "WebSocket connection is working perfectly! 🎉",
                    "metadata": {"test": True}
                })
                
            elif message_type == "get_live_stats":
                stats = await analytics.get_live_stats()
                await manager.send_personal_message(
                    WebSocketMessage(
                        type="live_stats",
                        data=stats,
                        timestamp=datetime.now()
                    ),
                    client_id
                )
                
            else:
                # Echo unknown messages for debugging
                await manager.send_personal_message(
                    WebSocketMessage(
                        type="error",
                        data={"message": f"Unknown message type: {message_type}"},
                        timestamp=datetime.now()
                    ),
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(
            WebSocketMessage(
                type="user_left",
                data={"user_id": client_id},
                timestamp=datetime.now()
            ),
            exclude=[client_id]
        )
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

# Initialize global connection manager
connection_manager = ConnectionManager() 