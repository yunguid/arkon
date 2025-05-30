"""
Google OAuth Authentication for Arkon Financial
Secure user authentication with Google Sign-In
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
import secrets
from typing import Optional, Dict, Any

from backend.database import get_db
from backend.models import User
from backend.utils.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)

# Create auth router
auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Security
security = HTTPBearer()


class GoogleAuthService:
    """Handle Google OAuth authentication"""
    
    def __init__(self):
        self.client_id = settings.GOOGLE_CLIENT_ID
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 24
        
    async def verify_google_token(self, token: str) -> Dict[str, Any]:
        """Verify Google ID token and extract user info"""
        try:
            # Verify the token with Google
            idinfo = id_token.verify_oauth2_token(
                token, 
                requests.Request(), 
                self.client_id
            )
            
            # Token is valid, extract user info
            return {
                'google_id': idinfo['sub'],
                'email': idinfo['email'],
                'name': idinfo.get('name', ''),
                'picture': idinfo.get('picture', ''),
                'email_verified': idinfo.get('email_verified', False)
            }
            
        except ValueError as e:
            logger.error(f"Invalid Google token: {e}")
            raise HTTPException(status_code=401, detail="Invalid authentication token")
            
    def create_jwt_token(self, user_id: str, email: str) -> str:
        """Create JWT token for authenticated user"""
        payload = {
            'user_id': user_id,
            'email': email,
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    async def get_or_create_user(
        self, 
        google_user_info: Dict[str, Any], 
        db: Session
    ) -> User:
        """Get existing user or create new one from Google info"""
        # Check if user exists
        user = db.query(User).filter(
            User.email == google_user_info['email']
        ).first()
        
        if not user:
            # Create new user
            user = User(
                id=f"google_{google_user_info['google_id']}",
                email=google_user_info['email'],
                full_name=google_user_info['name'],
                profile_picture=google_user_info['picture'],
                auth_provider='google',
                is_verified=google_user_info['email_verified'],
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            
            # Set up default preferences
            user.preferences = {
                'theme': 'light',
                'notifications': True,
                'currency': 'USD',
                'language': 'en',
                'voice_enabled': True
            }
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            logger.info(f"Created new user: {user.email}")
            
            # Trigger onboarding flow
            await self._trigger_onboarding(user)
        else:
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            
        return user
        
    async def _trigger_onboarding(self, user: User):
        """Trigger onboarding flow for new users"""
        # This would send welcome email, set up tutorial, etc.
        logger.info(f"Triggering onboarding for user: {user.email}")
        # TODO: Implement onboarding logic


# Initialize service
google_auth_service = GoogleAuthService()


# API Endpoints
@auth_router.post("/google")
async def google_login(
    request: Dict[str, str],
    db: Session = Depends(get_db)
):
    """Authenticate user with Google OAuth"""
    token = request.get('token')
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")
        
    try:
        # Verify Google token
        google_user_info = await google_auth_service.verify_google_token(token)
        
        # Get or create user
        user = await google_auth_service.get_or_create_user(google_user_info, db)
        
        # Create JWT token
        jwt_token = google_auth_service.create_jwt_token(user.id, user.email)
        
        # Return user info and token
        return {
            "token": jwt_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.full_name,
                "picture": user.profile_picture,
                "preferences": user.preferences,
                "is_new": user.created_at == user.last_login
            }
        }
        
    except Exception as e:
        logger.error(f"Google login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@auth_router.get("/verify")
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Verify JWT token and return user info"""
    try:
        # Extract token
        token = credentials.credentials
        
        # Verify token
        payload = google_auth_service.verify_jwt_token(token)
        
        # Get user
        user = db.query(User).filter(User.id == payload['user_id']).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        return {
            "valid": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.full_name,
                "picture": user.profile_picture
            }
        }
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


@auth_router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Logout user (invalidate token)"""
    try:
        # In a production system, you would:
        # 1. Add token to a blacklist
        # 2. Clear any server-side sessions
        # 3. Log the logout event
        
        token = credentials.credentials
        payload = google_auth_service.verify_jwt_token(token)
        
        logger.info(f"User {payload['email']} logged out")
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")


@auth_router.post("/refresh")
async def refresh_token(
    request: Dict[str, str],
    db: Session = Depends(get_db)
):
    """Refresh JWT token"""
    try:
        old_token = request.get('token')
        if not old_token:
            raise HTTPException(status_code=400, detail="Token is required")
            
        # Verify old token (even if expired)
        try:
            payload = jwt.decode(
                old_token, 
                google_auth_service.jwt_secret, 
                algorithms=[google_auth_service.jwt_algorithm],
                options={"verify_exp": False}  # Don't verify expiration
            )
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        # Check if token is not too old (e.g., within 7 days)
        token_age = datetime.utcnow() - datetime.fromtimestamp(payload['iat'])
        if token_age.days > 7:
            raise HTTPException(status_code=401, detail="Token too old, please login again")
            
        # Get user
        user = db.query(User).filter(User.id == payload['user_id']).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        # Create new token
        new_token = google_auth_service.create_jwt_token(user.id, user.email)
        
        return {
            "token": new_token,
            "expires_in": google_auth_service.jwt_expiration_hours * 3600
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


# Dependency to get current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        payload = google_auth_service.verify_jwt_token(token)
        
        user = db.query(User).filter(User.id == payload['user_id']).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        return user
        
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise HTTPException(status_code=401, detail="Authentication required") 