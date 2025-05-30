"""
Advanced Security and Authentication for Arkon
Multi-factor authentication, OAuth2, biometrics, and encryption
"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import struct
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jwt
import pyotp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import face_recognition
import cv2
import numpy as np
from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from sqlalchemy.orm import Session

from backend.models import User
from backend.utils.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)


class AuthenticationMethod(Enum):
    PASSWORD = "password"
    OAUTH2 = "oauth2"
    BIOMETRIC_FACE = "biometric_face"
    BIOMETRIC_FINGERPRINT = "biometric_fingerprint"
    HARDWARE_TOKEN = "hardware_token"
    SMS_OTP = "sms_otp"
    EMAIL_OTP = "email_otp"
    AUTHENTICATOR_APP = "authenticator_app"
    PASSKEY = "passkey"


class SecurityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AuthenticationToken:
    token: str
    token_type: str
    expires_at: datetime
    refresh_token: Optional[str]
    scope: List[str]
    user_id: int
    session_id: str
    device_id: str
    security_level: SecurityLevel


@dataclass
class BiometricData:
    type: str
    data: bytes
    template: Any
    quality_score: float
    captured_at: datetime
    device_info: Dict[str, Any]


@dataclass
class SecurityEvent:
    event_type: str
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any]
    risk_score: float


class AdvancedAuthManager:
    """Advanced authentication and security manager"""
    
    def __init__(self):
        self.password_hasher = PasswordHasher()
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.jwt_algorithm = "RS256"
        self.oauth = self._initialize_oauth()
        self.redis_client = None
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._load_rsa_keys()
        
    def _initialize_oauth(self) -> OAuth:
        """Initialize OAuth2 providers"""
        oauth = OAuth()
        
        # Google OAuth2
        oauth.register(
            name='google',
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'}
        )
        
        # GitHub OAuth2
        oauth.register(
            name='github',
            client_id=settings.GITHUB_CLIENT_ID,
            client_secret=settings.GITHUB_CLIENT_SECRET,
            access_token_url='https://github.com/login/oauth/access_token',
            access_token_params=None,
            authorize_url='https://github.com/login/oauth/authorize',
            authorize_params=None,
            api_base_url='https://api.github.com/',
            client_kwargs={'scope': 'user:email'}
        )
        
        # Microsoft OAuth2
        oauth.register(
            name='microsoft',
            client_id=settings.MICROSOFT_CLIENT_ID,
            client_secret=settings.MICROSOFT_CLIENT_SECRET,
            server_metadata_url='https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'}
        )
        
        return oauth
        
    def _load_rsa_keys(self):
        """Load or generate RSA keys for JWT signing"""
        private_key_path = Path("keys/private_key.pem")
        public_key_path = Path("keys/public_key.pem")
        
        if private_key_path.exists() and public_key_path.exists():
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            with open(public_key_path, "rb") as f:
                self.public_key = serialization.load_pem_public_key(
                    f.read(), backend=default_backend()
                )
        else:
            # Generate new keys
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.public_key = self.private_key.public_key()
            
            # Save keys
            private_key_path.parent.mkdir(exist_ok=True)
            
            with open(private_key_path, "wb") as f:
                f.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
                
            with open(public_key_path, "wb") as f:
                f.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
                
    async def authenticate_user(
        self,
        username: str,
        password: str,
        methods: List[AuthenticationMethod],
        device_id: str,
        ip_address: str,
        user_agent: str
    ) -> Tuple[bool, Optional[AuthenticationToken], List[str]]:
        """Multi-factor authentication"""
        try:
            # Get user from database
            user = await self._get_user_by_username(username)
            if not user:
                await self._log_security_event(
                    "login_failed",
                    None,
                    ip_address,
                    user_agent,
                    False,
                    {"reason": "user_not_found"}
                )
                return False, None, ["Invalid credentials"]
                
            # Check if account is locked
            if await self._is_account_locked(user.id):
                return False, None, ["Account is locked due to multiple failed attempts"]
                
            # Verify password
            if not await self._verify_password(password, user.password_hash):
                await self._handle_failed_login(user.id, ip_address)
                return False, None, ["Invalid credentials"]
                
            # Check required authentication methods
            required_methods = await self._get_required_auth_methods(user, ip_address, device_id)
            
            # Perform additional authentication
            mfa_results = []
            for method in required_methods:
                if method == AuthenticationMethod.AUTHENTICATOR_APP:
                    # This would be handled by a separate endpoint
                    mfa_results.append(("authenticator_required", False))
                elif method == AuthenticationMethod.SMS_OTP:
                    # Send SMS OTP
                    await self._send_sms_otp(user.phone_number)
                    mfa_results.append(("sms_otp_sent", False))
                elif method == AuthenticationMethod.EMAIL_OTP:
                    # Send email OTP
                    await self._send_email_otp(user.email)
                    mfa_results.append(("email_otp_sent", False))
                    
            # If MFA is required, return partial success
            if mfa_results:
                session_id = await self._create_mfa_session(user.id, mfa_results)
                return True, None, [f"mfa_required:{session_id}"]
                
            # Generate tokens
            token = await self._generate_authentication_token(
                user.id,
                device_id,
                SecurityLevel.MEDIUM
            )
            
            # Log successful login
            await self._log_security_event(
                "login_success",
                user.id,
                ip_address,
                user_agent,
                True,
                {"device_id": device_id}
            )
            
            return True, token, []
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, None, ["Authentication failed"]
            
    async def verify_mfa_code(
        self,
        session_id: str,
        method: AuthenticationMethod,
        code: str,
        device_id: str
    ) -> Tuple[bool, Optional[AuthenticationToken]]:
        """Verify MFA code"""
        try:
            # Get MFA session
            session_data = await self._get_mfa_session(session_id)
            if not session_data:
                return False, None
                
            user_id = session_data['user_id']
            
            # Verify code based on method
            if method == AuthenticationMethod.AUTHENTICATOR_APP:
                if not await self._verify_totp(user_id, code):
                    return False, None
            elif method == AuthenticationMethod.SMS_OTP:
                if not await self._verify_otp(user_id, code, "sms"):
                    return False, None
            elif method == AuthenticationMethod.EMAIL_OTP:
                if not await self._verify_otp(user_id, code, "email"):
                    return False, None
            else:
                return False, None
                
            # Check if all required methods are verified
            session_data['verified_methods'].append(method.value)
            
            if set(session_data['required_methods']) <= set(session_data['verified_methods']):
                # All methods verified, generate token
                token = await self._generate_authentication_token(
                    user_id,
                    device_id,
                    SecurityLevel.HIGH
                )
                
                # Clear MFA session
                await self._clear_mfa_session(session_id)
                
                return True, token
            else:
                # Update session
                await self._update_mfa_session(session_id, session_data)
                return True, None
                
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False, None
            
    async def setup_totp(self, user_id: int) -> Dict[str, str]:
        """Setup TOTP for authenticator apps"""
        try:
            # Generate secret
            secret = pyotp.random_base32()
            
            # Store encrypted secret
            encrypted_secret = self.cipher_suite.encrypt(secret.encode())
            await self._store_user_secret(user_id, "totp_secret", encrypted_secret)
            
            # Generate provisioning URI
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=f"user_{user_id}",
                issuer_name="Arkon Financial"
            )
            
            # Generate QR code
            import qrcode
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            qr_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                'secret': secret,
                'qr_code': f"data:image/png;base64,{qr_base64}",
                'provisioning_uri': provisioning_uri
            }
            
        except Exception as e:
            logger.error(f"TOTP setup error: {e}")
            raise
            
    async def register_biometric_face(
        self,
        user_id: int,
        face_image: bytes
    ) -> Tuple[bool, str]:
        """Register face biometric data"""
        try:
            # Decode image
            nparr = np.frombuffer(face_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_img)
            
            if len(face_locations) != 1:
                return False, "Please ensure exactly one face is visible"
                
            # Generate face encoding
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            if not face_encodings:
                return False, "Could not generate face encoding"
                
            # Encrypt and store face encoding
            face_data = {
                'encoding': face_encodings[0].tolist(),
                'quality_score': self._calculate_face_quality(img),
                'registered_at': datetime.now().isoformat()
            }
            
            encrypted_data = self.cipher_suite.encrypt(
                json.dumps(face_data).encode()
            )
            
            await self._store_biometric_data(
                user_id,
                "face",
                encrypted_data
            )
            
            return True, "Face biometric registered successfully"
            
        except Exception as e:
            logger.error(f"Face registration error: {e}")
            return False, "Failed to register face biometric"
            
    async def authenticate_biometric_face(
        self,
        user_id: int,
        face_image: bytes
    ) -> Tuple[bool, float]:
        """Authenticate using face biometric"""
        try:
            # Get stored face data
            encrypted_data = await self._get_biometric_data(user_id, "face")
            if not encrypted_data:
                return False, 0.0
                
            # Decrypt face data
            face_data = json.loads(
                self.cipher_suite.decrypt(encrypted_data).decode()
            )
            stored_encoding = np.array(face_data['encoding'])
            
            # Process new image
            nparr = np.frombuffer(face_image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                return False, 0.0
                
            # Generate face encoding
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            if not face_encodings:
                return False, 0.0
                
            # Compare faces
            matches = face_recognition.compare_faces(
                [stored_encoding],
                face_encodings[0],
                tolerance=0.6
            )
            
            if matches[0]:
                # Calculate distance for confidence score
                face_distance = face_recognition.face_distance(
                    [stored_encoding],
                    face_encodings[0]
                )[0]
                
                confidence = 1.0 - face_distance
                
                # Additional liveness detection
                if await self._check_liveness(img):
                    return True, confidence
                else:
                    return False, 0.0
            else:
                return False, 0.0
                
        except Exception as e:
            logger.error(f"Face authentication error: {e}")
            return False, 0.0
            
    async def _check_liveness(self, image: np.ndarray) -> bool:
        """Check if face is live (not a photo)"""
        try:
            # Simple liveness check based on:
            # 1. Eye blink detection
            # 2. Face texture analysis
            # 3. Motion detection (would require multiple frames)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Check texture using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Higher variance indicates real face (not printed photo)
            if variance < 100:
                return False
                
            # If eyes detected and good texture, consider live
            return len(eyes) >= 2
            
        except Exception as e:
            logger.error(f"Liveness check error: {e}")
            return False
            
    async def encrypt_sensitive_data(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data"""
        try:
            if isinstance(data, str):
                data = data.encode()
                
            # Use AES-GCM for authenticated encryption
            key = secrets.token_bytes(32)
            nonce = secrets.token_bytes(12)
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Combine nonce, ciphertext, and tag
            encrypted = nonce + ciphertext + encryptor.tag
            
            # Encrypt the AES key with RSA
            encrypted_key = self.public_key.encrypt(
                key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key and data
            final_data = encrypted_key + encrypted
            
            return base64.b64encode(final_data).decode()
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
            
    async def decrypt_sensitive_data(self, encrypted_data: str) -> bytes:
        """Decrypt sensitive data"""
        try:
            data = base64.b64decode(encrypted_data)
            
            # Extract encrypted key (first 256 bytes for 2048-bit RSA)
            encrypted_key = data[:256]
            encrypted_content = data[256:]
            
            # Decrypt the AES key
            key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Extract nonce, ciphertext, and tag
            nonce = encrypted_content[:12]
            tag = encrypted_content[-16:]
            ciphertext = encrypted_content[12:-16]
            
            # Decrypt data
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(ciphertext) + decryptor.finalize()
            
            return decrypted
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
            
    async def _generate_authentication_token(
        self,
        user_id: int,
        device_id: str,
        security_level: SecurityLevel
    ) -> AuthenticationToken:
        """Generate JWT authentication token"""
        try:
            session_id = secrets.token_urlsafe(32)
            now = datetime.utcnow()
            
            # Access token payload
            access_payload = {
                'user_id': user_id,
                'session_id': session_id,
                'device_id': device_id,
                'security_level': security_level.value,
                'exp': now + timedelta(minutes=15),
                'iat': now,
                'type': 'access'
            }
            
            # Refresh token payload
            refresh_payload = {
                'user_id': user_id,
                'session_id': session_id,
                'exp': now + timedelta(days=30),
                'iat': now,
                'type': 'refresh'
            }
            
            # Sign tokens with RSA private key
            access_token = jwt.encode(
                access_payload,
                self.private_key,
                algorithm=self.jwt_algorithm
            )
            
            refresh_token = jwt.encode(
                refresh_payload,
                self.private_key,
                algorithm=self.jwt_algorithm
            )
            
            # Store session in Redis
            await self._store_session(session_id, {
                'user_id': user_id,
                'device_id': device_id,
                'security_level': security_level.value,
                'created_at': now.isoformat(),
                'last_accessed': now.isoformat()
            })
            
            return AuthenticationToken(
                token=access_token,
                token_type="Bearer",
                expires_at=access_payload['exp'],
                refresh_token=refresh_token,
                scope=['read', 'write'],
                user_id=user_id,
                session_id=session_id,
                device_id=device_id,
                security_level=security_level
            )
            
        except Exception as e:
            logger.error(f"Token generation error: {e}")
            raise
            
    def _calculate_face_quality(self, image: np.ndarray) -> float:
        """Calculate face image quality score"""
        try:
            # Factors: brightness, contrast, sharpness, face size
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 127) / 127
            
            # Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 60, 1.0)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(sharpness / 500, 1.0)
            
            # Overall quality
            quality = (brightness_score + contrast_score + sharpness_score) / 3
            
            return quality
            
        except Exception as e:
            logger.error(f"Quality calculation error: {e}")
            return 0.5


class SecurityMonitor:
    """Monitor and respond to security threats"""
    
    def __init__(self):
        self.threat_threshold = 0.7
        self.anomaly_detector = AnomalyDetector()
        
    async def analyze_login_attempt(
        self,
        user_id: Optional[int],
        ip_address: str,
        user_agent: str,
        location: Optional[Dict[str, Any]]
    ) -> float:
        """Analyze login attempt and return risk score"""
        risk_score = 0.0
        
        # Check IP reputation
        ip_risk = await self._check_ip_reputation(ip_address)
        risk_score += ip_risk * 0.3
        
        # Check for impossible travel
        if user_id and location:
            travel_risk = await self._check_impossible_travel(user_id, location)
            risk_score += travel_risk * 0.3
            
        # Check device fingerprint
        device_risk = await self._analyze_device(user_agent)
        risk_score += device_risk * 0.2
        
        # Check login patterns
        if user_id:
            pattern_risk = await self._analyze_login_patterns(user_id)
            risk_score += pattern_risk * 0.2
            
        return min(risk_score, 1.0)
        
    async def _check_ip_reputation(self, ip_address: str) -> float:
        """Check IP address reputation"""
        # In production, this would check against threat intelligence feeds
        # For now, simple checks
        
        # Check if IP is from known VPN/proxy
        vpn_ranges = [
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16"
        ]
        
        # Check against blacklists
        # Return risk score 0-1
        return 0.1  # Low risk for demo
        
    async def _check_impossible_travel(
        self,
        user_id: int,
        current_location: Dict[str, Any]
    ) -> float:
        """Check for impossible travel scenarios"""
        # Get last known location
        last_location = await self._get_last_login_location(user_id)
        
        if not last_location:
            return 0.0
            
        # Calculate distance and time
        distance = self._calculate_distance(last_location, current_location)
        time_diff = (datetime.now() - last_location['timestamp']).total_seconds() / 3600
        
        # Check if travel is possible
        max_speed = 1000  # km/h (airplane speed)
        
        if time_diff > 0:
            required_speed = distance / time_diff
            if required_speed > max_speed:
                return 0.9  # High risk
                
        return 0.0
        
    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate distance between two locations in km"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = radians(loc1['lat']), radians(loc1['lon'])
        lat2, lon2 = radians(loc2['lat']), radians(loc2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c


class AnomalyDetector:
    """Detect anomalies in user behavior"""
    
    async def detect_transaction_anomaly(
        self,
        user_id: int,
        transaction: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """Detect anomalous transactions"""
        # Implement ML-based anomaly detection
        # For now, rule-based detection
        
        anomaly_score = 0.0
        reasons = []
        
        # Check amount
        avg_transaction = await self._get_average_transaction(user_id)
        if transaction['amount'] > avg_transaction * 5:
            anomaly_score += 0.4
            reasons.append("Unusually high amount")
            
        # Check merchant
        if await self._is_new_merchant(user_id, transaction['merchant']):
            anomaly_score += 0.2
            reasons.append("New merchant")
            
        # Check time
        if self._is_unusual_time(transaction['timestamp']):
            anomaly_score += 0.2
            reasons.append("Unusual time")
            
        # Check location
        if await self._is_unusual_location(user_id, transaction.get('location')):
            anomaly_score += 0.3
            reasons.append("Unusual location")
            
        is_anomaly = anomaly_score > 0.6
        
        return is_anomaly, anomaly_score, "; ".join(reasons)
        
    def _is_unusual_time(self, timestamp: datetime) -> bool:
        """Check if transaction time is unusual"""
        hour = timestamp.hour
        # Consider 2 AM - 5 AM as unusual
        return 2 <= hour <= 5 