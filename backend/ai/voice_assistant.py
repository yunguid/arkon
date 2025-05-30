"""
Voice Assistant and Natural Language Processing for Arkon
Advanced AI-powered voice interface for financial management
"""

import asyncio
import json
import wave
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

import speech_recognition as sr
import pyttsx3
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification
)
import spacy
import nltk
from rasa.core.agent import Agent
from rasa.core.interpreter import RasaNLUInterpreter
import openai
import numpy as np
from scipy.io import wavfile
import torch
import whisper

from backend.models import User, Transaction, Budget, FinancialDocument
from backend.services.ml_engine import MLEngine
from backend.services.budget_manager import BudgetManager
from backend.utils.logger import get_logger
from backend.config import settings

logger = get_logger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('named_entity_chunks')
nltk.download('maxent_ne_chunker')
nltk.download('words')


class IntentType(Enum):
    CHECK_BALANCE = "check_balance"
    SPENDING_ANALYSIS = "spending_analysis"
    BUDGET_STATUS = "budget_status"
    SET_BUDGET = "set_budget"
    TRANSACTION_QUERY = "transaction_query"
    FINANCIAL_ADVICE = "financial_advice"
    INVESTMENT_OPPORTUNITY = "investment_opportunity"
    BILL_REMINDER = "bill_reminder"
    SAVINGS_GOAL = "savings_goal"
    ANOMALY_EXPLANATION = "anomaly_explanation"
    MARKET_UPDATE = "market_update"
    TAX_QUESTION = "tax_question"
    GENERAL_HELP = "general_help"


@dataclass
class VoiceCommand:
    text: str
    intent: IntentType
    entities: Dict[str, Any]
    confidence: float
    timestamp: datetime
    user_id: int
    audio_features: Optional[Dict[str, float]] = None


@dataclass
class ConversationContext:
    user_id: int
    session_id: str
    history: List[Dict[str, Any]]
    current_topic: Optional[str]
    user_preferences: Dict[str, Any]
    last_interaction: datetime


class VoiceAssistant:
    """Advanced voice assistant for financial management"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = self._initialize_tts()
        self.nlp = self._initialize_nlp()
        self.ml_engine = MLEngine()
        self.budget_manager = BudgetManager()
        self.conversation_manager = ConversationManager()
        self.intent_classifier = self._load_intent_classifier()
        self.entity_extractor = self._load_entity_extractor()
        self.whisper_model = whisper.load_model("base")
        self.financial_qa_model = self._load_financial_qa_model()
        
    def _initialize_tts(self) -> pyttsx3.Engine:
        """Initialize text-to-speech engine"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Set voice based on user preference
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)  # Female voice
            
        return engine
        
    def _initialize_nlp(self) -> spacy.Language:
        """Initialize spaCy NLP model"""
        try:
            nlp = spacy.load("en_core_web_lg")
            # Add custom financial entity recognition
            nlp.add_pipe("financial_ner", last=True)
            return nlp
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return spacy.blank("en")
            
    def _load_intent_classifier(self):
        """Load fine-tuned intent classification model"""
        try:
            model_name = "finbert-tone"  # Financial BERT model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Failed to load intent classifier: {e}")
            return None
            
    def _load_entity_extractor(self):
        """Load NER model for entity extraction"""
        try:
            return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        except Exception as e:
            logger.error(f"Failed to load entity extractor: {e}")
            return None
            
    def _load_financial_qa_model(self):
        """Load question-answering model for financial queries"""
        try:
            return pipeline("question-answering", model="deepset/roberta-base-squad2")
        except Exception as e:
            logger.error(f"Failed to load QA model: {e}")
            return None
            
    async def listen(self, source: str = "microphone") -> Optional[VoiceCommand]:
        """Listen for voice commands"""
        try:
            if source == "microphone":
                with sr.Microphone() as mic:
                    logger.info("Listening for command...")
                    self.recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                    audio = self.recognizer.listen(mic, timeout=5, phrase_time_limit=10)
                    
            # Convert to text using multiple engines for accuracy
            text = await self._transcribe_audio(audio)
            
            if not text:
                return None
                
            # Process the command
            command = await self.process_command(text, audio)
            return command
            
        except sr.WaitTimeoutError:
            logger.info("No speech detected")
            return None
        except Exception as e:
            logger.error(f"Error in voice recognition: {e}")
            return None
            
    async def _transcribe_audio(self, audio: sr.AudioData) -> Optional[str]:
        """Transcribe audio using multiple engines"""
        transcriptions = []
        
        # Try Google Speech Recognition
        try:
            text = self.recognizer.recognize_google(audio)
            transcriptions.append((text, 0.8))
        except Exception as e:
            logger.error(f"Google transcription failed: {e}")
            
        # Try Whisper
        try:
            # Convert audio to format Whisper expects
            wav_data = io.BytesIO(audio.get_wav_data())
            audio_array = np.frombuffer(wav_data.read(), dtype=np.int16)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_array)
            transcriptions.append((result["text"], 0.9))
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            
        # Choose best transcription
        if transcriptions:
            # Sort by confidence and return highest
            transcriptions.sort(key=lambda x: x[1], reverse=True)
            return transcriptions[0][0]
            
        return None
        
    async def process_command(
        self,
        text: str,
        audio: Optional[sr.AudioData] = None,
        user_id: Optional[int] = None
    ) -> VoiceCommand:
        """Process voice command and extract intent/entities"""
        try:
            # Extract audio features if available
            audio_features = None
            if audio:
                audio_features = await self._extract_audio_features(audio)
                
            # Classify intent
            intent, confidence = await self._classify_intent(text)
            
            # Extract entities
            entities = await self._extract_entities(text, intent)
            
            # Get user context
            if user_id:
                context = await self.conversation_manager.get_context(user_id)
                
                # Enhance understanding with context
                intent, entities = await self._enhance_with_context(
                    text, intent, entities, context
                )
            
            command = VoiceCommand(
                text=text,
                intent=intent,
                entities=entities,
                confidence=confidence,
                timestamp=datetime.now(),
                user_id=user_id or 0,
                audio_features=audio_features
            )
            
            # Log command for improvement
            await self._log_command(command)
            
            return command
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return VoiceCommand(
                text=text,
                intent=IntentType.GENERAL_HELP,
                entities={},
                confidence=0.5,
                timestamp=datetime.now(),
                user_id=user_id or 0
            )
            
    async def _classify_intent(self, text: str) -> Tuple[IntentType, float]:
        """Classify user intent from text"""
        try:
            # Use custom intent classifier
            if self.intent_classifier:
                results = self.intent_classifier(text)
                
                # Map to our intent types
                intent_mapping = {
                    'balance': IntentType.CHECK_BALANCE,
                    'spending': IntentType.SPENDING_ANALYSIS,
                    'budget': IntentType.BUDGET_STATUS,
                    'transaction': IntentType.TRANSACTION_QUERY,
                    'advice': IntentType.FINANCIAL_ADVICE,
                    'investment': IntentType.INVESTMENT_OPPORTUNITY,
                    'savings': IntentType.SAVINGS_GOAL,
                    'anomaly': IntentType.ANOMALY_EXPLANATION,
                    'market': IntentType.MARKET_UPDATE,
                    'tax': IntentType.TAX_QUESTION
                }
                
                # Find best matching intent
                for result in results:
                    for key, intent_type in intent_mapping.items():
                        if key in result['label'].lower():
                            return intent_type, result['score']
            
            # Fallback: Rule-based classification
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['balance', 'how much', 'account']):
                return IntentType.CHECK_BALANCE, 0.8
            elif any(word in text_lower for word in ['spent', 'spending', 'expenses']):
                return IntentType.SPENDING_ANALYSIS, 0.8
            elif any(word in text_lower for word in ['budget', 'limit', 'overspent']):
                return IntentType.BUDGET_STATUS, 0.8
            elif any(word in text_lower for word in ['transaction', 'purchase', 'payment']):
                return IntentType.TRANSACTION_QUERY, 0.8
            elif any(word in text_lower for word in ['advice', 'suggest', 'should i']):
                return IntentType.FINANCIAL_ADVICE, 0.7
            elif any(word in text_lower for word in ['invest', 'stock', 'crypto']):
                return IntentType.INVESTMENT_OPPORTUNITY, 0.7
            elif any(word in text_lower for word in ['save', 'saving', 'goal']):
                return IntentType.SAVINGS_GOAL, 0.8
            elif any(word in text_lower for word in ['anomaly', 'unusual', 'weird']):
                return IntentType.ANOMALY_EXPLANATION, 0.7
            elif any(word in text_lower for word in ['market', 'economy', 'rates']):
                return IntentType.MARKET_UPDATE, 0.7
            elif any(word in text_lower for word in ['tax', 'deduction', 'irs']):
                return IntentType.TAX_QUESTION, 0.8
            else:
                return IntentType.GENERAL_HELP, 0.5
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentType.GENERAL_HELP, 0.5
            
    async def _extract_entities(self, text: str, intent: IntentType) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        try:
            # Use NER model
            if self.entity_extractor:
                ner_results = self.entity_extractor(text)
                
                for entity in ner_results:
                    entity_type = entity['entity_group'].lower()
                    
                    if entity_type == 'per':  # Person
                        entities['person'] = entity['word']
                    elif entity_type == 'org':  # Organization
                        entities['merchant'] = entity['word']
                    elif entity_type == 'loc':  # Location
                        entities['location'] = entity['word']
                    elif entity_type == 'misc':  # Miscellaneous
                        # Try to identify if it's a financial term
                        if any(term in entity['word'].lower() for term in ['dollar', '$', 'usd']):
                            entities['currency'] = 'USD'
            
            # Use spaCy for additional extraction
            doc = self.nlp(text)
            
            # Extract money amounts
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    amount = self._parse_money(ent.text)
                    if amount:
                        entities['amount'] = amount
                elif ent.label_ == "DATE":
                    entities['date'] = ent.text
                elif ent.label_ == "TIME":
                    entities['time'] = ent.text
                elif ent.label_ == "ORG":
                    entities['merchant'] = ent.text
                elif ent.label_ == "PERSON":
                    entities['person'] = ent.text
            
            # Extract categories based on intent
            if intent in [IntentType.SPENDING_ANALYSIS, IntentType.BUDGET_STATUS]:
                categories = self._extract_categories(text)
                if categories:
                    entities['categories'] = categories
            
            # Extract time periods
            time_period = self._extract_time_period(text)
            if time_period:
                entities['time_period'] = time_period
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            
        return entities
        
    def _parse_money(self, money_text: str) -> Optional[float]:
        """Parse money amount from text"""
        try:
            # Remove currency symbols and commas
            cleaned = money_text.replace('$', '').replace(',', '').strip()
            
            # Handle written numbers
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
                'hundred': 100, 'thousand': 1000, 'million': 1000000
            }
            
            # Try to convert written numbers
            words = cleaned.lower().split()
            total = 0
            current = 0
            
            for word in words:
                if word in word_to_num:
                    if word in ['hundred', 'thousand', 'million']:
                        current = current * word_to_num[word] if current else word_to_num[word]
                    else:
                        current += word_to_num[word]
                elif word == 'and':
                    total += current
                    current = 0
            
            total += current
            
            if total > 0:
                return float(total)
            
            # Try direct conversion
            return float(cleaned)
            
        except ValueError:
            return None
            
    def _extract_categories(self, text: str) -> List[str]:
        """Extract spending categories from text"""
        categories = []
        
        category_keywords = {
            'food': ['food', 'restaurant', 'grocery', 'dining', 'eat'],
            'transport': ['transport', 'uber', 'gas', 'fuel', 'transit'],
            'entertainment': ['entertainment', 'movie', 'netflix', 'spotify'],
            'shopping': ['shopping', 'clothes', 'amazon', 'online'],
            'bills': ['bills', 'utilities', 'rent', 'mortgage', 'insurance'],
            'health': ['health', 'medical', 'pharmacy', 'doctor', 'gym']
        }
        
        text_lower = text.lower()
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
                
        return categories
        
    def _extract_time_period(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract time period from text"""
        text_lower = text.lower()
        now = datetime.now()
        
        if 'today' in text_lower:
            return {
                'start': now.replace(hour=0, minute=0, second=0),
                'end': now
            }
        elif 'yesterday' in text_lower:
            yesterday = now - timedelta(days=1)
            return {
                'start': yesterday.replace(hour=0, minute=0, second=0),
                'end': yesterday.replace(hour=23, minute=59, second=59)
            }
        elif 'this week' in text_lower:
            start = now - timedelta(days=now.weekday())
            return {
                'start': start.replace(hour=0, minute=0, second=0),
                'end': now
            }
        elif 'last week' in text_lower:
            start = now - timedelta(days=now.weekday() + 7)
            end = start + timedelta(days=6)
            return {
                'start': start.replace(hour=0, minute=0, second=0),
                'end': end.replace(hour=23, minute=59, second=59)
            }
        elif 'this month' in text_lower:
            return {
                'start': now.replace(day=1, hour=0, minute=0, second=0),
                'end': now
            }
        elif 'last month' in text_lower:
            first_day_this_month = now.replace(day=1)
            last_day_last_month = first_day_this_month - timedelta(days=1)
            first_day_last_month = last_day_last_month.replace(day=1)
            return {
                'start': first_day_last_month.replace(hour=0, minute=0, second=0),
                'end': last_day_last_month.replace(hour=23, minute=59, second=59)
            }
            
        return None
        
    async def execute_command(self, command: VoiceCommand) -> Dict[str, Any]:
        """Execute voice command and return response"""
        try:
            intent_handlers = {
                IntentType.CHECK_BALANCE: self._handle_check_balance,
                IntentType.SPENDING_ANALYSIS: self._handle_spending_analysis,
                IntentType.BUDGET_STATUS: self._handle_budget_status,
                IntentType.SET_BUDGET: self._handle_set_budget,
                IntentType.TRANSACTION_QUERY: self._handle_transaction_query,
                IntentType.FINANCIAL_ADVICE: self._handle_financial_advice,
                IntentType.INVESTMENT_OPPORTUNITY: self._handle_investment_opportunity,
                IntentType.SAVINGS_GOAL: self._handle_savings_goal,
                IntentType.ANOMALY_EXPLANATION: self._handle_anomaly_explanation,
                IntentType.MARKET_UPDATE: self._handle_market_update,
                IntentType.TAX_QUESTION: self._handle_tax_question,
                IntentType.GENERAL_HELP: self._handle_general_help
            }
            
            handler = intent_handlers.get(command.intent, self._handle_general_help)
            response = await handler(command)
            
            # Update conversation context
            await self.conversation_manager.update_context(
                command.user_id,
                command,
                response
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                'success': False,
                'message': "I'm sorry, I encountered an error processing your request.",
                'error': str(e)
            }
            
    async def speak(self, text: str, emotion: str = "neutral"):
        """Convert text to speech with emotion"""
        try:
            # Adjust speech parameters based on emotion
            if emotion == "happy":
                self.tts_engine.setProperty('rate', 160)
                self.tts_engine.setProperty('pitch', 1.1)
            elif emotion == "concerned":
                self.tts_engine.setProperty('rate', 140)
                self.tts_engine.setProperty('pitch', 0.9)
            elif emotion == "urgent":
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('pitch', 1.2)
            else:  # neutral
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('pitch', 1.0)
                
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            
    async def _handle_check_balance(self, command: VoiceCommand) -> Dict[str, Any]:
        """Handle balance check requests"""
        # Implementation would fetch actual balance data
        return {
            'success': True,
            'intent': 'check_balance',
            'message': f"Your current balance is $5,432.10. You have $2,100 in savings and $3,332.10 in checking.",
            'data': {
                'total_balance': 5432.10,
                'checking': 3332.10,
                'savings': 2100.00
            },
            'speak_text': "Your current balance is five thousand four hundred thirty two dollars and ten cents.",
            'emotion': 'neutral'
        }
        
    async def _handle_spending_analysis(self, command: VoiceCommand) -> Dict[str, Any]:
        """Handle spending analysis requests"""
        entities = command.entities
        time_period = entities.get('time_period', {'start': datetime.now() - timedelta(days=30), 'end': datetime.now()})
        categories = entities.get('categories', [])
        
        # Get spending data
        # In production, this would fetch from database
        spending_data = {
            'total': 2845.50,
            'by_category': {
                'food': 450.30,
                'transport': 280.20,
                'shopping': 890.00,
                'bills': 1225.00
            },
            'trend': 'increasing',
            'vs_last_period': 12.5
        }
        
        message = f"In the selected period, you spent ${spending_data['total']:.2f}. "
        
        if categories:
            for cat in categories:
                if cat in spending_data['by_category']:
                    message += f"{cat.capitalize()}: ${spending_data['by_category'][cat]:.2f}. "
        else:
            # Top categories
            top_categories = sorted(spending_data['by_category'].items(), key=lambda x: x[1], reverse=True)[:3]
            message += "Your top spending categories were: "
            for cat, amount in top_categories:
                message += f"{cat} (${amount:.2f}), "
                
        if spending_data['trend'] == 'increasing':
            message += f"Your spending is up {spending_data['vs_last_period']}% compared to last period."
            emotion = 'concerned'
        else:
            message += f"Your spending is down {abs(spending_data['vs_last_period'])}% compared to last period."
            emotion = 'happy'
            
        return {
            'success': True,
            'intent': 'spending_analysis',
            'message': message,
            'data': spending_data,
            'speak_text': message,
            'emotion': emotion
        }
        
    async def _handle_financial_advice(self, command: VoiceCommand) -> Dict[str, Any]:
        """Handle financial advice requests using AI"""
        question = command.text
        
        # Get user's financial context
        user_context = await self._get_user_financial_context(command.user_id)
        
        # Use GPT for personalized advice
        if settings.OPENAI_API_KEY:
            try:
                openai.api_key = settings.OPENAI_API_KEY
                
                prompt = f"""
                As a financial advisor, provide personalized advice based on:
                User Question: {question}
                Financial Context: {json.dumps(user_context)}
                
                Provide practical, actionable advice in 2-3 sentences.
                """
                
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7
                )
                
                advice = response.choices[0].text.strip()
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                advice = "Based on your spending patterns, I recommend setting up automatic transfers to savings and reviewing your subscriptions for potential cuts."
        else:
            # Fallback advice
            advice = "I suggest creating a budget for each spending category and tracking your expenses weekly to stay on target."
            
        return {
            'success': True,
            'intent': 'financial_advice',
            'message': advice,
            'speak_text': advice,
            'emotion': 'neutral',
            'data': {
                'context_used': user_context,
                'personalized': True
            }
        }


class ConversationManager:
    """Manages conversation context and history"""
    
    def __init__(self):
        self.contexts: Dict[int, ConversationContext] = {}
        
    async def get_context(self, user_id: int) -> ConversationContext:
        """Get or create conversation context for user"""
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(
                user_id=user_id,
                session_id=f"session_{user_id}_{datetime.now().timestamp()}",
                history=[],
                current_topic=None,
                user_preferences={},
                last_interaction=datetime.now()
            )
        return self.contexts[user_id]
        
    async def update_context(
        self,
        user_id: int,
        command: VoiceCommand,
        response: Dict[str, Any]
    ):
        """Update conversation context with new interaction"""
        context = await self.get_context(user_id)
        
        # Add to history
        context.history.append({
            'timestamp': command.timestamp,
            'user_input': command.text,
            'intent': command.intent.value,
            'entities': command.entities,
            'response': response['message'],
            'success': response['success']
        })
        
        # Keep only last 10 interactions
        if len(context.history) > 10:
            context.history = context.history[-10:]
            
        # Update current topic
        context.current_topic = command.intent.value
        context.last_interaction = datetime.now()
        
    async def get_conversation_summary(self, user_id: int) -> str:
        """Generate conversation summary"""
        context = await self.get_context(user_id)
        
        if not context.history:
            return "No conversation history"
            
        # Summarize topics discussed
        topics = set(item['intent'] for item in context.history)
        summary = f"Topics discussed: {', '.join(topics)}. "
        summary += f"Total interactions: {len(context.history)}."
        
        return summary 