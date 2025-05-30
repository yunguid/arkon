import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './VoiceAssistant.css';

const VoiceAssistant = ({ user }) => {
  const [isActive, setIsActive] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [isMinimized, setIsMinimized] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [audioLevel, setAudioLevel] = useState(0);
  
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const micStreamRef = useRef(null);
  const audioWorkletRef = useRef(null);
  
  useEffect(() => {
    // Initialize audio context when component mounts
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    
    return () => {
      disconnectVoiceAssistant();
    };
  }, []);
  
  const connectVoiceAssistant = async () => {
    try {
      setConnectionStatus('connecting');
      
      // Get microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      micStreamRef.current = stream;
      
      // Connect to backend WebSocket that interfaces with OpenAI
      const ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/voice-assistant`);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log('Connected to voice assistant');
        setConnectionStatus('connected');
        setIsActive(true);
        
        // Send initial configuration
        ws.send(JSON.stringify({
          type: 'configure',
          userId: user?.id,
          preferences: {
            voice: 'alloy',
            language: 'en'
          }
        }));
        
        startAudioCapture();
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
      
      ws.onclose = () => {
        console.log('Disconnected from voice assistant');
        setConnectionStatus('disconnected');
        setIsActive(false);
      };
      
    } catch (error) {
      console.error('Failed to connect:', error);
      setConnectionStatus('error');
    }
  };
  
  const disconnectVoiceAssistant = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    
    setIsActive(false);
    setIsListening(false);
    setIsSpeaking(false);
    setConnectionStatus('disconnected');
  };
  
  const startAudioCapture = async () => {
    if (!audioContextRef.current || !micStreamRef.current) return;
    
    try {
      const source = audioContextRef.current.createMediaStreamSource(micStreamRef.current);
      const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      source.connect(processor);
      processor.connect(audioContextRef.current.destination);
      
      processor.onaudioprocess = (e) => {
        if (!isListening || !wsRef.current) return;
        
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Calculate audio level for visualization
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        setAudioLevel(Math.min(1, rms * 10));
        
        // Convert to 16-bit PCM
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Send audio data to server
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(pcm16.buffer);
        }
      };
      
      audioWorkletRef.current = processor;
      
    } catch (error) {
      console.error('Failed to start audio capture:', error);
    }
  };
  
  const handleServerMessage = (data) => {
    switch (data.type) {
      case 'speech_started':
        setIsListening(true);
        setIsSpeaking(false);
        break;
        
      case 'speech_stopped':
        setIsListening(false);
        break;
        
      case 'transcript':
        setTranscript(data.text);
        break;
        
      case 'assistant_speaking':
        setIsSpeaking(true);
        setResponse(data.text);
        break;
        
      case 'assistant_done':
        setIsSpeaking(false);
        break;
        
      case 'audio_data':
        playAudioChunk(data.audio);
        break;
        
      case 'function_call':
        handleFunctionCall(data.function, data.arguments);
        break;
        
      case 'error':
        console.error('Voice assistant error:', data.message);
        setConnectionStatus('error');
        break;
    }
  };
  
  const playAudioChunk = async (base64Audio) => {
    if (!audioContextRef.current) return;
    
    try {
      // Decode base64 audio
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Convert to audio buffer and play
      const audioBuffer = await audioContextRef.current.decodeAudioData(bytes.buffer);
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      source.start();
      
    } catch (error) {
      console.error('Failed to play audio:', error);
    }
  };
  
  const handleFunctionCall = async (functionName, args) => {
    // Handle function calls from the assistant
    console.log('Function call:', functionName, args);
    
    // This would trigger actions in your app
    switch (functionName) {
      case 'check_balance':
        // Trigger balance check
        break;
      case 'get_spending_summary':
        // Show spending summary
        break;
      case 'set_budget':
        // Open budget setting dialog
        break;
      case 'find_transactions':
        // Search transactions
        break;
    }
  };
  
  const toggleListening = () => {
    if (!isActive) {
      connectVoiceAssistant();
    } else {
      setIsListening(!isListening);
      if (wsRef.current) {
        wsRef.current.send(JSON.stringify({
          type: isListening ? 'stop_listening' : 'start_listening'
        }));
      }
    }
  };
  
  return (
    <>
      {/* Floating Voice Assistant Button */}
      <motion.div
        className={`voice-assistant-container ${isMinimized ? 'minimized' : ''}`}
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        {!isMinimized && (
          <motion.div 
            className="voice-assistant-panel"
            layoutId="voice-panel"
          >
            <div className="voice-header">
              <h3>Arkon Voice Assistant</h3>
              <div className="voice-controls">
                <button 
                  className="minimize-btn"
                  onClick={() => setIsMinimized(true)}
                >
                  −
                </button>
                <button 
                  className="close-btn"
                  onClick={disconnectVoiceAssistant}
                >
                  ×
                </button>
              </div>
            </div>
            
            <div className="voice-status">
              <div className={`status-indicator ${connectionStatus}`}></div>
              <span>{connectionStatus === 'connected' ? 'Ready' : connectionStatus}</span>
            </div>
            
            <div className="voice-content">
              {transcript && (
                <motion.div 
                  className="transcript"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <span className="label">You:</span>
                  <p>{transcript}</p>
                </motion.div>
              )}
              
              {response && (
                <motion.div 
                  className="response"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <span className="label">Arkon:</span>
                  <p>{response}</p>
                </motion.div>
              )}
              
              {!transcript && !response && (
                <div className="voice-placeholder">
                  <p>Click the microphone to start talking</p>
                  <p className="hint">Try: "What's my current balance?" or "Show me this month's spending"</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
        
        <motion.button
          className={`voice-button ${isActive ? 'active' : ''} ${isListening ? 'listening' : ''} ${isSpeaking ? 'speaking' : ''}`}
          onClick={toggleListening}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <div className="voice-button-content">
            {/* Animated rings */}
            <AnimatePresence>
              {(isListening || isSpeaking) && (
                <>
                  <motion.div
                    className="voice-ring"
                    initial={{ scale: 1, opacity: 0.8 }}
                    animate={{ 
                      scale: 2 + audioLevel, 
                      opacity: 0 
                    }}
                    transition={{ 
                      duration: 1, 
                      repeat: Infinity,
                      ease: "easeOut"
                    }}
                  />
                  <motion.div
                    className="voice-ring"
                    initial={{ scale: 1, opacity: 0.8 }}
                    animate={{ 
                      scale: 2.5 + audioLevel, 
                      opacity: 0 
                    }}
                    transition={{ 
                      duration: 1.5, 
                      repeat: Infinity,
                      ease: "easeOut",
                      delay: 0.2
                    }}
                  />
                </>
              )}
            </AnimatePresence>
            
            {/* Microphone icon */}
            <svg 
              className="mic-icon" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor"
            >
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" y1="19" x2="12" y2="23" />
              <line x1="8" y1="23" x2="16" y2="23" />
            </svg>
          </div>
        </motion.button>
      </motion.div>
      
      {/* Voice Visualizer Overlay */}
      <AnimatePresence>
        {isListening && !isMinimized && (
          <motion.div 
            className="voice-visualizer"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="visualizer-bars">
              {[...Array(5)].map((_, i) => (
                <motion.div
                  key={i}
                  className="bar"
                  animate={{
                    height: `${20 + Math.random() * 60 * audioLevel}px`
                  }}
                  transition={{
                    duration: 0.1,
                    ease: "easeInOut"
                  }}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default VoiceAssistant; 