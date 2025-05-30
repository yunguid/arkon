/**
 * Arkon Financial Analyzer - React Native Mobile App
 * Advanced mobile app with biometric auth, offline support, and native features
 */

import React, { useEffect, useState, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Animated,
  Dimensions,
  Platform,
  StatusBar,
  RefreshControl,
  Alert,
  Vibration,
  AppState,
  NativeModules,
  NativeEventEmitter,
} from 'react-native';
import {
  NavigationContainer,
  DarkTheme,
  DefaultTheme,
} from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { Camera } from 'react-native-vision-camera';
import Sound from 'react-native-sound';
import Haptics from 'react-native-haptic-feedback';
import LinearGradient from 'react-native-linear-gradient';
import { BlurView } from '@react-native-community/blur';
import TouchID from 'react-native-touch-id';
import FaceID from 'react-native-faceid';
import { GraphQLClient } from 'graphql-request';
import { QueryClient, QueryClientProvider, useQuery, useMutation } from 'react-query';
import { Provider as PaperProvider, MD3DarkTheme, MD3LightTheme } from 'react-native-paper';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import LottieView from 'lottie-react-native';
import { GestureHandlerRootView, PanGestureHandler, State } from 'react-native-gesture-handler';
import Reanimated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  interpolate,
  runOnJS,
  useAnimatedGestureHandler,
} from 'react-native-reanimated';
import VoiceRecognition from 'react-native-voice';
import PushNotification from 'react-native-push-notification';
import BackgroundFetch from 'react-native-background-fetch';
import { LineChart, PieChart, BarChart } from 'react-native-chart-kit';
import MapView, { Marker, Heatmap } from 'react-native-maps';
import * as Keychain from 'react-native-keychain';
import RNSecureStorage from 'react-native-secure-storage';
import CryptoJS from 'crypto-js';
import moment from 'moment';

// Initialize modules
const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();
const queryClient = new QueryClient();
const AnimatedLinearGradient = Animated.createAnimatedComponent(LinearGradient);

// Custom hooks
const useTheme = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  useEffect(() => {
    const getTheme = async () => {
      const theme = await AsyncStorage.getItem('theme');
      setIsDarkMode(theme === 'dark');
    };
    getTheme();
  }, []);
  
  const toggleTheme = async () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    await AsyncStorage.setItem('theme', newTheme ? 'dark' : 'light');
  };
  
  return { isDarkMode, toggleTheme };
};

const useOfflineSync = () => {
  const [isOffline, setIsOffline] = useState(false);
  const [pendingSync, setPendingSync] = useState([]);
  
  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsOffline(!state.isConnected);
      if (state.isConnected && pendingSync.length > 0) {
        syncOfflineData();
      }
    });
    
    return () => unsubscribe();
  }, [pendingSync]);
  
  const syncOfflineData = async () => {
    for (const item of pendingSync) {
      try {
        await syncItem(item);
      } catch (error) {
        console.error('Sync failed:', error);
      }
    }
    setPendingSync([]);
  };
  
  const addToSync = async (data) => {
    const newPending = [...pendingSync, data];
    setPendingSync(newPending);
    await AsyncStorage.setItem('pendingSync', JSON.stringify(newPending));
  };
  
  return { isOffline, addToSync, pendingSync };
};

// Biometric Authentication Component
const BiometricAuth = ({ onSuccess, onFail }) => {
  const [biometryType, setBiometryType] = useState(null);
  
  useEffect(() => {
    checkBiometrics();
  }, []);
  
  const checkBiometrics = async () => {
    try {
      const biometrySupported = await TouchID.isSupported();
      setBiometryType(biometrySupported);
    } catch (error) {
      console.log('Biometry not supported');
    }
  };
  
  const authenticate = async () => {
    const optionalConfigObject = {
      title: 'Authentication Required',
      imageColor: '#e00606',
      imageErrorColor: '#ff0000',
      sensorDescription: 'Touch sensor',
      sensorErrorDescription: 'Failed',
      cancelText: 'Cancel',
      fallbackLabel: 'Show Passcode',
      unifiedErrors: false,
      passcodeFallback: false,
    };
    
    try {
      const biometryType = await TouchID.isSupported();
      const success = await TouchID.authenticate(
        'Authenticate to access Arkon Financial',
        optionalConfigObject
      );
      
      if (success) {
        Haptics.trigger('notificationSuccess');
        onSuccess();
      }
    } catch (error) {
      Haptics.trigger('notificationError');
      onFail(error);
    }
  };
  
  return (
    <TouchableOpacity onPress={authenticate} style={styles.biometricButton}>
      <Icon 
        name={biometryType === 'FaceID' ? 'face-recognition' : 'fingerprint'} 
        size={64} 
        color="#4A90E2" 
      />
      <Text style={styles.biometricText}>
        Tap to authenticate with {biometryType || 'Biometrics'}
      </Text>
    </TouchableOpacity>
  );
};

// Dashboard Screen with Advanced Visualizations
const DashboardScreen = ({ navigation }) => {
  const scrollY = useSharedValue(0);
  const [refreshing, setRefreshing] = useState(false);
  const { isOffline } = useOfflineSync();
  
  const { data: financialData, refetch } = useQuery(
    'financialSummary',
    fetchFinancialSummary,
    {
      enabled: !isOffline,
      staleTime: 5 * 60 * 1000,
    }
  );
  
  const onRefresh = async () => {
    setRefreshing(true);
    await refetch();
    setRefreshing(false);
  };
  
  const headerAnimatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        {
          translateY: interpolate(
            scrollY.value,
            [0, 100],
            [0, -50],
            'clamp'
          ),
        },
      ],
      opacity: interpolate(
        scrollY.value,
        [0, 100],
        [1, 0],
        'clamp'
      ),
    };
  });
  
  return (
    <SafeAreaView style={styles.container}>
      <Reanimated.View style={[styles.header, headerAnimatedStyle]}>
        <LinearGradient
          colors={['#4A90E2', '#7B68EE']}
          style={styles.headerGradient}
        >
          <Text style={styles.headerTitle}>Good {getGreeting()}, User!</Text>
          <Text style={styles.headerSubtitle}>
            Your financial health score is {financialData?.healthScore || 'A'}
          </Text>
        </LinearGradient>
      </Reanimated.View>
      
      <Animated.ScrollView
        onScroll={Reanimated.useAnimatedScrollHandler({
          onScroll: (event) => {
            scrollY.value = event.contentOffset.y;
          },
        })}
        scrollEventThrottle={16}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Balance Cards */}
        <View style={styles.balanceContainer}>
          <BalanceCard
            title="Total Balance"
            amount={financialData?.totalBalance || 0}
            trend={financialData?.balanceTrend || 0}
            icon="wallet"
          />
          <BalanceCard
            title="Monthly Spending"
            amount={financialData?.monthlySpending || 0}
            trend={financialData?.spendingTrend || 0}
            icon="trending-down"
            negative
          />
        </View>
        
        {/* Interactive Charts */}
        <View style={styles.chartContainer}>
          <Text style={styles.sectionTitle}>Spending Overview</Text>
          <SpendingChart data={financialData?.spendingData || []} />
        </View>
        
        {/* AI Insights */}
        <View style={styles.insightsContainer}>
          <Text style={styles.sectionTitle}>AI Insights</Text>
          {financialData?.insights?.map((insight, index) => (
            <InsightCard key={index} insight={insight} />
          ))}
        </View>
        
        {/* Quick Actions */}
        <View style={styles.quickActionsContainer}>
          <QuickAction
            icon="microphone"
            label="Voice Command"
            onPress={() => navigation.navigate('VoiceAssistant')}
          />
          <QuickAction
            icon="camera"
            label="Scan Receipt"
            onPress={() => navigation.navigate('ReceiptScanner')}
          />
          <QuickAction
            icon="bell"
            label="Alerts"
            onPress={() => navigation.navigate('Alerts')}
          />
          <QuickAction
            icon="chart-timeline"
            label="Analytics"
            onPress={() => navigation.navigate('Analytics')}
          />
        </View>
      </Animated.ScrollView>
    </SafeAreaView>
  );
};

// Advanced Receipt Scanner with OCR
const ReceiptScannerScreen = () => {
  const [hasPermission, setHasPermission] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const device = Camera.getDevice('back');
  const camera = useRef(null);
  
  useEffect(() => {
    checkCameraPermission();
  }, []);
  
  const checkCameraPermission = async () => {
    const status = await Camera.requestCameraPermission();
    setHasPermission(status === 'authorized');
  };
  
  const takePhoto = async () => {
    if (camera.current) {
      setIsScanning(true);
      const photo = await camera.current.takePhoto({
        qualityPrioritization: 'quality',
      });
      
      // Process with OCR
      const extractedData = await processReceiptOCR(photo.path);
      
      // Show results
      Alert.alert(
        'Receipt Scanned',
        `Total: $${extractedData.total}\nMerchant: ${extractedData.merchant}`,
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Save', 
            onPress: () => saveReceipt(extractedData) 
          },
        ]
      );
      
      setIsScanning(false);
    }
  };
  
  if (!hasPermission) {
    return (
      <View style={styles.centerContainer}>
        <Text>Camera permission required</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
      />
      
      {/* Scanning overlay */}
      <View style={styles.scanOverlay}>
        <View style={styles.scanFrame}>
          <View style={[styles.scanCorner, styles.scanCornerTL]} />
          <View style={[styles.scanCorner, styles.scanCornerTR]} />
          <View style={[styles.scanCorner, styles.scanCornerBL]} />
          <View style={[styles.scanCorner, styles.scanCornerBR]} />
        </View>
        
        <Text style={styles.scanInstructions}>
          Position receipt within frame
        </Text>
      </View>
      
      {/* Capture button */}
      <TouchableOpacity
        style={styles.captureButton}
        onPress={takePhoto}
        disabled={isScanning}
      >
        <Icon name="camera" size={32} color="white" />
      </TouchableOpacity>
    </View>
  );
};

// Voice Assistant Screen
const VoiceAssistantScreen = () => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const waveAnimation = useSharedValue(0);
  
  useEffect(() => {
    VoiceRecognition.onSpeechResults = onSpeechResults;
    VoiceRecognition.onSpeechPartialResults = onSpeechPartialResults;
    
    return () => {
      VoiceRecognition.destroy().then(VoiceRecognition.removeAllListeners);
    };
  }, []);
  
  const startListening = async () => {
    try {
      setIsListening(true);
      setTranscript('');
      await VoiceRecognition.start('en-US');
      
      // Animate wave
      waveAnimation.value = withSpring(1, {
        damping: 10,
        stiffness: 100,
      });
    } catch (error) {
      console.error(error);
    }
  };
  
  const stopListening = async () => {
    try {
      await VoiceRecognition.stop();
      setIsListening(false);
      
      // Reset wave animation
      waveAnimation.value = withTiming(0);
      
      // Process command
      if (transcript) {
        processVoiceCommand(transcript);
      }
    } catch (error) {
      console.error(error);
    }
  };
  
  const onSpeechResults = (event) => {
    setTranscript(event.value[0]);
  };
  
  const onSpeechPartialResults = (event) => {
    setTranscript(event.value[0]);
  };
  
  const processVoiceCommand = async (command) => {
    // Send to backend for processing
    const result = await sendVoiceCommand(command);
    setResponse(result.response);
    
    // Text-to-speech response
    speakResponse(result.response);
  };
  
  const waveStyle = useAnimatedStyle(() => {
    return {
      transform: [
        {
          scale: interpolate(
            waveAnimation.value,
            [0, 1],
            [1, 1.5]
          ),
        },
      ],
      opacity: interpolate(
        waveAnimation.value,
        [0, 1],
        [0.3, 0.8]
      ),
    };
  });
  
  return (
    <View style={styles.voiceContainer}>
      <LinearGradient
        colors={['#1a1a2e', '#0f0f1e']}
        style={styles.voiceGradient}
      >
        {/* Animated voice wave */}
        <View style={styles.voiceWaveContainer}>
          <Reanimated.View style={[styles.voiceWave, waveStyle]}>
            <LinearGradient
              colors={['#4A90E2', '#7B68EE']}
              style={styles.voiceWaveGradient}
            />
          </Reanimated.View>
          
          <TouchableOpacity
            style={styles.voiceButton}
            onPressIn={startListening}
            onPressOut={stopListening}
          >
            <Icon 
              name={isListening ? 'microphone' : 'microphone-outline'} 
              size={48} 
              color="white" 
            />
          </TouchableOpacity>
        </View>
        
        {/* Transcript */}
        {transcript ? (
          <View style={styles.transcriptContainer}>
            <Text style={styles.transcriptLabel}>You said:</Text>
            <Text style={styles.transcriptText}>{transcript}</Text>
          </View>
        ) : null}
        
        {/* Response */}
        {response ? (
          <View style={styles.responseContainer}>
            <Text style={styles.responseLabel}>Arkon says:</Text>
            <Text style={styles.responseText}>{response}</Text>
          </View>
        ) : null}
        
        {/* Suggestions */}
        <View style={styles.suggestionsContainer}>
          <Text style={styles.suggestionsTitle}>Try saying:</Text>
          <TouchableOpacity style={styles.suggestionChip}>
            <Text style={styles.suggestionText}>What's my balance?</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.suggestionChip}>
            <Text style={styles.suggestionText}>Show spending this month</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.suggestionChip}>
            <Text style={styles.suggestionText}>Set budget for groceries</Text>
          </TouchableOpacity>
        </View>
      </LinearGradient>
    </View>
  );
};

// Crypto Portfolio Screen with Real-time Updates
const CryptoPortfolioScreen = () => {
  const [portfolio, setPortfolio] = useState([]);
  const [totalValue, setTotalValue] = useState(0);
  const ws = useRef(null);
  
  useEffect(() => {
    // Connect to WebSocket for real-time prices
    ws.current = new WebSocket('wss://crypto-prices.arkon.com');
    
    ws.current.onmessage = (event) => {
      const update = JSON.parse(event.data);
      updatePortfolioPrices(update);
    };
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);
  
  const updatePortfolioPrices = (priceUpdate) => {
    setPortfolio(prev => 
      prev.map(coin => 
        coin.symbol === priceUpdate.symbol
          ? { ...coin, price: priceUpdate.price, change24h: priceUpdate.change24h }
          : coin
      )
    );
  };
  
  return (
    <ScrollView style={styles.cryptoContainer}>
      {/* Portfolio Summary */}
      <LinearGradient
        colors={['#2E3192', '#1BFFFF']}
        style={styles.cryptoSummary}
      >
        <Text style={styles.cryptoTotalLabel}>Total Portfolio Value</Text>
        <Text style={styles.cryptoTotalValue}>
          ${totalValue.toLocaleString()}
        </Text>
        <Text style={styles.cryptoChange}>
          +12.5% (24h)
        </Text>
      </LinearGradient>
      
      {/* Holdings */}
      {portfolio.map((coin, index) => (
        <CryptoHoldingCard key={index} coin={coin} />
      ))}
      
      {/* DeFi Opportunities */}
      <View style={styles.defiSection}>
        <Text style={styles.sectionTitle}>DeFi Opportunities</Text>
        <DeFiOpportunityCard
          protocol="Aave"
          asset="USDC"
          apy={3.5}
          riskLevel="Low"
        />
        <DeFiOpportunityCard
          protocol="Yearn"
          asset="ETH"
          apy={6.8}
          riskLevel="Medium"
        />
      </View>
    </ScrollView>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1000,
  },
  headerGradient: {
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.8)',
  },
  balanceContainer: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    marginTop: 140,
  },
  chartContainer: {
    backgroundColor: 'white',
    margin: 15,
    padding: 20,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 5,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 15,
    color: '#2c3e50',
  },
  insightsContainer: {
    paddingHorizontal: 15,
  },
  quickActionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 10,
    marginTop: 20,
    marginBottom: 30,
  },
  biometricButton: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 30,
  },
  biometricText: {
    marginTop: 15,
    fontSize: 16,
    color: '#4A90E2',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanFrame: {
    width: 250,
    height: 350,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.5)',
    backgroundColor: 'transparent',
  },
  scanCorner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: '#4A90E2',
  },
  scanCornerTL: {
    top: 0,
    left: 0,
    borderTopWidth: 3,
    borderLeftWidth: 3,
  },
  scanCornerTR: {
    top: 0,
    right: 0,
    borderTopWidth: 3,
    borderRightWidth: 3,
  },
  scanCornerBL: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
  },
  scanCornerBR: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 3,
    borderRightWidth: 3,
  },
  scanInstructions: {
    marginTop: 20,
    fontSize: 16,
    color: 'white',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  captureButton: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#4A90E2',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 5,
  },
  voiceContainer: {
    flex: 1,
  },
  voiceGradient: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  voiceWaveContainer: {
    width: 200,
    height: 200,
    justifyContent: 'center',
    alignItems: 'center',
  },
  voiceWave: {
    position: 'absolute',
    width: 150,
    height: 150,
    borderRadius: 75,
  },
  voiceWaveGradient: {
    flex: 1,
    borderRadius: 75,
  },
  voiceButton: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#4A90E2',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 10,
  },
  transcriptContainer: {
    marginTop: 40,
    paddingHorizontal: 30,
  },
  transcriptLabel: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.6)',
    marginBottom: 5,
  },
  transcriptText: {
    fontSize: 18,
    color: 'white',
    fontWeight: '500',
  },
  responseContainer: {
    marginTop: 30,
    paddingHorizontal: 30,
    padding: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 15,
    marginHorizontal: 20,
  },
  responseLabel: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.6)',
    marginBottom: 5,
  },
  responseText: {
    fontSize: 16,
    color: 'white',
    lineHeight: 24,
  },
  suggestionsContainer: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
  },
  suggestionsTitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.6)',
    marginBottom: 10,
  },
  suggestionChip: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  suggestionText: {
    color: 'white',
    fontSize: 14,
  },
  cryptoContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  cryptoSummary: {
    padding: 30,
    alignItems: 'center',
  },
  cryptoTotalLabel: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.8)',
  },
  cryptoTotalValue: {
    fontSize: 36,
    fontWeight: 'bold',
    color: 'white',
    marginVertical: 10,
  },
  cryptoChange: {
    fontSize: 18,
    color: '#4ADE80',
    fontWeight: '500',
  },
  defiSection: {
    padding: 20,
  },
});

// Helper functions
const getGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return 'Morning';
  if (hour < 18) return 'Afternoon';
  return 'Evening';
};

const fetchFinancialSummary = async () => {
  // Implement API call
  return {
    totalBalance: 12543.67,
    monthlySpending: 3421.89,
    healthScore: 'A',
    balanceTrend: 5.2,
    spendingTrend: -2.1,
    insights: [
      { type: 'saving', message: 'You saved 15% more than last month!' },
      { type: 'alert', message: 'Unusual spending detected at Store XYZ' },
    ],
  };
};

const processReceiptOCR = async (imagePath) => {
  // Implement OCR processing
  return {
    total: 45.67,
    merchant: 'Whole Foods',
    date: new Date(),
    items: [],
  };
};

const sendVoiceCommand = async (command) => {
  // Implement voice command processing
  return {
    response: "Your current balance is $12,543.67",
  };
};

const speakResponse = (text) => {
  // Implement text-to-speech
};

// App Component
export default function App() {
  const { isDarkMode, toggleTheme } = useTheme();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  
  useEffect(() => {
    configurePushNotifications();
    setupBackgroundSync();
  }, []);
  
  const configurePushNotifications = () => {
    PushNotification.configure({
      onRegister: function (token) {
        console.log('TOKEN:', token);
      },
      onNotification: function (notification) {
        console.log('NOTIFICATION:', notification);
      },
      permissions: {
        alert: true,
        badge: true,
        sound: true,
      },
      popInitialNotification: true,
      requestPermissions: true,
    });
  };
  
  const setupBackgroundSync = () => {
    BackgroundFetch.configure(
      {
        minimumFetchInterval: 15,
        stopOnTerminate: false,
        startOnBoot: true,
      },
      async (taskId) => {
        console.log('[BackgroundFetch] taskId:', taskId);
        // Perform background sync
        BackgroundFetch.finish(taskId);
      },
      (error) => {
        console.log('[BackgroundFetch] failed to start:', error);
      }
    );
  };
  
  if (!isAuthenticated) {
    return (
      <View style={styles.container}>
        <LinearGradient
          colors={['#4A90E2', '#7B68EE']}
          style={StyleSheet.absoluteFill}
        >
          <SafeAreaView style={styles.container}>
            <View style={styles.centerContainer}>
              <LottieView
                source={require('./assets/animations/finance-logo.json')}
                autoPlay
                loop
                style={{ width: 200, height: 200 }}
              />
              <Text style={[styles.headerTitle, { marginTop: 30 }]}>
                Arkon Financial
              </Text>
              <BiometricAuth
                onSuccess={() => setIsAuthenticated(true)}
                onFail={(error) => console.log(error)}
              />
            </View>
          </SafeAreaView>
        </LinearGradient>
      </View>
    );
  }
  
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <QueryClientProvider client={queryClient}>
        <PaperProvider theme={isDarkMode ? MD3DarkTheme : MD3LightTheme}>
          <SafeAreaProvider>
            <NavigationContainer theme={isDarkMode ? DarkTheme : DefaultTheme}>
              <Tab.Navigator
                screenOptions={{
                  tabBarActiveTintColor: '#4A90E2',
                  tabBarInactiveTintColor: 'gray',
                  tabBarStyle: {
                    backgroundColor: isDarkMode ? '#1a1a1a' : 'white',
                    borderTopWidth: 0,
                    elevation: 10,
                    shadowColor: '#000',
                    shadowOffset: { width: 0, height: -3 },
                    shadowOpacity: 0.1,
                    shadowRadius: 5,
                  },
                }}
              >
                <Tab.Screen
                  name="Dashboard"
                  component={DashboardScreen}
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Icon name="view-dashboard" color={color} size={size} />
                    ),
                    headerShown: false,
                  }}
                />
                <Tab.Screen
                  name="Transactions"
                  component={TransactionsScreen}
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Icon name="format-list-bulleted" color={color} size={size} />
                    ),
                  }}
                />
                <Tab.Screen
                  name="Scan"
                  component={ReceiptScannerScreen}
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Icon name="camera" color={color} size={size} />
                    ),
                  }}
                />
                <Tab.Screen
                  name="Voice"
                  component={VoiceAssistantScreen}
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Icon name="microphone" color={color} size={size} />
                    ),
                    headerShown: false,
                  }}
                />
                <Tab.Screen
                  name="Crypto"
                  component={CryptoPortfolioScreen}
                  options={{
                    tabBarIcon: ({ color, size }) => (
                      <Icon name="bitcoin" color={color} size={size} />
                    ),
                  }}
                />
              </Tab.Navigator>
            </NavigationContainer>
          </SafeAreaProvider>
        </PaperProvider>
      </QueryClientProvider>
    </GestureHandlerRootView>
  );
} 