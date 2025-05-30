/**
 * Arkon Financial Intelligence - Service Worker
 * Advanced offline capabilities with intelligent caching and background sync
 */

const CACHE_VERSION = 'v1.0.0';
const CACHE_NAME = `arkon-cache-${CACHE_VERSION}`;
const DATA_CACHE_NAME = `arkon-data-${CACHE_VERSION}`;
const IMAGE_CACHE_NAME = `arkon-images-${CACHE_VERSION}`;

// Files to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/static/css/main.css',
  '/static/js/main.js',
  '/manifest.json',
  '/offline.html'
];

// API routes to cache
const API_ROUTES = [
  '/api/files',
  '/api/budgets',
  '/api/statistics',
  '/api/ml/health-score'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => {
      console.log('[ServiceWorker] Skip waiting');
      return self.skipWaiting();
    })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName.startsWith('arkon-') && cacheName !== CACHE_NAME && 
              cacheName !== DATA_CACHE_NAME && cacheName !== IMAGE_CACHE_NAME) {
            console.log('[ServiceWorker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[ServiceWorker] Claiming clients');
      return self.clients.claim();
    })
  );
});

// Fetch event - intelligent caching strategy
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-HTTP(S) requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // Handle API requests
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/ml/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }

  // Handle image requests
  if (request.destination === 'image') {
    event.respondWith(handleImageRequest(request));
    return;
  }

  // Handle static assets and HTML
  event.respondWith(handleStaticRequest(request));
});

// API request handler with network-first strategy
async function handleApiRequest(request) {
  const cache = await caches.open(DATA_CACHE_NAME);
  
  try {
    // Try network first
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[ServiceWorker] Network request failed, trying cache:', error);
    
    // Fall back to cache
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      // Add warning header to indicate cached data
      const headers = new Headers(cachedResponse.headers);
      headers.set('X-From-Cache', 'true');
      headers.set('X-Cache-Time', new Date().toISOString());
      
      return new Response(cachedResponse.body, {
        status: cachedResponse.status,
        statusText: cachedResponse.statusText,
        headers: headers
      });
    }
    
    // Return error response
    return new Response(JSON.stringify({
      error: 'Network error and no cached data available',
      offline: true
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Image request handler with cache-first strategy
async function handleImageRequest(request) {
  const cache = await caches.open(IMAGE_CACHE_NAME);
  
  // Check cache first
  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    // Fetch from network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[ServiceWorker] Image fetch failed:', error);
    
    // Return placeholder image
    return caches.match('/images/placeholder.png');
  }
}

// Static asset handler with cache-first strategy
async function handleStaticRequest(request) {
  const cache = await caches.open(CACHE_NAME);
  
  // Check cache first
  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    // Update cache in background
    event.waitUntil(updateCache(request, cache));
    return cachedResponse;
  }
  
  try {
    // Fetch from network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('[ServiceWorker] Static fetch failed:', error);
    
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match('/offline.html');
    }
    
    // Return 404 for other requests
    return new Response('Not found', { status: 404 });
  }
}

// Update cache in background
async function updateCache(request, cache) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      cache.put(request, networkResponse);
    }
  } catch (error) {
    console.log('[ServiceWorker] Background update failed:', error);
  }
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[ServiceWorker] Sync event:', event.tag);
  
  if (event.tag === 'upload-transactions') {
    event.waitUntil(syncUploadTransactions());
  } else if (event.tag === 'update-budgets') {
    event.waitUntil(syncBudgetUpdates());
  } else if (event.tag === 'sync-all') {
    event.waitUntil(syncAllPendingData());
  }
});

// Sync pending transaction uploads
async function syncUploadTransactions() {
  const db = await openIndexedDB();
  const pendingUploads = await getPendingUploads(db);
  
  for (const upload of pendingUploads) {
    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: upload.data,
        headers: upload.headers
      });
      
      if (response.ok) {
        await removePendingUpload(db, upload.id);
        await notifyClients('upload-success', { id: upload.id });
      }
    } catch (error) {
      console.error('[ServiceWorker] Upload sync failed:', error);
    }
  }
}

// Sync budget updates
async function syncBudgetUpdates() {
  const db = await openIndexedDB();
  const pendingBudgets = await getPendingBudgetUpdates(db);
  
  for (const budget of pendingBudgets) {
    try {
      const response = await fetch(`/api/budgets/${budget.id}`, {
        method: budget.method,
        body: JSON.stringify(budget.data),
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        await removePendingBudget(db, budget.id);
        await notifyClients('budget-sync-success', { id: budget.id });
      }
    } catch (error) {
      console.error('[ServiceWorker] Budget sync failed:', error);
    }
  }
}

// Sync all pending data
async function syncAllPendingData() {
  await Promise.all([
    syncUploadTransactions(),
    syncBudgetUpdates()
  ]);
}

// Push notification handler
self.addEventListener('push', (event) => {
  console.log('[ServiceWorker] Push received');
  
  let data = {};
  
  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data = { title: 'Arkon Alert', body: event.data.text() };
    }
  }
  
  const options = {
    body: data.body || 'You have a new notification',
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: data,
    tag: data.tag || 'arkon-notification',
    requireInteraction: data.requireInteraction || false,
    actions: data.actions || [
      { action: 'view', title: 'View' },
      { action: 'dismiss', title: 'Dismiss' }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title || 'Arkon Financial', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  console.log('[ServiceWorker] Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'dismiss') {
    return;
  }
  
  const urlToOpen = event.notification.data?.url || '/';
  
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((windowClients) => {
        // Check if there's already a window/tab open
        for (const client of windowClients) {
          if (client.url === urlToOpen && 'focus' in client) {
            return client.focus();
          }
        }
        
        // Open new window if none found
        if (clients.openWindow) {
          return clients.openWindow(urlToOpen);
        }
      })
  );
});

// Message handler for client communication
self.addEventListener('message', (event) => {
  console.log('[ServiceWorker] Message received:', event.data);
  
  const { type, data } = event.data;
  
  switch (type) {
    case 'skip-waiting':
      self.skipWaiting();
      break;
      
    case 'cache-update':
      updateSpecificCache(data.urls);
      break;
      
    case 'clear-cache':
      clearAllCaches();
      break;
      
    case 'enable-offline':
      enableOfflineMode();
      break;
      
    default:
      console.log('[ServiceWorker] Unknown message type:', type);
  }
});

// Utility functions
async function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('ArkonOfflineDB', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      
      if (!db.objectStoreNames.contains('pendingUploads')) {
        db.createObjectStore('pendingUploads', { keyPath: 'id', autoIncrement: true });
      }
      
      if (!db.objectStoreNames.contains('pendingBudgets')) {
        db.createObjectStore('pendingBudgets', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

async function notifyClients(type, data) {
  const clients = await self.clients.matchAll();
  
  clients.forEach(client => {
    client.postMessage({
      type: type,
      data: data,
      timestamp: new Date().toISOString()
    });
  });
}

async function updateSpecificCache(urls) {
  const cache = await caches.open(CACHE_NAME);
  
  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        await cache.put(url, response);
      }
    } catch (error) {
      console.error('[ServiceWorker] Failed to update cache for:', url);
    }
  }
}

async function clearAllCaches() {
  const cacheNames = await caches.keys();
  
  await Promise.all(
    cacheNames.map(cacheName => {
      if (cacheName.startsWith('arkon-')) {
        return caches.delete(cacheName);
      }
    })
  );
}

// Enable enhanced offline mode
function enableOfflineMode() {
  // Pre-cache additional resources for better offline experience
  const additionalResources = [
    '/api/files?limit=50',
    '/api/budgets',
    '/api/statistics',
    '/api/ml/health-score'
  ];
  
  caches.open(DATA_CACHE_NAME).then(cache => {
    additionalResources.forEach(async (url) => {
      try {
        const response = await fetch(url);
        if (response.ok) {
          cache.put(url, response);
        }
      } catch (error) {
        console.log('[ServiceWorker] Pre-caching failed for:', url);
      }
    });
  });
}

console.log('[ServiceWorker] Loaded successfully'); 