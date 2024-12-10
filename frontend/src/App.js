import React, { useState, useEffect, useRef } from 'react';
import DocumentsList from './components/DocumentsList';
import UploadForm from './components/UploadForm';
import ChartsView from './components/ChartsView';
import './App.css';
import * as PIXI from 'pixi.js';

function App() {
  const [currentSummary, setCurrentSummary] = useState(null);
  const [error, setError] = useState(null);
  const [refreshDocs, setRefreshDocs] = useState(false);
  const [selectedFileId, setSelectedFileId] = useState(null);
  const canvasRef = useRef(null);
  const appRef = useRef(null);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    async function initPixi() {
      try {
        await PIXI.Assets.init();

        const app = new PIXI.Application();
        await app.init({
          width: windowWidth,
          height: 220,
          backgroundColor: 0x2c3e50,
          antialias: true,
          resolution: window.devicePixelRatio || 1,
          autoDensity: true,
        });
        
        appRef.current = app;
        
        if (canvasRef.current) {
          canvasRef.current.appendChild(app.canvas);
        }

        const container = new PIXI.Container();
        app.stage.addChild(container);

        // Add ARKON text first
        const arkonStyle = new PIXI.TextStyle({
          fontFamily: 'monospace',
          fontSize: 32,
          fill: ['#3498db', '#2980b9'],
          fillGradientType: 1,
          fillGradientStops: [0, 1],
          letterSpacing: 8,
          dropShadow: true,
          dropShadowColor: '#000000',
          dropShadowBlur: 4,
          dropShadowDistance: 2,
        });

        const arkonText = new PIXI.Text('ARKON', arkonStyle);
        arkonText.x = (app.screen.width - arkonText.width) / 2;
        arkonText.y = (220 - arkonText.height) / 2;
        container.addChild(arkonText);

        // Background pattern
        const chars = '⚡✧⚛⚜◈❈✥╬═║│┃┆┇┊┋╌╍╎╏';
        const style = new PIXI.TextStyle({
          fontFamily: 'monospace',
          fontSize: 16,
          fill: ['#3498db', '#2980b9'],
          fillGradientType: 1,
          fillGradientStops: [0, 1],
          letterSpacing: 2,
          dropShadow: true,
          dropShadowColor: '#000000',
          dropShadowBlur: 4,
          dropShadowDistance: 1,
          alpha: 0.7,  // Make background pattern slightly more subtle
        });

        const spacing = 24;
        const columns = Math.ceil(app.screen.width / spacing) + 1; // Add extra column
        const rows = Math.ceil(app.screen.height / spacing) + 1;   // Add extra row
        const particles = [];

        // Create particles
        for (let i = 0; i < columns * rows; i++) {
          const char = chars[Math.floor(Math.random() * chars.length)];
          const text = new PIXI.Text(char, style);
          const col = i % columns;
          const row = Math.floor(i / columns);
          
          const offset = row % 2 ? spacing / 2 : 0;
          text.x = (col * spacing) + offset;
          text.y = row * spacing;
          
          text.alpha = 0.3 + Math.random() * 0.3; // More subtle background
          container.addChild(text);
          particles.push({
            sprite: text,
            x: text.x,
            y: text.y,
            speed: 0.1 + Math.random() * 0.2,
            phase: Math.random() * Math.PI * 2
          });
        }

        // Animation loop
        app.ticker.add(() => {
          const time = performance.now() * 0.001;
          particles.forEach((p) => {
            const wave1 = Math.sin(time + p.phase) * 0.5;
            const wave2 = Math.cos(time * 0.5 + p.x * 0.02) * 0.3;
            p.sprite.y = p.y + (wave1 + wave2) * 15;
            p.sprite.alpha = 0.3 + (wave1 + 0.5) * 0.3;
            p.sprite.rotation = wave2 * 0.2;
          });
          
          // Subtle animation for ARKON text
          arkonText.y = (220 - arkonText.height) / 2 + Math.sin(time) * 3;
          arkonText.alpha = 0.8 + Math.sin(time * 0.5) * 0.2;
        });

        // Handle resize
        const handleResize = () => {
          setWindowWidth(window.innerWidth);
          app.renderer.resize(window.innerWidth, 220);
          arkonText.x = (app.screen.width - arkonText.width) / 2;
        };
        window.addEventListener('resize', handleResize);

        return () => {
          window.removeEventListener('resize', handleResize);
          if (app) {
            app.destroy(true, { children: true, texture: true, baseTexture: true });
          }
        };
      } catch (err) {
        console.error('PIXI initialization error:', err);
      }
    }

    initPixi();
  }, [windowWidth]);

  const handleSelectFile = async (id) => {
    setSelectedFileId(id);
    try {
      const res = await fetch(`http://localhost:8000/file/${id}`);
      if (!res.ok) throw new Error('Failed to fetch summary');
      const data = await res.json();
      if (data.summary) {
        setCurrentSummary(data.summary);
        setError(null);
      } else {
        throw new Error('Invalid summary data');
      }
    } catch (err) {
      setError(err.message);
      console.error('Error fetching summary:', err);
    }
  };

  const handleNewSummary = (summary, fileId) => {
    setCurrentSummary(summary);
    setError(null);
    setRefreshDocs(true);
    if (fileId) {
      handleSelectFile(fileId);
    }
  };

  useEffect(() => {
    if (refreshDocs) {
      setRefreshDocs(false);
    }
  }, [refreshDocs]);

  return (
    <div className="App">
      <div className="webgl-container" ref={canvasRef}></div>
      <div className="content-wrapper">
        <DocumentsList 
          onSelect={handleSelectFile} 
          refreshTrigger={refreshDocs} 
          selectedFileId={selectedFileId} 
        />
        <div className="main-content">
          <UploadForm onNewSummary={handleNewSummary} />
          {error && <div className="error-message">{error}</div>}
          {currentSummary && <ChartsView summary={currentSummary} />}
        </div>
      </div>
    </div>
  );
}

export default App;