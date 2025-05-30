import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { gsap } from 'gsap';
import './Financial3DVisualization.css';

const Financial3DVisualization = ({ data, type = 'spending' }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const composerRef = useRef(null);
  const controlsRef = useRef(null);
  const raycasterRef = useRef(null);
  const mouseRef = useRef(new THREE.Vector2());
  const [selectedObject, setSelectedObject] = useState(null);
  const [tooltip, setTooltip] = useState({ visible: false, content: '', x: 0, y: 0 });
  
  useEffect(() => {
    if (!data || !mountRef.current) return;
    
    // Initialize Three.js scene
    initializeScene();
    
    // Create visualization based on type
    switch (type) {
      case 'spending':
        createSpendingVisualization();
        break;
      case 'categories':
        createCategoryVisualization();
        break;
      case 'timeline':
        createTimelineVisualization();
        break;
      case 'network':
        createNetworkVisualization();
        break;
      default:
        createSpendingVisualization();
    }
    
    // Start animation loop
    animate();
    
    // Cleanup
    return () => {
      cleanup();
    };
  }, [data, type]);

  const initializeScene = () => {
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 10, 20);
    cameraRef.current = camera;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height),
      0.5, // strength
      0.4, // radius
      0.85 // threshold
    );
    composer.addPass(bloomPass);
    composerRef.current = composer;
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    controls.maxPolarAngle = Math.PI / 2;
    controlsRef.current = controls;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.left = -20;
    directionalLight.shadow.camera.right = 20;
    directionalLight.shadow.camera.top = 20;
    directionalLight.shadow.camera.bottom = -20;
    scene.add(directionalLight);
    
    const pointLight = new THREE.PointLight(0x3498db, 1, 50);
    pointLight.position.set(0, 10, 0);
    scene.add(pointLight);
    
    // Raycaster for interaction
    raycasterRef.current = new THREE.Raycaster();
    
    // Event listeners
    window.addEventListener('resize', handleResize);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('click', handleClick);
  };

  const createSpendingVisualization = () => {
    const scene = sceneRef.current;
    if (!scene || !data.daily_expenses) return;
    
    // Create base platform
    const platformGeometry = new THREE.CylinderGeometry(15, 15, 0.5, 32);
    const platformMaterial = new THREE.MeshStandardMaterial({
      color: 0x1a1a1a,
      metalness: 0.7,
      roughness: 0.3
    });
    const platform = new THREE.Mesh(platformGeometry, platformMaterial);
    platform.position.y = -0.25;
    platform.receiveShadow = true;
    scene.add(platform);
    
    // Create bars for daily expenses
    const maxAmount = Math.max(...data.daily_expenses.map(d => d.amount));
    const barGroup = new THREE.Group();
    
    data.daily_expenses.forEach((expense, index) => {
      const height = (expense.amount / maxAmount) * 10;
      const barGeometry = new THREE.BoxGeometry(0.8, height, 0.8);
      
      // Color based on amount
      const hue = 0.6 - (expense.amount / maxAmount) * 0.4;
      const color = new THREE.Color().setHSL(hue, 0.7, 0.5);
      
      const barMaterial = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.2,
        metalness: 0.5,
        roughness: 0.5
      });
      
      const bar = new THREE.Mesh(barGeometry, barMaterial);
      const angle = (index / data.daily_expenses.length) * Math.PI * 2;
      const radius = 10;
      
      bar.position.x = Math.cos(angle) * radius;
      bar.position.y = height / 2;
      bar.position.z = Math.sin(angle) * radius;
      bar.castShadow = true;
      bar.receiveShadow = true;
      
      // Store data for interaction
      bar.userData = {
        type: 'expense',
        date: expense.date,
        amount: expense.amount,
        index: index
      };
      
      barGroup.add(bar);
      
      // Animate bars appearing
      bar.scale.y = 0;
      gsap.to(bar.scale, {
        y: 1,
        duration: 1,
        delay: index * 0.05,
        ease: 'elastic.out(1, 0.5)'
      });
    });
    
    scene.add(barGroup);
    
    // Add floating particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 1000;
    const positions = new Float32Array(particlesCount * 3);
    
    for (let i = 0; i < particlesCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 40;
      positions[i + 1] = Math.random() * 20;
      positions[i + 2] = (Math.random() - 0.5) * 40;
    }
    
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const particlesMaterial = new THREE.PointsMaterial({
      color: 0x3498db,
      size: 0.05,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });
    
    const particles = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particles);
  };

  const createCategoryVisualization = () => {
    const scene = sceneRef.current;
    if (!scene || !data.ai_categories) return;
    
    const categoryData = data.ai_categories.slice(0, 10);
    const total = categoryData.reduce((sum, cat) => sum + cat.amount, 0);
    
    categoryData.forEach((category, index) => {
      const percentage = category.amount / total;
      const size = Math.sqrt(percentage) * 10;
      
      // Create sphere for each category
      const geometry = new THREE.SphereGeometry(size, 32, 32);
      const material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color().setHSL(index / categoryData.length, 0.7, 0.5),
        metalness: 0.3,
        roughness: 0.4,
        clearcoat: 1,
        clearcoatRoughness: 0,
        transparent: true,
        opacity: 0.8
      });
      
      const sphere = new THREE.Mesh(geometry, material);
      
      // Position spheres in a spiral
      const angle = index * 0.5;
      const radius = 5 + index * 1.5;
      sphere.position.x = Math.cos(angle) * radius;
      sphere.position.y = size;
      sphere.position.z = Math.sin(angle) * radius;
      
      sphere.castShadow = true;
      sphere.userData = {
        type: 'category',
        name: category.main_category,
        amount: category.amount,
        percentage: (percentage * 100).toFixed(1)
      };
      
      scene.add(sphere);
      
      // Animate spheres
      gsap.to(sphere.position, {
        y: size + Math.sin(Date.now() * 0.001 + index) * 0.5,
        duration: 2 + Math.random(),
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut'
      });
      
      // Add label
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = 256;
      canvas.height = 64;
      
      context.fillStyle = 'white';
      context.font = 'bold 24px Arial';
      context.textAlign = 'center';
      context.fillText(category.main_category, 128, 32);
      
      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.position.copy(sphere.position);
      sprite.position.y += size + 1;
      sprite.scale.set(4, 1, 1);
      
      scene.add(sprite);
    });
  };

  const createTimelineVisualization = () => {
    const scene = sceneRef.current;
    if (!scene || !data.daily_expenses) return;
    
    // Create timeline path
    const points = [];
    const maxAmount = Math.max(...data.daily_expenses.map(d => d.amount));
    
    data.daily_expenses.forEach((expense, index) => {
      const x = (index - data.daily_expenses.length / 2) * 0.5;
      const y = (expense.amount / maxAmount) * 10;
      const z = 0;
      points.push(new THREE.Vector3(x, y, z));
    });
    
    // Create curve
    const curve = new THREE.CatmullRomCurve3(points);
    const curvePoints = curve.getPoints(100);
    const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
    
    const material = new THREE.LineBasicMaterial({
      color: 0x3498db,
      linewidth: 3,
      transparent: true,
      opacity: 0.8
    });
    
    const line = new THREE.Line(geometry, material);
    scene.add(line);
    
    // Add glowing orbs at data points
    data.daily_expenses.forEach((expense, index) => {
      const orbGeometry = new THREE.SphereGeometry(0.3, 16, 16);
      const orbMaterial = new THREE.MeshBasicMaterial({
        color: 0x3498db,
        emissive: 0x3498db,
        emissiveIntensity: 1
      });
      
      const orb = new THREE.Mesh(orbGeometry, orbMaterial);
      orb.position.copy(points[index]);
      orb.userData = {
        type: 'timeline-point',
        date: expense.date,
        amount: expense.amount
      };
      
      scene.add(orb);
      
      // Pulse animation
      gsap.to(orb.scale, {
        x: 1.2,
        y: 1.2,
        z: 1.2,
        duration: 1,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
        delay: index * 0.1
      });
    });
  };

  const createNetworkVisualization = () => {
    const scene = sceneRef.current;
    if (!scene || !data.recurring_transactions) return;
    
    // Create central node
    const centralGeometry = new THREE.SphereGeometry(2, 32, 32);
    const centralMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x3498db,
      metalness: 0.5,
      roughness: 0.2,
      emissive: 0x3498db,
      emissiveIntensity: 0.3
    });
    const centralNode = new THREE.Mesh(centralGeometry, centralMaterial);
    scene.add(centralNode);
    
    // Create transaction nodes
    data.recurring_transactions.forEach((transaction, index) => {
      const nodeGeometry = new THREE.SphereGeometry(0.5 + transaction.count * 0.1, 16, 16);
      const nodeColor = new THREE.Color().setHSL(index / data.recurring_transactions.length, 0.7, 0.5);
      const nodeMaterial = new THREE.MeshStandardMaterial({
        color: nodeColor,
        emissive: nodeColor,
        emissiveIntensity: 0.2
      });
      
      const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
      const angle = (index / data.recurring_transactions.length) * Math.PI * 2;
      const radius = 8 + Math.random() * 4;
      
      node.position.x = Math.cos(angle) * radius;
      node.position.y = (Math.random() - 0.5) * 4;
      node.position.z = Math.sin(angle) * radius;
      
      node.userData = {
        type: 'recurring',
        description: transaction.description,
        count: transaction.count,
        amount: transaction.averageamount
      };
      
      scene.add(node);
      
      // Create connection to center
      const points = [centralNode.position, node.position];
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const lineMaterial = new THREE.LineBasicMaterial({
        color: nodeColor,
        transparent: true,
        opacity: 0.3
      });
      const connection = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(connection);
      
      // Orbit animation
      gsap.to(node.position, {
        x: Math.cos(angle + Date.now() * 0.001) * radius,
        z: Math.sin(angle + Date.now() * 0.001) * radius,
        duration: 20 + Math.random() * 10,
        repeat: -1,
        ease: 'none'
      });
    });
  };

  const handleResize = () => {
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;
    
    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    
    rendererRef.current.setSize(width, height);
    composerRef.current.setSize(width, height);
  };

  const handleMouseMove = (event) => {
    const rect = rendererRef.current.domElement.getBoundingClientRect();
    mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Check for hover
    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const intersects = raycasterRef.current.intersectObjects(sceneRef.current.children, true);
    
    if (intersects.length > 0 && intersects[0].object.userData.type) {
      const object = intersects[0].object;
      const data = object.userData;
      
      let content = '';
      switch (data.type) {
        case 'expense':
          content = `Date: ${data.date}<br>Amount: $${data.amount.toFixed(2)}`;
          break;
        case 'category':
          content = `${data.name}<br>Amount: $${data.amount.toFixed(2)}<br>${data.percentage}%`;
          break;
        case 'timeline-point':
          content = `${data.date}<br>$${data.amount.toFixed(2)}`;
          break;
        case 'recurring':
          content = `${data.description}<br>Count: ${data.count}<br>Avg: $${data.amount.toFixed(2)}`;
          break;
      }
      
      setTooltip({
        visible: true,
        content: content,
        x: event.clientX + 10,
        y: event.clientY - 30
      });
      
      // Highlight object
      if (selectedObject !== object) {
        if (selectedObject) {
          gsap.to(selectedObject.scale, { x: 1, y: 1, z: 1, duration: 0.3 });
        }
        gsap.to(object.scale, { x: 1.2, y: 1.2, z: 1.2, duration: 0.3 });
        setSelectedObject(object);
      }
    } else {
      setTooltip({ visible: false, content: '', x: 0, y: 0 });
      if (selectedObject) {
        gsap.to(selectedObject.scale, { x: 1, y: 1, z: 1, duration: 0.3 });
        setSelectedObject(null);
      }
    }
  };

  const handleClick = (event) => {
    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const intersects = raycasterRef.current.intersectObjects(sceneRef.current.children, true);
    
    if (intersects.length > 0 && intersects[0].object.userData.type) {
      const data = intersects[0].object.userData;
      console.log('Clicked:', data);
      // Emit event or handle click action
    }
  };

  const animate = () => {
    requestAnimationFrame(animate);
    
    if (controlsRef.current) {
      controlsRef.current.update();
    }
    
    // Rotate particles
    const particles = sceneRef.current.getObjectByProperty('type', 'Points');
    if (particles) {
      particles.rotation.y += 0.0005;
    }
    
    if (composerRef.current) {
      composerRef.current.render();
    }
  };

  const cleanup = () => {
    if (mountRef.current && rendererRef.current) {
      mountRef.current.removeChild(rendererRef.current.domElement);
    }
    
    window.removeEventListener('resize', handleResize);
    
    if (rendererRef.current) {
      rendererRef.current.dispose();
    }
    
    if (sceneRef.current) {
      sceneRef.current.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(material => material.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
    }
  };

  return (
    <div className="financial-3d-visualization">
      <div ref={mountRef} className="threejs-container" />
      {tooltip.visible && (
        <div 
          className="visualization-tooltip"
          style={{ left: tooltip.x, top: tooltip.y }}
          dangerouslySetInnerHTML={{ __html: tooltip.content }}
        />
      )}
      <div className="visualization-controls">
        <button onClick={() => setType('spending')} className={type === 'spending' ? 'active' : ''}>
          Spending
        </button>
        <button onClick={() => setType('categories')} className={type === 'categories' ? 'active' : ''}>
          Categories
        </button>
        <button onClick={() => setType('timeline')} className={type === 'timeline' ? 'active' : ''}>
          Timeline
        </button>
        <button onClick={() => setType('network')} className={type === 'network' ? 'active' : ''}>
          Network
        </button>
      </div>
    </div>
  );
};

export default Financial3DVisualization; 