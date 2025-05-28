import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import io, { Socket } from 'socket.io-client';

// Pages
import Registration from './pages/Registration';
import LiveRecognition from './pages/LiveRecognition';
import Chat from './pages/Chat';
// Components
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import BackendWarning from './components/BackendWarning';

const App: React.FC = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [showBackendWarning, setShowBackendWarning] = useState(true);

  useEffect(() => {
    const socketIo = io('http://localhost:5000', {
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socketIo.on('connect', () => {
      console.log('Connected to WebSocket server');
    });

    socketIo.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
    });

    setSocket(socketIo);

    return () => {
      socketIo.disconnect();
    };
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {showBackendWarning && (
          <BackendWarning onClose={() => setShowBackendWarning(false)} />
        )}
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Navigate to="/register" replace />} />
            <Route path="/register" element={<Registration socket={socket} />} />
            <Route path="/recognition" element={<LiveRecognition socket={socket} />} />
            <Route path="/chat" element={<Chat socket={socket}/>} /> {/* Remove socket prop */}
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
};

export default App;