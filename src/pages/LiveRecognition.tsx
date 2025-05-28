import React, { useState, useEffect, useRef } from 'react';
import { Socket } from 'socket.io-client';
import { AlertCircle, Video, VideoOff } from 'lucide-react';

interface LiveRecognitionProps {
  socket: Socket | null;
}

interface Face {
  name: string;
  confidence: number;
  location: [number, number, number, number]; // [top, right, bottom, left]
  age?: number;
}

const LiveRecognition: React.FC<LiveRecognitionProps> = ({ socket }) => {
  const [faces, setFaces] = useState<Face[]>([]);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<Face | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (!socket) return;

    socket.on('connect', () => {
      setError(null);
      console.log('Connected to Socket.IO server');
    });

    socket.on('recognizeResult', (data: { faces?: Face[]; count?: number; message?: string; error?: string }) => {
      setIsLoading(false);
      if (data.error) {
        setError(`Recognition error: ${data.error}`);
        setFaces([]);
        setAnalysisResult(null);
        return;
      }
      if (data.message === 'No faces detected') {
        setFaces([]);
        setAnalysisResult(null);
        return;
      }
      setFaces(data.faces || []);
      if (data.faces && data.faces.length > 0) {
        setAnalysisResult(data.faces[0]);
      }
      setError(null);
    });

    socket.on('error', (data: { error: string }) => {
      setIsLoading(false);
      setError(data.error.includes('encoding')
        ? 'Failed to process face data. Ensure valid faces are registered.'
        : `Server error: ${data.error}`);
      setFaces([]);
      setAnalysisResult(null);
    });

    socket.on('disconnect', () => {
      setIsLoading(false);
      setError('Disconnected from server. Please check the backend.');
      setFaces([]);
      setAnalysisResult(null);
      stopCamera();
    });

    return () => {
      socket.off('connect');
      socket.off('recognizeResult');
      socket.off('error');
      socket.off('disconnect');
    };
  }, [socket]);

  const fetchRecognitionResult = async (imageData: string) => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/recognize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
        signal: AbortSignal.timeout(5000), // Timeout after 5 seconds
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: { faces?: Face[]; message?: string; error?: string } = await response.json();

      if (result.error) {
        setError(`Recognition error: ${result.error}`);
        setFaces([]);
        setAnalysisResult(null);
        return;
      }

      if (result.message === 'No faces detected') {
        setFaces([]);
        setAnalysisResult(null);
        return;
      }

      setFaces(result.faces || []);
      if (result.faces && result.faces.length > 0) {
        setAnalysisResult(result.faces[0]);
      }
      setError(null);
    } catch (err) {
      setError(`Fetch error: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setFaces([]);
      setAnalysisResult(null);
      console.error('Fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraOn(true);
        setError(null);
        sendFrames();
      }
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
    setFaces([]);
    setAnalysisResult(null);
    setIsLoading(false);
  };

  const sendFrames = () => {
    if (!isCameraOn || !videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');

    if (!context) {
      setError('Failed to get canvas context.');
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const captureFrame = () => {
      if (!isCameraOn || !videoRef.current || !canvasRef.current) return;

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);

      // Validate image data
      if (!imageData.startsWith('data:image/jpeg;base64,')) {
        setError('Invalid image data format.');
        return;
      }

      if (socket && socket.connected) {
        socket.emit('message', {
          type: 'recognizeFace',
          image: imageData,
        });
      } else {
        fetchRecognitionResult(imageData);
      }

      if (isCameraOn) {
        setTimeout(captureFrame, 500); // Reduced to 500ms for smoother updates
      }
    };

    captureFrame();
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-4">Live Face Recognition</h1>
        <p className="text-gray-600 mb-4">
          Stream video from your webcam for real-time face recognition. Analysis updates every 0.5 seconds with logs stored in MongoDB.
        </p>

        {error && (
          <div className="flex items-center p-4 mb-4 text-red-800 border-l-4 border-red-300 bg-red-50">
            <AlertCircle className="h-5 w-5 mr-2 text-red-500" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        <div className="relative mb-4">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full rounded-md border border-gray-300"
            style={{ display: isCameraOn ? 'block' : 'none' }}
          />
          <canvas ref={canvasRef} className="hidden" />
          {isCameraOn && (
            <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
              {faces.map((face, index) => (
                <div
                  key={index}
                  className="absolute border-2 border-green-500"
                  style={{
                    top: `${face.location[0]}px`,
                    left: `${face.location[3]}px`,
                    width: `${face.location[1] - face.location[3]}px`,
                    height: `${face.location[2] - face.location[0]}px`,
                  }}
                >
                  <div className="absolute -top-8 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded">
                    {face.name} ({(face.confidence * 100).toFixed(2)}%) {face.age ? `| Age: ${face.age}` : ''}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex space-x-4">
          <button
            onClick={isCameraOn ? stopCamera : startCamera}
            className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
              isCameraOn
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-blue-600 hover:bg-blue-700'
            } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed`}
            disabled={isLoading}
          >
            {isCameraOn ? (
              <>
                <VideoOff className="h-4 w-4 mr-2" />
                Stop Camera
              </>
            ) : (
              <>
                <Video className="h-4 w-4 mr-2" />
                {isLoading ? 'Starting...' : 'Start Camera'}
              </>
            )}
          </button>
        </div>

        <div className="mt-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-2">Analysis Dashboard</h2>
          {isLoading ? (
            <p className="text-sm text-gray-500 italic">Processing...</p>
          ) : analysisResult ? (
            <div className="bg-gray-50 rounded-md p-4 border border-gray-200">
              <p className="text-sm text-gray-700">
                <strong>Name:</strong> {analysisResult.name}
              </p>
              <p className="text-sm text-gray-700">
                <strong>Accuracy Rate:</strong> {(analysisResult.confidence * 100).toFixed(2)}%
              </p>
              {analysisResult.age && (
                <p className="text-sm text-gray-700">
                  <strong>Estimated Age:</strong> {analysisResult.age}
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500 italic">
              No face detected or waiting for analysis...
            </p>
          )}
        </div>

        {faces.length > 0 && (
          <div className="mt-4">
            <h2 className="text-lg font-semibold text-gray-800">Real-Time Recognized Faces</h2>
            <ul className="mt-2 space-y-2">
              {faces.map((face, index) => (
                <li key={index} className="text-gray-600">
                  {face.name} - Accuracy: {(face.confidence * 100).toFixed(2)}% {face.age ? `- Age: ${face.age}` : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveRecognition;