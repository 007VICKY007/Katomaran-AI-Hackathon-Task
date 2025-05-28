import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Socket } from 'socket.io-client';
import Webcam from 'react-webcam';
import { AlertCircle, Camera, X, Check, User, CheckCircle } from 'lucide-react';
import { usePersonStore, Person } from '../store/personStore';

interface RegistrationProps {
  socket: Socket | null;
}

const Registration: React.FC<RegistrationProps> = ({ socket }) => {
  const [name, setName] = useState('');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [registrationDetails, setRegistrationDetails] = useState<{
    name: string;
    timestamp: string;
    date: string;
    day: string;
    id: string;
  } | null>(null);
  
  const webcamRef = useRef<Webcam>(null);
  const { people, setPeople } = usePersonStore();

  const fetchRegisteredFaces = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:5000/api/faces');
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.statusText}`);
      }
      const data = await response.json();
      setPeople(
        (data.faces || []).map((person: { name: string; timestamp: string; created_at: string }) => ({
          id: Date.now().toString() + Math.random(),
          name: person.name,
          timestamp: person.timestamp,
          created_at: new Date(person.created_at),
          image: undefined,
        })),
      );
    } catch (error) {
      console.error('Error fetching faces:', error);
      setError(`Could not fetch registered faces: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, [setPeople]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (showSuccessModal) {
      const timer = setTimeout(() => {
        setShowSuccessModal(false);
        setRegistrationDetails(null);
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [showSuccessModal]);

  useEffect(() => {
    if (!socket) return;

    socket.on('registerResult', (data: { 
      message?: string; 
      id?: string; 
      timestamp?: string; 
      date?: string;
      day?: string;
      name?: string;
      error?: string;
    }) => {
      setIsLoading(false);
      if (data.error) {
        setError(data.error);
        setSuccess(null);
        setShowSuccessModal(false);
      } else {
        setSuccess(data.message || 'Face registered successfully!');
        setError(null);
        setRegistrationDetails({
          name: data.name || name,
          timestamp: data.timestamp || new Date().toISOString(),
          date: data.date || new Date().toISOString().split('T')[0],
          day: data.day || new Date().toLocaleDateString('en-US', { weekday: 'long' }),
          id: data.id || Date.now().toString()
        });
        setShowSuccessModal(true);
        setName('');
        setCapturedImage(null);
        setIsCameraActive(false);
        fetchRegisteredFaces();
      }
    });

    socket.on('error', (data: { error: string }) => {
      setIsLoading(false);
      setError(data.error);
      setSuccess(null);
      setShowSuccessModal(false);
    });

    return () => {
      socket.off('registerResult');
      socket.off('error');
    };
  }, [socket, fetchRegisteredFaces, name]);

  const startCamera = () => {
    setIsCameraActive(true);
    setCapturedImage(null);
  };

  const stopCamera = () => {
    setIsCameraActive(false);
    setCapturedImage(null);
  };

  const captureImage = useCallback(() => {
    if (!webcamRef.current) return;
    setIsCapturing(true);

    setTimeout(() => {
      const imageSrc = webcamRef.current?.getScreenshot();
      setCapturedImage(imageSrc || null);
      setIsCapturing(false);
    }, 1500);
  }, []);

  const resetCapture = () => {
    setCapturedImage(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !capturedImage || !socket || isLoading) {
      setError('Please provide a name and capture an image.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccess(null);
    setShowSuccessModal(false);

    try {
      const person: Person = {
        id: Date.now().toString() + Math.random(),
        name,
        image: capturedImage,
        timestamp: new Date().toISOString(),
        created_at: new Date(),
      };
      setPeople([...people, person]);

      socket.emit('message', {
        type: 'registerFace',
        name,
        image: capturedImage,
      });
    } catch (err) {
      setIsLoading(false);
      setError('An error occurred while processing the image.');
      console.error('File processing error:', err);
    }
  };

  const closeSuccessModal = () => {
    setShowSuccessModal(false);
    setRegistrationDetails(null);
  };

  const videoConstraints = {
    width: 720,
    height: 480,
    facingMode: 'user',
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      {showSuccessModal && registrationDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-8 max-w-md w-full mx-4 transform transition-all">
            <div className="text-center">
              <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-4">
                <CheckCircle className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Registration Successful!
              </h3>
              <div className="text-sm text-gray-600 space-y-2 mb-6">
                <p><strong>Name:</strong> {registrationDetails.name}</p>
                <p><strong>Registration ID:</strong> {registrationDetails.id}</p>
                <p><strong>Timestamp:</strong> {new Date(registrationDetails.timestamp).toLocaleString()}</p>
                <p><strong>Date:</strong> {registrationDetails.date}</p>
                <p><strong>Day:</strong> {registrationDetails.day}</p>
                <p className="text-green-600 font-medium mt-3">
                  ✓ Face data stored successfully in MongoDB
                </p>
              </div>
              <button
                onClick={closeSuccessModal}
                className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-4">Register a Face</h1>
        <p className="text-gray-600 mb-4">
          Capture an image using your webcam to register a face for recognition.
        </p>

        {error && (
          <div className="flex items-center p-4 mb-4 text-red-800 border-l-4 border-red-300 bg-red-50">
            <AlertCircle className="h-5 w-5 mr-2 text-red-500" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        {success && (
          <div className="flex items-center p-4 mb-4 text-green-800 border-l-4 border-green-300 bg-green-50 animate-pulse">
            <CheckCircle className="h-5 w-5 mr-2 text-green-500" />
            <p className="text-sm font-medium">{success}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700">
              Name
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter person's name"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm border px-3 py-2"
              disabled={isLoading}
              required
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-700">Face Image</label>
              <div className="space-x-2">
                {!isCameraActive ? (
                  <button
                    type="button"
                    onClick={startCamera}
                    className="inline-flex items-center px-3 py-2 text-sm rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition-colors"
                    disabled={isLoading}
                  >
                    <Camera className="h-4 w-4 mr-2" />
                    Start Camera
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopCamera}
                    className="inline-flex items-center px-3 py-2 text-sm rounded-md text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 transition-colors"
                    disabled={isLoading}
                  >
                    <X className="h-4 w-4 mr-2" />
                    Stop Camera
                  </button>
                )}
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg border aspect-video flex items-center justify-center">
              {isCameraActive && !capturedImage ? (
                <div className="relative w-full h-full">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="w-full h-full object-cover rounded-lg"
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    {isCapturing && (
                      <div className="bg-black bg-opacity-50 rounded-full p-4 animate-pulse">
                        <User className="h-12 w-12 text-white animate-bounce" />
                      </div>
                    )}
                  </div>
                  <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
                    <button
                      type="button"
                      onClick={captureImage}
                      disabled={isCapturing || isLoading}
                      className="px-4 py-2 text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition-colors"
                    >
                      {isCapturing ? 'Detecting Face...' : 'Capture Image'}
                    </button>
                  </div>
                </div>
              ) : capturedImage ? (
                <div className="relative w-full h-full">
                  <img
                    src={capturedImage}
                    alt="Captured face"
                    className="w-full h-full object-cover rounded-lg"
                  />
                  <div className="absolute bottom-4 right-4">
                    <button
                      type="button"
                      onClick={resetCapture}
                      className="px-3 py-2 text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 disabled:opacity-50 transition-colors"
                      disabled={isLoading}
                    >
                      <X className="h-4 w-4 mr-2 inline" />
                      Retake
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center p-12">
                  <Camera className="mx-auto h-12 w-12 text-gray-400" />
                  <p className="mt-2 text-sm text-gray-500">Start camera to capture an image</p>
                </div>
              )}
            </div>
          </div>

          <div className="pt-4 flex justify-end">
            <button
              type="submit"
              disabled={!name.trim() || !capturedImage || isLoading}
              className="inline-flex items-center px-6 py-2 text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Processing...
                </>
              ) : (
                <>
                  <Check className="h-4 w-4 mr-2" />
                  Register Face
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {people.length > 0 && (
        <div className="mt-8 bg-white rounded-lg shadow-md overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-bold text-gray-800">Registered Faces</h2>
            <p className="text-sm text-gray-600 mt-1">{people.length} face(s) registered</p>
          </div>
          <ul className="divide-y divide-gray-200">
            {people.map((person) => (
              <li key={person.id} className="p-6 flex items-center hover:bg-gray-50 transition-colors">
                <div className="h-16 w-16 rounded-full overflow-hidden bg-gray-100 mr-4 border-2 border-gray-200">
                  {person.image ? (
                    <img
                      src={person.image}
                      alt={`${person.name}'s face`}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <User className="h-full w-full text-gray-400 p-2" />
                  )}
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-gray-900">{person.name}</h3>
                  <p className="text-sm text-gray-500">
                    Registered on {person.created_at.toLocaleString()}
                  </p>
                </div>
                <div className="text-right">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    ✓ Stored
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default Registration;