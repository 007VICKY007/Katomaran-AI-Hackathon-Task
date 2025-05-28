import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Socket } from 'socket.io-client';
import { SendHorizontal, Bot, User, AlertCircle, MessageCircle } from 'lucide-react';
import { usePersonStore } from '../store/personStore';

interface ChatProps {
  socket: Socket | null;
}

interface ChatMessage {
  query: string;
  response: string;
  timestamp: string;
  isUser: boolean;
}

const Chat: React.FC<ChatProps> = ({ socket }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      query: 'Hello! I can answer questions about registered faces. Try asking "Who was registered last?" or "How many people are registered?"',
      response: '',
      timestamp: new Date().toISOString(),
      isUser: false,
    },
  ]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { people, setPeople } = usePersonStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchRegisteredFaces = useCallback(async (retries = 3, delay = 1000) => {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await fetch('http://localhost:5000/api/faces', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
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
        setError(null);
        return;
      } catch (error) {
        console.error(`Attempt ${attempt} failed:`, error);
        if (attempt === retries) {
          setError(`Could not fetch registered faces: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        if (attempt < retries) {
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }
  }, [setPeople]);

  useEffect(() => {
    console.log('Socket connected:', socket?.connected);
    fetchRegisteredFaces();

    if (!socket) return;

    socket.on('queryResult', (data: { prompt: string; answer: string; timestamp: string }) => {
      const botMessage: ChatMessage = {
        query: data.prompt,
        response: data.answer || 'No response received',
        timestamp: data.timestamp || new Date().toISOString(),
        isUser: false,
      };
      setMessages((prev) => [...prev, botMessage]);
      setIsLoading(false);
      setError(null);
      fetchRegisteredFaces();
    });

    socket.on('error', (data: { error: string }) => {
      setError(`Server error: ${data.error}`);
      setIsLoading(false);
    });

    return () => {
      socket.off('queryResult');
      socket.off('error');
    };
  }, [socket, fetchRegisteredFaces]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) {
      setError('Please enter a valid prompt.');
      return;
    }

    const userMessage: ChatMessage = {
      query,
      response: '',
      timestamp: new Date().toISOString(),
      isUser: true,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      if (socket && socket.connected) {
        socket.emit('message', {
          type: 'query',
          prompt: query,
        });
      } else {
        console.log('Sending HTTP request with body:', JSON.stringify({ prompt: query }));
        const response = await fetch('http://localhost:5000/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({ prompt: query }, null, 2),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const data = await response.json();
        console.log('HTTP response:', data);
        const botMessage: ChatMessage = {
          query: data.prompt,
          response: data.answer || 'No response received',
          timestamp: data.timestamp || new Date().toISOString(),
          isUser: false,
        };
        setMessages((prev) => [...prev, botMessage]);
        fetchRegisteredFaces();
        setIsLoading(false);
      }
      setQuery('');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Query error:', errorMessage);
      setError(`Failed to send query: ${errorMessage}`);
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-12rem)] p-6">
      <div className="bg-white rounded-lg shadow-md flex flex-col h-full">
        <div className="p-6 border-b border-gray-200">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center">
            <MessageCircle className="h-6 w-6 mr-2" />
            Face Data Query
          </h1>
          <p className="mt-2 text-gray-600">
            Ask questions about registered faces (e.g., "Who was registered last?" or "How many people are registered?").
          </p>
        </div>

        {people.length === 0 && (
          <div className="mx-6 mt-4 flex items-center p-4 text-yellow-800 border-l-4 border-yellow-300 bg-yellow-50">
            <AlertCircle className="h-5 w-5 mr-2 text-yellow-500" />
            <p className="text-sm">
              No faces registered yet. Please register faces in the Registration tab first.
            </p>
          </div>
        )}

        {error && (
          <div className="mx-6 mt-4 flex items-center p-4 text-red-800 border-l-4 border-red-300 bg-red-50">
            <AlertCircle className="h-5 w-5 mr-2 text-red-500" />
            <p className="text-sm">{error}</p>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={`${message.timestamp}-${index}`}
                className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    message.isUser
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-gray-100 text-gray-800 rounded-bl-none'
                  }`}
                >
                  <div className="flex items-center mb-1">
                    {message.isUser ? (
                      <User className="h-4 w-4 mr-1" />
                    ) : (
                      <Bot className="h-4 w-4 mr-1 text-green-600" />
                    )}
                    <span className="text-xs opacity-75">
                      {new Date(message.timestamp).toLocaleString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                  <div
                    className="whitespace-pre-wrap"
                    dangerouslySetInnerHTML={{
                      __html: (message.isUser ? message.query : message.response).replace(
                        /\*\*(.*?)\*\*/g,
                        '<strong>$1</strong>'
                      ),
                    }}
                  />
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-lg px-4 py-2 bg-gray-100 text-gray-800 rounded-bl-none">
                  <div className="flex items-center space-x-1">
                    <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="p-4 border-t border-gray-200">
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about registered faces..."
              className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm border px-3 py-2 disabled:opacity-50"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <SendHorizontal className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chat;