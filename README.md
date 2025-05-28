# FaceID - Real-time Facial Recognition Platform

**Developed by: Vignesh Pandiya G**

A comprehensive browser-based facial recognition platform that enables users to register faces, perform real-time recognition, and interact with an intelligent chat interface powered by RAG (Retrieval-Augmented Generation) technology.

## 🚀 Features

### Core Modules

1. **Face Registration**
   - Real-time webcam access for face capture
   - Single face detection and validation
   - Secure face encoding storage with metadata
   - Unique name validation to prevent duplicates

2. **Live Face Recognition**
   - Continuous webcam stream processing
   - Multi-face detection and recognition
   - Real-time bounding box overlays with confidence scores
   - Optimized processing for performance

3. **Intelligent Chat Interface**
   - RAG-powered query system using LangChain + FAISS
   - Natural language queries about registration data
   - WebSocket-based real-time communication
   - Context-aware responses with source citations

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework with WebSocket support
- **MongoDB** - Database for face encodings and metadata
- **Face Recognition** - OpenCV-based face detection and encoding
- **RAG Pipeline**:
  - **LangChain** - Framework for LLM applications
  - **FAISS** - Vector similarity search
  - **HuggingFace Transformers** - Local LLM (Gemma-2b-it)
  - **Sentence Transformers** - Text embeddings

### Key Libraries
- `flask-socketio` - Real-time WebSocket communication
- `pymongo` - MongoDB integration
- `face-recognition` - Facial recognition capabilities
- `PIL` - Image processing
- `numpy` - Numerical computations
- `scipy` - Scientific computing for distance calculations

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- MongoDB (local instance)
- CUDA-compatible GPU (optional, for better performance)
- Webcam/Camera access

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (for faster LLM inference)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space for models and dependencies

## 🔧 Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd faceid-platform
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install flask flask-socketio flask-cors
pip install pymongo
pip install face-recognition
pip install Pillow numpy scipy
pip install langchain langchain-huggingface langchain-community
pip install transformers torch
pip install faiss-cpu  # or faiss-gpu if you have CUDA
pip install sentence-transformers
pip install python-dotenv
pip install dlib
pip install huggingface-hub
```

### 4. Setup MongoDB
```bash
# Install MongoDB Community Edition
# Start MongoDB service
mongod --dbpath /path/to/your/db
```

### 5. Environment Configuration
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
MONGODB_URI=mongodb://localhost:27017/
```

### 6. Install Additional Requirements (if needed)
```bash
# For GPU acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For dlib with CUDA support (optional)
# Follow dlib installation guide for CUDA support
```

## 🚀 Running the Application

### 1. Start MongoDB
```bash
mongod --dbpath /path/to/your/database
```

### 2. Run the Flask Backend
```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Available Endpoints

#### REST API Endpoints
- `POST /api/register` - Register a new face
- `POST /api/recognize` - Recognize faces in an image
- `POST /api/query` - Query the RAG system
- `POST /api/delete` - Delete a registered face
- `GET /api/faces` - Get all registered faces
- `GET /api/health` - System health check

#### WebSocket Events
- `registerFace` - Register face via WebSocket
- `recognizeFace` - Recognize face via WebSocket
- `query` - Query RAG system via WebSocket
- `deleteFace` - Delete face via WebSocket

## 📝 API Usage Examples

### Register a Face
```bash
curl -X POST http://localhost:5000/api/register \
  -F "name=John Doe" \
  -F "image=@face_image.jpg"
```

### Recognize Faces
```bash
curl -X POST http://localhost:5000/api/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,<base64_encoded_image>"}'
```

### Query RAG System
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Who was the last person registered?"}'
```

## 🧠 RAG System Capabilities

The intelligent chat interface can answer queries such as:
- "Who was the last person registered?"
- "At what time was [Name] registered?"
- "How many people are currently registered?"
- "List all registered faces"
- "When was the first registration made?"

## 🔍 System Architecture

### Face Recognition Pipeline
1. **Image Capture** → Camera/Upload
2. **Face Detection** → OpenCV + dlib
3. **Face Encoding** → 128-dimensional vector
4. **Storage** → MongoDB with metadata
5. **Recognition** → Euclidean distance matching

### RAG Pipeline
1. **Document Retrieval** → MongoDB face metadata
2. **Text Embedding** → Sentence Transformers
3. **Vector Storage** → FAISS index
4. **Query Processing** → LangChain RetrievalQA
5. **Response Generation** → Local Gemma-2b-it model

## ⚙️ Configuration Options

### Face Recognition Settings
- **Recognition Threshold**: 0.6 (adjustable in code)
- **Image Processing**: Auto-resize to 500px max dimension
- **Confidence Calculation**: 1.0 - (distance / threshold)

### RAG Settings
- **Embedding Model**: `sentence-transformers/all-MiniLM-L12-v2`
- **LLM Model**: `google/gemma-2b-it`
- **Vector Store**: FAISS with similarity search
- **Max Tokens**: 100 for responses

## 📊 Performance Optimization

### GPU Acceleration
- CUDA support for faster face recognition
- GPU inference for LLM operations
- Optimized image processing pipeline

### Memory Management
- Image resizing for reduced memory usage
- Efficient vector storage with FAISS
- Streaming processing for real-time operations

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Install CUDA toolkit and cuDNN
   - Verify GPU compatibility
   - Fallback to CPU processing automatically

2. **MongoDB Connection Error**
   - Ensure MongoDB service is running
   - Check connection string in environment variables
   - Verify database permissions

3. **Face Recognition Accuracy**
   - Ensure good lighting conditions
   - Use high-resolution images
   - Adjust recognition threshold if needed

4. **RAG System Not Responding**
   - Check HuggingFace token validity
   - Verify model downloads completed
   - Monitor memory usage during inference

### Logging
- Application logs are stored in `app.log`
- Set logging level to DEBUG for detailed information
- Monitor WebSocket connections in browser console

## 🔒 Security Considerations

- Face encodings are stored as numerical vectors (not images)
- Input validation for all API endpoints
- CORS protection configured
- WebSocket origin validation
- No raw image storage in database

## 📈 Future Enhancements

- [ ] Multi-camera support
- [ ] Advanced emotion detection
- [ ] Cloud deployment with Docker
- [ ] Enhanced RAG with external knowledge bases
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is developed by **Vignesh Pandiya G** for educational and demonstration purposes.

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review application logs
- Create an issue in the repository

---

**Developed with ❤️ by Vignesh Pandiya G**
