# FaceID - Real-time Facial Recognition Platform

# Demo Video use this link
https://drive.google.com/file/d/1AafVX6-nNZ8u4otWIoC8wgaSozumPYEe/view?usp=sharing


**Developed by: Vignesh Pandiya G**

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


## 🚀 Running the Application

### 1. Start MongoDB
```bash
mongod --dbpath /path/to/your/database
```

### 2. Run the Node Backend
```bash
python app.py
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

# Output Registration

![image](https://github.com/user-attachments/assets/89d626e1-7aac-4f70-beef-aab4dbdc5986)

# Face Detection 

![image](https://github.com/user-attachments/assets/7d0e20d5-d171-4ba9-9994-53b5099140b8)

# simillarity check 

![image](https://github.com/user-attachments/assets/8c1c5a58-0c68-4555-abda-3a0def266b26)

# registered Faces

![image](https://github.com/user-attachments/assets/cf8ffa75-acb3-4eb9-86cc-0dc26fece453)

# RAG MODEL Answerin System

![image](https://github.com/user-attachments/assets/5c39e50a-eede-40cd-b160-e486405c0fc3)


**Developed with ❤️ by Vignesh Pandiya G**
