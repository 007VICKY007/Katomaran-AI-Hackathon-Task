from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from pymongo import MongoClient
from PIL import Image
import io
import face_recognition
import numpy as np
from datetime import datetime
import base64
from scipy.spatial import distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging
import os
from dotenv import load_dotenv
import dlib
from huggingface_hub import login
import shutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['facial_recognition_db']
collection = db['faces']

# Initialize RAG components
vector_store = None
qa_chain = None

def check_system_requirements():
    """Verify system requirements."""
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Using CPU for LLM inference, which may be slower.")
    else:
        logging.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    if not dlib.DLIB_USE_CUDA:
        logging.warning("Dlib CUDA not enabled. Using CPU for face_recognition.")

def initialize_rag():
    """Initialize RAG pipeline with fine-tuned components for face recognition queries."""
    global vector_store, qa_chain
    try:
        logging.debug("Starting fine-tuned RAG pipeline initialization...")

        # Log in to Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logging.debug("Logged in to Hugging Face.")
        else:
            logging.warning("No HF_TOKEN found. Using open-access models.")

        # Fetch and preprocess documents from MongoDB
        logging.debug("Fetching face details from MongoDB...")
        documents = []
        metadata_list = []
        for doc in collection.find(
            {"name": {"$ne": "No Faces Registered"}, "timestamp": {"$exists": True}},
            {'name': 1, 'timestamp': 1, 'created_at': 1, '_id': 0}
        ):
            try:
                if not all(key in doc for key in ['name', 'timestamp', 'created_at']):
                    logging.warning(f"Skipping invalid document: {doc}")
                    continue
                if not isinstance(doc['timestamp'], str):
                    logging.warning(f"Invalid timestamp in document: {doc}")
                    continue
                text = (
                    f"Person: {doc['name']}, "
                    f"Registration Timestamp: {doc['timestamp']}, "
                    f"Registration Date: {doc['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                documents.append(text)
                metadata_list.append({"name": doc['name'], "timestamp": doc['timestamp']})
            except Exception as e:
                logging.warning(f"Error processing document {doc}: {str(e)}")
                continue
        logging.debug(f"Fetched {len(documents)} valid face documents.")

        if not documents:
            logging.warning("No valid face documents found. Using default document.")
            default_name = "No Faces Registered"
            if not collection.find_one({"name": default_name}):
                timestamp = datetime.now()
                collection.insert_one({
                    "name": default_name,
                    "encoding": [],
                    "timestamp": timestamp.isoformat(),
                    "created_at": timestamp
                })
            documents = ["No registered faces available."]
            metadata_list = [{"name": default_name, "timestamp": timestamp.isoformat()}]

        # Clear existing FAISS index to ensure fresh data
        faiss_index_path = "faiss_index"
        if os.path.exists(faiss_index_path):
            logging.debug("Clearing existing FAISS index...")
            shutil.rmtree(faiss_index_path)

        # Use optimized embeddings model
        logging.debug("Loading embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        logging.debug("Embeddings model loaded.")

        # Create new FAISS index
        logging.debug("Creating new FAISS index...")
        vector_store = FAISS.from_texts(
            texts=documents,
            embedding=embeddings,
            metadatas=metadata_list
        )
        vector_store.save_local(faiss_index_path)
        logging.debug("FAISS vector store initialized.")

        # Define fine-tuned prompt template
        logging.debug("Setting up fine-tuned prompt template...")
        prompt_template = """You are an assistant for a facial recognition system. Use the following context about registered faces to answer the question accurately. Provide concise answers, focusing only on the requested information. If the answer is not in the context, state that clearly.

Context: {context}

Question: {question}

Answer: """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Initialize a more capable local LLM
        logging.debug("Initializing fine-tuned LLM...")
        model_id = "google/gemma-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.6,
            top_p=0.85,
            device_map="auto",
            return_full_text=False
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logging.debug("Fine-tuned LLM initialized.")

        # Create RetrievalQA chain with optimized retriever
        logging.debug("Creating fine-tuned RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1}  # Reduced to 1 for exact retrieval
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        logging.debug("Fine-tuned RAG pipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize fine-tuned RAG pipeline: {str(e)}")
        raise

def query_rag(prompt):
    """Handle RAG queries with fine-tuned pipeline."""
    try:
        if not qa_chain:
            logging.error("RAG pipeline not initialized.")
            return {"error": "RAG pipeline not initialized. Please register faces or check server configuration."}
        
        # Verify MongoDB state
        current_names = {doc['name'] for doc in collection.find(
            {"name": {"$ne": "No Faces Registered"}, "timestamp": {"$exists": True}},
            {'name': 1}
        )}
        logging.debug(f"Current MongoDB names: {current_names}")

        result = qa_chain({"query": prompt})
        answer = result["result"].strip()
        
        if answer.startswith("Answer:"):
            answer = answer[len("Answer:"):].strip()

        # Validate retrieved documents against MongoDB
        source_docs = [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in result.get("source_documents", [])
        ]
        for doc in source_docs:
            doc_name = doc['metadata'].get('name', '')
            if doc_name not in current_names and doc_name != "No Faces Registered":
                logging.warning(f"Retrieved document for {doc_name} not in MongoDB. Reinitializing RAG...")
                initialize_rag()
                result = qa_chain({"query": prompt})
                answer = result["result"].strip()
                if answer.startswith("Answer:"):
                    answer = answer[len("Answer:"):].strip()
                source_docs = [
                    {"text": doc.page_content, "metadata": doc.metadata}
                    for doc in result.get("source_documents", [])
                ]
                break

        logging.info(f"Query processed: {prompt} -> {answer}")
        return {
            "prompt": prompt,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "source_documents": source_docs
        }
    except Exception as e:
        logging.error(f"Query failed: {str(e)}")
        return {"error": f"Query failed: {str(e)}"}

def resize_image(image, max_size=500):
    """Resize image to reduce processing time."""
    image.thumbnail((max_size, max_size))
    return image

@app.route('/api/register', methods=['POST'])
def register_face():
    """Register a face with name and store in MongoDB."""
    try:
        if 'image' not in request.files or 'name' not in request.form:
            logging.error("Image and name required in request.")
            return jsonify({'error': 'Image and name required'}), 400

        file = request.files['image']
        name = request.form['name']

        if file.filename == '' or not name.strip():
            logging.error("Invalid image or name provided.")
            return jsonify({'error': 'Invalid image or name'}), 400

        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        image = resize_image(image)
        image_np = np.array(image)

        face_locations = face_recognition.face_locations(image_np)
        if len(face_locations) != 1:
            logging.error(f"Detected {len(face_locations)} faces. Exactly one face required.")
            return jsonify({'error': 'Exactly one face should be detected'}), 400

        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        encoding_list = face_encodings[0].tolist()

        if collection.find_one({"name": name}):
            logging.error(f'Name "{name}" already exists in database.')
            return jsonify({'error': f'Name "{name}" already exists'}), 400

        timestamp = datetime.now()
        result = collection.insert_one({
            "name": name,
            "encoding": encoding_list,
            "timestamp": timestamp.isoformat(),
            "created_at": timestamp
        })

        socketio.emit('registerResult', {
            'message': f'Successfully registered {name}',
            'id': str(result.inserted_id),
            'name': name,
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%Y-%m-%d'),
            'day': timestamp.strftime('%A')
        })

        initialize_rag()  # Reinitialize RAG to include new document

        logging.info(f"Successfully registered face: {name}")
        return jsonify({
            'message': f'Successfully registered {name}',
            'id': str(result.inserted_id),
            'timestamp': timestamp.isoformat()
        })
    except Exception as e:
        logging.error(f"Registration failed: {str(e)}")
        socketio.emit('error', {'error': f'Registration failed: {str(e)}'})
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/delete', methods=['POST'])
def delete_face():
    """Delete a face from MongoDB and update RAG pipeline."""
    try:
        if not request.is_json:
            logging.error("Invalid Content-Type: Expected application/json")
            return jsonify({'error': 'Invalid Content-Type: Expected application/json'}), 415

        data = request.get_json()
        if not data or 'name' not in data:
            logging.error("No name provided in request.")
            return jsonify({'error': 'No name provided'}), 400

        name = data['name']
        result = collection.delete_one({"name": name})
        if result.deleted_count == 0:
            logging.error(f"No face found with name: {name}")
            return jsonify({'error': f'No face found with name: {name}'}), 404

        initialize_rag()  # Reinitialize RAG to reflect deletion

        socketio.emit('deleteResult', {
            'message': f'Successfully deleted {name}',
            'timestamp': datetime.now().isoformat()
        })

        logging.info(f"Successfully deleted face: {name}")
        return jsonify({
            'message': f'Successfully deleted {name}',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Deletion failed: {str(e)}")
        socketio.emit('error', {'error': f'Deletion failed: {str(e)}'})
        return jsonify({'error': f'Deletion failed: {str(e)}'}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize faces in an image and return matches."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            logging.error("No image provided in request.")
            return jsonify({'error': 'No image provided'}), 400

        base64_image = data['image']
        image_data = base64.b64decode(base64_image.split(',')[1])
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        image = resize_image(image)
        image_np = np.array(image)

        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        if len(face_encodings) == 0:
            logging.info("No faces detected in image.")
            return jsonify({'message': 'No faces detected', 'count': 0}), 200

        known_faces = list(collection.find({"name": {"$ne": "No Faces Registered"}, "encoding": {"$exists": True, "$ne": []}}, {'name': 1, 'encoding': 1}))
        if not known_faces:
            logging.error("No registered faces in database.")
            return jsonify({'error': 'No registered faces'}), 400

        results = []
        threshold = 0.6

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = 'Unknown'
            confidence = 0.0
            min_distance = float('inf')

            for known_face in known_faces:
                try:
                    if 'encoding' not in known_face or not isinstance(known_face['encoding'], list):
                        continue
                    known_encoding = np.array(known_face['encoding'])
                    if known_encoding.shape != (128,):
                        continue
                    dist = distance.euclidean(encoding, known_encoding)
                    if dist < min_distance and dist < threshold:
                        min_distance = dist
                        name = known_face['name']
                        confidence = 1.0 - (dist / threshold)
                except Exception:
                    continue

            results.append({
                'name': name,
                'confidence': confidence,
                'location': [top, right, bottom, left]
            })

        socketio.emit('recognizeResult', {
            'faces': results,
            'count': len(results),
            'message': f'Detected {len(results)} face(s)'
        })

        logging.info(f"Recognized {len(results)} face(s).")
        return jsonify({
            'faces': results,
            'count': len(results),
            'message': f'Detected {len(results)} face(s)'
        })
    except Exception as e:
        logging.error(f"Recognition failed: {str(e)}")
        socketio.emit('error', {'error': f'Recognition failed: {str(e)}'})
        return jsonify({'error': f'Recognition failed: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def query_rag_endpoint():
    """Handle RAG queries using fine-tuned pipeline."""
    try:
        if not request.is_json:
            logging.error("Invalid Content-Type: Expected application/json")
            return jsonify({'error': 'Invalid Content-Type: Expected application/json'}), 415

        data = request.get_json()
        if not data or 'prompt' not in data:
            logging.error("No prompt provided in request.")
            return jsonify({'error': 'No prompt provided'}), 400

        prompt = data['prompt']
        result = query_rag(prompt)
        
        if "error" in result:
            socketio.emit('error', {'error': result["error"]})
            return jsonify({'error': result["error"]}), 400

        socketio.emit('queryResult', result)
        logging.info(f"Query processed successfully: {prompt}")
        return jsonify(result)
    except ValueError as ve:
        logging.error(f"JSON parsing error: {ve}")
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        logging.error(f"Query failed: {str(e)}")
        socketio.emit('error', {'error': f'Query failed: {str(e)}'})
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/faces', methods=['GET'])
def get_registered_faces():
    """Retrieve all registered faces from MongoDB."""
    try:
        logging.info("Fetching registered faces from database...")
        faces = list(collection.find({"name": {"$ne": "No Faces Registered"}, "timestamp": {"$exists": True}}, {"_id": 0, "encoding": 0}))
        return jsonify({
            'faces': faces,
            'count': len(faces)
        }), 200
    except Exception as e:
        logging.error(f"Failed to fetch faces: {str(e)}")
        return jsonify({'error': f'Failed to fetch faces: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check system health and connectivity."""
    try:
        ping_result = client.admin.command('ping')
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'mongodb_connected': ping_result,
            'rag_initialized': qa_chain is not None,
            'dlib_cuda_enabled': dlib.DLIB_USE_CUDA,
            'cuda_available': torch.cuda.is_available()
        })
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@socketio.on('message')
def handle_message(data):
    """Handle WebSocket messages for face registration, recognition, queries, and deletion."""
    try:
        if data['type'] == 'registerFace':
            name = data['name']
            base64_image = data['image']

            image_data = base64.b64decode(base64_image.split(',')[1])
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            image = resize_image(image)
            image_np = np.array(image)

            face_locations = face_recognition.face_locations(image_np)
            if len(face_locations) != 1:
                logging.error(f"Detected {len(face_locations)} faces. Exactly one face required.")
                emit('error', {'error': 'Exactly one face should be detected'})
                return

            face_encodings = face_recognition.face_encodings(image_np, face_locations)
            encoding_list = face_encodings[0].tolist()

            if collection.find_one({"name": name}):
                logging.error(f'Name "{name}" already exists.')
                emit('error', {'error': f'Name "{name}" already exists'})
                return

            timestamp = datetime.now()
            result = collection.insert_one({
                "name": name,
                "encoding": encoding_list,
                "timestamp": timestamp.isoformat(),
                "created_at": timestamp
            })

            emit('registerResult', {
                'message': f'Successfully registered {name}',
                'id': str(result.inserted_id),
                'name': name,
                'timestamp': timestamp.isoformat(),
                'date': timestamp.strftime('%Y-%m-%d'),
                'day': timestamp.strftime('%A')
            })

            initialize_rag()
            logging.info(f"WebSocket: Successfully registered face: {name}")

        elif data['type'] == 'recognizeFace':
            base64_image = data['image']

            image_data = base64.b64decode(base64_image.split(',')[1])
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            image = resize_image(image)
            image_np = np.array(image)

            face_locations = face_recognition.face_locations(image_np)
            face_encodings = face_recognition.face_encodings(image_np, face_locations)

            if len(face_encodings) == 0:
                logging.info("WebSocket: No faces detected.")
                emit('recognizeResult', {'message': 'No faces detected', 'count': 0})
                return

            known_faces = list(collection.find({"name": {"$ne": "No Faces Registered"}, "encoding": {"$exists": True, "$ne": []}}, {'name': 1, 'encoding': 1}))
            if not known_faces:
                logging.error("WebSocket: No registered faces.")
                emit('error', {'error': 'No registered faces'})
                return

            results = []
            threshold = 0.6

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                name = 'Unknown'
                confidence = 0.0
                min_distance = float('inf')

                for known_face in known_faces:
                    try:
                        if 'encoding' not in known_face or not isinstance(known_face['encoding'], list):
                            continue
                        known_encoding = np.array(known_face['encoding'])
                        if known_encoding.shape != (128,):
                            continue
                        dist = distance.euclidean(encoding, known_encoding)
                        if dist < min_distance and dist < threshold:
                            min_distance = dist
                            name = known_face['name']
                            confidence = 1.0 - (dist / threshold)
                    except Exception:
                        continue

                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': [top, right, bottom, left]
                })

            emit('recognizeResult', {
                'faces': results,
                'count': len(results),
                'message': f'Detected {len(results)} face(s)'
            })
            logging.info(f"WebSocket: Recognized {len(results)} face(s).")

        elif data['type'] == 'query':
            prompt = data['prompt']
            result = query_rag(prompt)
            
            if "error" in result:
                logging.error(f"WebSocket: {result['error']}")
                emit('error', {'error': result['error']})
                return
            
            emit('queryResult', result)
            logging.info(f"WebSocket: Query processed: {prompt}")

        elif data['type'] == 'deleteFace':
            name = data['name']
            result = collection.delete_one({"name": name})
            if result.deleted_count == 0:
                logging.error(f"WebSocket: No face found with name: {name}")
                emit('error', {'error': f'No face found with name: {name}'})
                return

            initialize_rag()
            emit('deleteResult', {
                'message': f'Successfully deleted {name}',
                'timestamp': datetime.now().isoformat()
            })
            logging.info(f"WebSocket: Successfully deleted face: {name}")

    except Exception as e:
        logging.error(f"WebSocket failed: {str(e)}")
        emit('error', {'error': f'WebSocket failed: {str(e)}'})

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket client connection."""
    logging.info('WebSocket client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket client disconnection."""
    logging.info('WebSocket client disconnected')

if __name__ == '__main__':
    try:
        logging.debug("Checking system requirements...")
        check_system_requirements()
        logging.debug("Initializing RAG...")
        initialize_rag()
        logging.debug("Starting Flask app...")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Application startup failed: {str(e)}")
        raise
