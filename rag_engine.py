import pymongo
import sys

# MongoDB setup
client = pymongo.Mongoclient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["faces"]

# Mock LLM response (replace with actual OpenAI API call)
def mock_llm(query, context):
    return f"Based on the data: {context}, the answer to '{query}' is mocked for now."

def answer_query(query):
    if "last person registered" in query.lower():
        result = collection.find().sort("timestamp", -1).limit(1)
        result = list(result)[0] if result else None
        if result:
            context = f"Last person: {result['name']} at {result['timestamp']}"
            print(mock_llm(query, context))
    
    elif "how many people" in query.lower():
        count = collection.count_documents({})
        context = f"Total registered: {count}"
        print(mock_llm(query, context))
    
    elif "at what time" in query.lower():
        for doc in collection.find():
            if doc["name"].lower() in query.lower():
                context = f"{doc['name']} was registered at {doc['timestamp']}"
                print(mock_llm(query, context))
                break
        else:
            print("Person not found")

if __name__ == "__main__":
    query = sys.argv[1]
    answer_query(query)