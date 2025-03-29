import os
import openai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime

# API key setup
api_key = "sk-5ec02f12083d4f97aad73e1adb3a6f48"
if not api_key:
    api_key = input("Please enter your DeepSeek API key: ")
    os.environ["DEEPSEEK_API_KEY"] = api_key

# Configure the client for completions
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"  # DeepSeek API endpoint
)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # A lightweight, effective embedding model
)

# Create or load vector database
CHROMA_DIRECTORY = "chroma_db"
if os.path.exists(CHROMA_DIRECTORY):
    # Load existing database
    vector_db = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)
else:
    # Create new database
    vector_db = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)

# Conversation history storage
HISTORY_FILE = "conversation_history.json"
conversation_history = []

# Load existing conversation history if available
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, 'r') as f:
            conversation_history = json.load(f)
    except Exception as e:
        print(f"Error loading conversation history: {str(e)}")

# Function to save conversation history
def save_conversation_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(conversation_history, f)

# Function to add conversation turn to history
def add_to_history(role, content):
    timestamp = datetime.now().isoformat()
    conversation_history.append({
        "timestamp": timestamp,
        "role": role,
        "content": content
    })
    save_conversation_history()
    
    # Also add to vector database for retrieval
    # Create a special document format for conversation turns
    doc_content = f"[{timestamp}] {role}: {content}"
    vector_db.add_documents([Document(page_content=doc_content, metadata={"type": "conversation", "role": role, "timestamp": timestamp})])
    vector_db.persist()

# Function to add documents to the vector database
def add_documents(texts, metadata=None):
    if metadata is None:
        metadata = [{"source": f"doc_{i}", "type": "document"} for i in range(len(texts))]
    
    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []
    
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata=metadata[i]) for chunk in chunks]
        all_chunks.extend(docs)
    
    # Add to vector database
    vector_db.add_documents(all_chunks)
    vector_db.persist()  # Save to disk
    return len(all_chunks)

# Function to query the vector database
def query_vector_db(query, k=5):
    results = vector_db.similarity_search(query, k=k)
    return results

# Function to get formatted context from conversation history
def get_conversation_context(recent_turns=10):
    # Get the most recent turns from conversation history
    recent_history = conversation_history[-recent_turns:] if len(conversation_history) > recent_turns else conversation_history
    
    context = "Recent conversation history:\n"
    for turn in recent_history:
        context += f"{turn['role']}: {turn['content']}\n"
    
    return context

# Initialize message history for the current API call
messages = [
    {"role": "system", "content": "You are a helpful assistant with access to a knowledge base and conversation history. Pay close attention to any information shared earlier in the conversation, as it may be important context for the current query."}
]

print("Interactive DeepSeek Chat with Vector-Stored Conversation History")
print("----------------------------------------------------------------")
print("Commands: 'add' to add document, 'history' to view history, 'exit' to quit")

while True:
    user_input = input("\nYou: ")
    
    # Check commands
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("\nGoodbye!")
        save_conversation_history()
        break
        
    elif user_input.lower() == 'history':
        print("\nConversation History:")
        for i, turn in enumerate(conversation_history[-10:]):  # Show last 10 turns
            print(f"[{i+1}] {turn['role']}: {turn['content']}")
        continue
        
    elif user_input.lower().startswith('add '):
        # Format: add <filename>
        filename = user_input[4:].strip()
        try:
            with open(filename, 'r') as file:
                content = file.read()
            
            chunks_added = add_documents([content], [{"source": filename, "type": "document"}])
            print(f"Added {chunks_added} chunks from {filename} to the vector database.")
            
            # Add this interaction to history
            add_to_history("user", f"Added document: {filename}")
            continue
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            continue
    
    # Add user input to history
    add_to_history("user", user_input)
    
    # Regular query with historical context
    try:
        # Retrieve relevant context from vector DB that might be relevant to this query
        context_docs = query_vector_db(user_input)
        
        # Separate document context and conversation context
        doc_context = ""
        conv_context = ""
        
        for doc in context_docs:
            if doc.metadata.get("type") == "conversation":
                conv_context += f"{doc.page_content}\n"
            else:
                doc_context += f"Document from {doc.metadata.get('source', 'unknown')}: {doc.page_content}\n\n"
        
        # Also add the most recent conversation turns
        recent_conv_context = get_conversation_context(5)  # Get last 5 turns
        
        # Combine all context
        full_context = f"Recent conversation history:\n{recent_conv_context}\n\nRelevant past conversations:\n{conv_context}\n\nRelevant documents:\n{doc_context}\n\nUser query: {user_input}"
        
        # Add context to messages for this API call
        messages.append({"role": "user", "content": full_context})
        
        # Stream the response
        print("\nDeepSeek: ", end="", flush=True)
        stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True
        )
        
        assistant_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_chunk = chunk.choices[0].delta.content
                assistant_response += content_chunk
                print(content_chunk, end="", flush=True)
        print()
        
        # Add assistant response to history
        add_to_history("assistant", assistant_response)
        
        # Reset messages for next turn
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to a knowledge base and conversation history. Pay close attention to any information shared earlier in the conversation, as it may be important context for the current query."}
        ]
        
    except Exception as e:
        print(f"\nError: {str(e)}")