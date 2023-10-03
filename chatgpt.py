import os
import sys
from flask import Flask, request, jsonify
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import constants
# Constants
os.environ["OPENAI_API_KEY"] = constants.APIKEY
PERSIST = False

# Initialize the Flask app
app = Flask(__name__)

# Define a global variable to hold the chat history
chat_history = []

# Create a conversation chain
def create_conversation_chain():
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = TextLoader("data/data.txt")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    return chain

# Route for handling conversation
@app.route('/conversation', methods=['POST'])
def handle_conversation():
    global chat_history

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'Missing query parameter'}), 400

    if query in ['quit', 'q', 'exit']:
        sys.exit()

    chain = create_conversation_chain()
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    return jsonify({'answer': result['answer']})

if __name__ == '__main__':
    app.run()
