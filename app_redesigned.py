import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import asyncio
import tempfile
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
RETRIEVER_K = 4

# Configure page
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4ade80;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --text-color: #1f2937;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: var(--bg-gradient);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        max-width: 80%;
    }
    
    .user-message {
        background: var(--bg-gradient);
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: var(--text-color);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--bg-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: var(--bg-gradient);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-text {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

def create_header():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Document AI Assistant</h1>
        <p>Transform your documents into intelligent conversations</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    with st.sidebar:
        st.markdown("### 🔑 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Enter your API key...",
            help="Get your API key from Google AI Studio"
        )
        
        if api_key:
            st.session_state.api_key = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
            if len(api_key) > 20 and api_key.startswith("AIza"):
                st.success("✅ API key configured")
            else:
                st.error("❌ Invalid API key format")
        
        st.divider()
        
        # Model selection
        st.markdown("### 🧠 AI Model")
        model = st.selectbox(
            "Choose Model",
            [
                "gemini-2.5-pro",
                "gemini-2.5-flash", 
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash"
            ],
            help="Select the AI model for responses"
        )
        st.session_state.model = model
        
        st.divider()
        
        # Session info
        st.markdown("### 📊 Session Status")
        
        col1, col2 = st.columns(2)
        with col1:
            doc_status = "✅" if st.session_state.document_processed else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem;">{doc_status}</div>
                <div style="font-size: 0.8rem; color: #6b7280;">Document</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            msg_count = len(st.session_state.messages)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem;">{msg_count}</div>
                <div style="font-size: 0.8rem; color: #6b7280;">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Controls
        st.markdown("### 🎛️ Controls")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 New Document", use_container_width=True):
            for key in ["retriever", "document_name", "document_processed"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.document_processed = False
            st.session_state.messages = []
            st.success("Ready for new document!")
            time.sleep(1)
            st.rerun()
        
        # Export chat
        if st.session_state.messages:
            st.divider()
            chat_text = ""
            for msg in st.session_state.messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                chat_text += f"{role}: {msg.content}\n\n"
            
            st.download_button(
                "📥 Export Chat",
                chat_text,
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

def create_upload_section():
    st.markdown("""
    <div class="custom-card">
        <h3 style="margin-top: 0;">📁 Upload Document</h3>
        <p style="color: #6b7280; margin-bottom: 1rem;">
            Upload a PDF or text file to start chatting with your document
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT (Max size: 100MB)"
    )
    
    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        st.markdown(f"""
        <div class="status-indicator status-success">
            📄 {uploaded_file.name} ({file_size:.1f} MB)
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            process_document(uploaded_file)
    
    return uploaded_file

def process_document(uploaded_file):
    if not st.session_state.get("api_key"):
        st.error("❌ Please enter your API key first!")
        return
    
    with st.spinner("Processing document..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save file
            status_text.markdown('<p class="loading-text">💾 Saving document...</p>', unsafe_allow_html=True)
            progress_bar.progress(25)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load document
            status_text.markdown('<p class="loading-text">📖 Loading content...</p>', unsafe_allow_html=True)
            progress_bar.progress(50)
            
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path)
            
            documents = loader.load()
            
            # Split into chunks
            status_text.markdown('<p class="loading-text">✂️ Processing chunks...</p>', unsafe_allow_html=True)
            progress_bar.progress(75)
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(documents)
            
            # Create embeddings
            status_text.markdown('<p class="loading-text">🧠 Creating embeddings...</p>', unsafe_allow_html=True)
            progress_bar.progress(100)
            
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
            
            st.session_state.retriever = retriever
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_processed = True
            
            os.unlink(tmp_file_path)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Document processed successfully!")
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def display_chat():
    if not st.session_state.messages:
        st.markdown("""
        <div class="custom-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #6b7280;">👋 Welcome!</h3>
            <p style="color: #9ca3af;">Upload a document and start asking questions about its content.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

def handle_chat_input():
    if not st.session_state.get("api_key"):
        st.warning("⚠️ Please enter your API key to start chatting")
        return
    
    if not st.session_state.document_processed:
        st.info("📄 Please upload and process a document first")
        return
    
    if prompt := st.chat_input("Ask me anything about your document..."):
        # Add user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get relevant documents
                    retriever = st.session_state.retriever
                    retrieved_docs = retriever.invoke(prompt)
                    
                    if not retrieved_docs:
                        response = "I couldn't find relevant information in the document."
                        st.write(response)
                        st.session_state.messages.append(AIMessage(content=response))
                        return
                    
                    # Create context
                    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    
                    # Create prompt template
                    template = """Based on the following document content, answer the question:

Context: {context}

Question: {question}

Answer:"""
                    
                    prompt_template = PromptTemplate(
                        template=template,
                        input_variables=["context", "question"]
                    )
                    
                    # Get chat model
                    chat_model = ChatGoogleGenerativeAI(model=st.session_state.model)
                    
                    # Create chain
                    chain = prompt_template | chat_model | StrOutputParser()
                    
                    # Stream response
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in chain.stream({"context": context, "question": prompt}):
                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append(AIMessage(content=full_response))
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))

# Main app
def main():
    init_session_state()
    create_header()
    create_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = create_upload_section()
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="margin-top: 0;">💬 Chat with Document</h3>
        </div>
        """, unsafe_allow_html=True)
        
        display_chat()
        handle_chat_input()

if __name__ == "__main__":
    main()