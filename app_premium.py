import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import tempfile
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
import time
import json

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS with glassmorphism and modern animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding: 1rem;
        max-width: 1400px;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Header with floating effect */
    .floating-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .floating-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .floating-header p {
        color: #64748b;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Status pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 500;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        color: #059669;
        border-color: rgba(34, 197, 94, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #d97706;
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    .status-info {
        background: rgba(59, 130, 246, 0.2);
        color: #2563eb;
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    /* Upload zone with drag & drop styling */
    .upload-zone {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .upload-zone:hover {
        border-color: rgba(255, 255, 255, 0.5);
        background: rgba(255, 255, 255, 0.15);
        transform: scale(1.02);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    
    /* Chat messages with better styling */
    .chat-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Floating action buttons */
    .fab {
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* Animated buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 200%;
        animation: gradient 2s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Metrics cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 2s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Chat input */
    .stChatInput > div > div > div > div > div > textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        color: white;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    defaults = {
        "messages": [],
        "document_processed": False,
        "api_key": "",
        "model": "gemini-2.5-flash",
        "processing": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_header():
    st.markdown("""
    <div class="floating-header">
        <h1>🧠 DocuMind AI</h1>
        <p>Intelligent Document Conversations Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

def create_status_bar():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_status = "✅ Connected" if st.session_state.api_key else "❌ No API Key"
        status_class = "status-success" if st.session_state.api_key else "status-warning"
        st.markdown(f'<div class="status-pill {status_class}">🔑 {api_status}</div>', unsafe_allow_html=True)
    
    with col2:
        doc_status = "✅ Ready" if st.session_state.document_processed else "⏳ No Document"
        status_class = "status-success" if st.session_state.document_processed else "status-info"
        st.markdown(f'<div class="status-pill {status_class}">📄 {doc_status}</div>', unsafe_allow_html=True)
    
    with col3:
        msg_count = len(st.session_state.messages)
        st.markdown(f'<div class="status-pill status-info">💬 {msg_count} Messages</div>', unsafe_allow_html=True)
    
    with col4:
        model_name = st.session_state.model.replace("gemini-", "").upper()
        st.markdown(f'<div class="status-pill status-info">🤖 {model_name}</div>', unsafe_allow_html=True)

def create_config_panel():
    with st.expander("⚙️ Configuration", expanded=not st.session_state.api_key):
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "🔑 Google Gemini API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="Enter your API key..."
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                if api_key:
                    os.environ["GOOGLE_API_KEY"] = api_key
        
        with col2:
            model = st.selectbox(
                "🤖 AI Model",
                ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"],
                index=0
            )
            st.session_state.model = model

def create_upload_section():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "txt"],
        help="Supported: PDF, TXT files (Max: 100MB)"
    )
    
    if uploaded_file:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        <div class="status-pill status-success">
            📄 {uploaded_file.name} ({size_mb:.1f} MB)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                if not st.session_state.api_key:
                    st.error("❌ Please enter your API key first!")
                else:
                    process_document(uploaded_file)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return uploaded_file

def process_document(uploaded_file):
    st.session_state.processing = True
    
    with st.spinner(""):
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            steps = [
                ("💾 Saving document", 25),
                ("📖 Loading content", 50), 
                ("✂️ Creating chunks", 75),
                ("🧠 Building knowledge base", 100)
            ]
            
            for step_text, progress in steps:
                status_text.markdown(f'<p style="text-align:center; color:white;">{step_text}<span class="loading-dots"></span></p>', unsafe_allow_html=True)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            # Process document
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path) if uploaded_file.name.endswith(".pdf") else TextLoader(tmp_file_path)
            documents = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            st.session_state.document_processed = True
            
            os.unlink(tmp_file_path)
            progress_container.empty()
            
            st.success("✅ Document processed successfully!")
            st.balloons()
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            st.session_state.processing = False

def create_chat_section():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💬 AI Conversation")
    
    # Chat container
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.7);">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <h3>Ready to Chat!</h3>
                <p>Upload a document and ask me anything about its content.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(message.content)
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_chat():
    if not st.session_state.api_key:
        st.info("🔑 Please enter your API key to start chatting")
        return
    
    if not st.session_state.document_processed:
        st.info("📄 Please upload and process a document first")
        return
    
    if prompt := st.chat_input("Ask me anything about your document..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            try:
                retriever = st.session_state.retriever
                retrieved_docs = retriever.invoke(prompt)
                
                if not retrieved_docs:
                    response = "I couldn't find relevant information in the document."
                    st.write(response)
                    st.session_state.messages.append(AIMessage(content=response))
                    return
                
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                template = """Based on this document content, provide a helpful and accurate answer:

Context: {context}

Question: {question}

Answer:"""
                
                prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
                chat_model = ChatGoogleGenerativeAI(model=st.session_state.model)
                chain = prompt_template | chat_model | StrOutputParser()
                
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

def create_action_buttons():
    if st.session_state.messages:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 New Document", use_container_width=True):
                st.session_state.document_processed = False
                st.session_state.messages = []
                if "retriever" in st.session_state:
                    del st.session_state.retriever
                st.success("Ready for new document!")
                time.sleep(1)
                st.rerun()
        
        with col3:
            chat_text = "\n\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in st.session_state.messages])
            st.download_button(
                "📥 Export Chat",
                chat_text,
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

def main():
    init_session_state()
    create_header()
    create_status_bar()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    create_config_panel()
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        create_upload_section()
        st.markdown("<br>", unsafe_allow_html=True)
        create_action_buttons()
    
    with col2:
        create_chat_section()
        handle_chat()

if __name__ == "__main__":
    main()