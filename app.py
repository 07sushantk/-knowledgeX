# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import asyncio
import tempfile
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import time
from langchain_community.document_loaders import PyPDFLoader

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are Document RAG Assistant 📄🤖. 
Your role is to help users understand and explore the content of uploaded documents.

Follow these rules:
1. Always prioritize the document context when answering questions.
2. If the answer is not in the document, clearly say you don't know.
3. Keep responses friendly, clear, and concise.
"""

# Load environment variables
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # if "current_video_id" not in st.session_state:
    # st.session_state.current_video_id = None


def configure_page():
    st.set_page_config(
        page_title="KnowledgeX",
        page_icon="🧠",
        layout="wide",
    )

    # Modern header with gradient
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="color: white; font-size: 2.5rem; margin: 0;">🧠 KnowledgeX</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0.5rem 0 0 0;">Intelligent Document Conversations</p>
    </div>
    """, unsafe_allow_html=True)


def apply_modern_styling():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .main .block-container {
            max-width: 1200px;
            padding: 1rem;
        }
        
        /* Card styling */
        .custom-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
        }
        
        /* Status indicators */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
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
        
        .status-info {
            background: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        
        /* Hide Streamlit branding */
        footer {visibility: hidden;}
        
        /* File uploader styling */
        .stFileUploader {
            background: #f8fafc;
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
        }
        
        /* Metrics styling */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def handle_new_document_button():
    if st.sidebar.button("🔄 New Document", use_container_width=True):
        # Clear document-related session state
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        if "document_name" in st.session_state:
            del st.session_state["document_name"]

        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
        st.success("🔄 Ready for new document!")
        time.sleep(1)
        st.rerun()


def create_status_bar():
    """Create a modern status bar"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_status = "✅ Connected" if st.session_state.get("api_key") else "❌ No API Key"
        status_class = "status-success" if st.session_state.get("api_key") else "status-warning"
        st.markdown(f'<div class="status-pill {status_class}">🔑 {api_status}</div>', unsafe_allow_html=True)
    
    with col2:
        doc_status = "✅ Ready" if st.session_state.get("retriever") else "⏳ No Document"
        status_class = "status-success" if st.session_state.get("retriever") else "status-info"
        st.markdown(f'<div class="status-pill {status_class}">📄 {doc_status}</div>', unsafe_allow_html=True)
    
    with col3:
        msg_count = len(st.session_state.messages) - 1
        st.markdown(f'<div class="status-pill status-info">💬 {msg_count} Messages</div>', unsafe_allow_html=True)
    
    with col4:
        model_name = st.session_state.get("model", "gemini-2.5-pro").replace("gemini-", "").upper()
        st.markdown(f'<div class="status-pill status-info">🤖 {model_name}</div>', unsafe_allow_html=True)

def handle_sidebar():
    # Sidebar for API key
    st.sidebar.markdown("### 🔑 Configuration")

    api_key = st.sidebar.text_input(
        "Your Google Gemini API Key",
        type="password",
        placeholder="Enter your API key...",
        help="Your key is kept only in your current browser session.",
        value=st.session_state.get("api_key", ""),
    )
    if api_key:
        st.session_state.api_key = api_key
        if len(api_key) < 20:
            st.sidebar.error("⚠️ This API key looks too short. Please check it.")
        elif not api_key.startswith("AIza"):
            st.sidebar.warning(
                "⚠️ This doesn't look like a Google API key. Double-check it."
            )
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.sidebar.success("✅ API key set for this session")
    else:
        st.sidebar.info("💡 Enter your API key to start chatting")

    st.sidebar.divider()

    selected_model = st.sidebar.selectbox(
        "Generation Models",
        [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-image-preview",
            "gemini-live-2.5-flash-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-2.0-flash-live-001",
            "gemini-2.0-flash-live-preview-04-09",
            "gemini-2.0-flash-preview-image-generation",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        help="Choose the Gemini model for generation",
    )

    st.session_state.model = selected_model

    st.sidebar.divider()

    st.sidebar.subheader("💬 Chat Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()

    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    handle_new_document_button()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Info")

    message_count = len(st.session_state.messages) - 1  # Exclude system message
    document_processed = (
        "retriever" in st.session_state
        and st.session_state.get("retriever") is not None
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Messages", message_count)
    with col2:
        st.metric("Document", "✅" if document_processed else "❌")

    if document_processed:
        st.sidebar.success("📄  Document ready for chat")
    else:
        st.sidebar.info("📄  No Document processed yet")

    st.sidebar.info(f"**Current Model:**\n{selected_model}")

    if message_count > 0:
        st.sidebar.divider()
        chat_text = ""
        for msg in st.session_state.messages[1:]:  # Skip system message
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"

        st.sidebar.download_button(
            "📥 Download Chat",
            chat_text,
            f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True,
            help="Download your conversation history",
        )

    return selected_model, st.session_state.get("api_key")


def create_upload_section():
    """Create modern upload section"""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose your document",
        type=["pdf", "txt"],
        help="Upload a PDF or text file to chat with (Max: 100MB)"
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
                handle_document_processing(uploaded_file)
    
    st.markdown('</div>', unsafe_allow_html=True)
    return uploaded_file

def handle_document_processing(uploaded_file):
    if True:
        user_api_key = st.session_state.get("api_key", "")
        if not user_api_key:
            st.error("❌ Please enter your Google Gemini API key in the sidebar first!")
            return
        else:
            with st.spinner("Processing document..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:

                    # Step 1: Extract transcript
                    status_text.text("🔄 Step 1/4: Saving document...")
                    progress_bar.progress(25)

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    status_text.text("📄 Step 2/4: Loading document...")
                    progress_bar.progress(50)

                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                    else:  # txt file
                        loader = TextLoader(tmp_file_path)

                    documents = loader.load()

                    status_text.text("✂️ Step 3/4: Splitting into chunks...")
                    progress_bar.progress(75)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    chunks = splitter.split_documents(documents)

                    status_text.text("🧠 Step 4/4: Creating embeddings...")
                    progress_bar.progress(100)
                    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
                    )

                    st.session_state["retriever"] = retriever
                    st.session_state["document_name"] = uploaded_file.name

                    os.unlink(tmp_file_path)
                    progress_bar.empty()
                    status_text.empty()

                    st.success("✅ Document processed! Ready for questions.")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


@st.cache_resource()
def get_chat_model(model_name: str, api_key_keyed_for_cache: str | None):
    # api_key_keyed_for_cache is unused except for cache key isolation across different keys
    return ChatGoogleGenerativeAI(model=model_name)


def create_chat_section():
    """Create modern chat section"""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 💬 AI Conversation")
    
    if len(st.session_state.messages) <= 1:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6b7280;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
            <h3>Ready to Chat!</h3>
            <p>Upload a document and ask me anything about its content.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages[1:]:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_action_buttons():
    """Create action buttons"""
    if len(st.session_state.messages) > 1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
                st.rerun()
        
        with col2:
            if st.button("🔄 New Document", use_container_width=True):
                for key in ["retriever", "document_name"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
                st.success("Ready for new document!")
                time.sleep(1)
                st.rerun()
        
        with col3:
            chat_text = ""
            for msg in st.session_state.messages[1:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                chat_text += f"{role}: {msg.content}\n\n"
            
            st.download_button(
                "📥 Export Chat",
                chat_text,
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )


def handle_user_input(chat_model, input_disabled: bool = False):
    if prompt := st.chat_input(
        "Ask a question about the document...", disabled=input_disabled
    ):
        if not prompt.strip():
            st.warning("Please type a message before sending!")
            return

        st.session_state.messages.append(HumanMessage(content=prompt))

        prompt_template = PromptTemplate(
            template="""Based on this document content:

            {context}

            Question: {question}""",
            input_variables=["context", "question"],
        )

        with st.chat_message("user"):
            st.write(prompt)

        retriever = st.session_state.get("retriever")
        if not retriever:
            with st.chat_message("assistant"):
                error_msg = (
                    "❌ Please process a document first to enable question answering."
                )
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
            return
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing document content..."):
                try:
                    retrieved_docs = retriever.invoke(prompt)
                    if not retrieved_docs:
                        no_context_msg = "🤷‍♂️ I couldn't find relevant information in the document for your question."
                        st.warning(no_context_msg)
                        st.session_state.messages.append(
                            AIMessage(content=no_context_msg)
                        )
                        return
                    parallel_chain = RunnableParallel(
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                    )
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt_template | chat_model | parser

                    message_placeholder = st.empty()
                    full_response = ""

                    # Stream the response using stream method (synchronous)
                    for chunk in main_chain.stream(prompt):
                        if chunk and chunk.strip():
                            full_response += chunk
                            message_placeholder.markdown(
                                full_response + "▌"
                            )  # Cursor indicator

                    # Remove cursor and display final response
                    if full_response and full_response.strip():
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append(
                            AIMessage(content=full_response)
                        )
                    else:
                        error_msg = (
                            "🚫 No response received. Please try a different model."
                        )
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append(AIMessage(content=error_msg))

                    # Rerun to refresh the UI after streaming
                    st.rerun()

                except Exception as e:
                    error_message = str(e).lower()
                    if "not found" in error_message or "invalid" in error_message:
                        error_msg = "❌ This model is not available. Please select a different model."
                    elif "quota" in error_message or "limit" in error_message:
                        error_msg = "📊 API quota exceeded. Please try again later or use a different model."
                    elif "timeout" in error_message:
                        error_msg = (
                            "⏱️ Request timed out. Try a different model or try again."
                        )
                    else:
                        error_msg = f"❌ An error occurred. Try selecting different model or check your api key:("

                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
            # st.rerun()


# Main app execution
init_session_state()
configure_page()
apply_modern_styling()

# Status bar
create_status_bar()
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
selected_model, user_api_key = handle_sidebar()

# Main layout
col1, col2 = st.columns([1, 1.5])

with col1:
    uploaded_file = create_upload_section()
    st.markdown("<br>", unsafe_allow_html=True)
    create_action_buttons()

with col2:
    create_chat_section()
    
    # Chat model setup
    chat_model = None
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
        chat_model = get_chat_model(selected_model, user_api_key)
    
    # Handle chat input
    if chat_model is None:
        st.info("🔑 Please enter your API key in the sidebar to start chatting")
    elif not st.session_state.get("retriever"):
        st.info("📄 Please upload and process a document first")
    else:
        handle_user_input(chat_model, input_disabled=False)
