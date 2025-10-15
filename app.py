import os
import base64
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from PIL import Image
import PyPDF2
import docx
import random
# ========== MODEL CONFIGURATION ==========
OLLAMA_CHAT_MODEL = "deepseek-r1:1.5b"
OLLAMA_VISION_MODEL = "llama3.2-vision"
WHISPER_MODEL_SIZE = "base"
# =========================================

# ========== PRE-LOAD CONFIGURATION ==========
PRELOAD_TEXT_DIR = "preload_data/text"
PRELOAD_IMAGE_DIR = "preload_data/images"
PRELOAD_AUDIO_DIR = "preload_data/audio"
# ===========================================

# Audio support
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their content from the text"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()


class MultimodalRAG:
    def __init__(self, db_path: str = "./rag_db"):
        """Initialize the enhanced multimodal RAG system"""
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Cache embedding model
        if 'embedding_model' not in st.session_state:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model = st.session_state.embedding_model
        
        # Cache Whisper model
        if WHISPER_AVAILABLE and 'whisper_model' not in st.session_state:
            with st.spinner("Loading Whisper model..."):
                st.session_state.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        
        if WHISPER_AVAILABLE:
            self.whisper_model = st.session_state.whisper_model
        
        # Single unified collection for semantic search
        self.collection = self.client.get_or_create_collection(
            name="multimodal_docs",
            metadata={"description": "Unified semantic space for all modalities"}
        )
    
    def _embed(self, text: str) -> List[float]:
        """Generate embedding for semantic search"""
        return self.embedding_model.encode(text).tolist()
    
    def _describe_image(self, image_path: str) -> str:
        """Describe image using Ollama Vision with enhanced detail"""
        try:
            with open(image_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = ollama.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in comprehensive detail including: objects, people, text visible in the image, colors, setting, context, any dates or timestamps visible, and any other identifying information. Be thorough and specific.',
                    'images': [img_data]
                }]
            )
            content = response['message']['content']
            return remove_think_tags(content)
        except Exception as e:
            st.error(f"Error describing image: {e}")
            return f"Image: {os.path.basename(image_path)}"
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio with timestamps using Whisper"""
        if not WHISPER_AVAILABLE:
            return {
                'text': f"Audio: {os.path.basename(audio_path)}",
                'segments': []
            }
        
        try:
            result = self.whisper_model.transcribe(audio_path, verbose=False)
            return {
                'text': result["text"],
                'segments': result.get("segments", [])
            }
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return {
                'text': f"Audio: {os.path.basename(audio_path)}",
                'segments': []
            }
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF with page numbers"""
        try:
            text_by_page = []
            full_text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text_by_page.append({
                        'page': page_num,
                        'content': page_text
                    })
                    full_text += f"\n[Page {page_num}]\n{page_text}"
            
            return {
                'full_text': full_text.strip(),
                'pages': text_by_page,
                'total_pages': len(text_by_page)
            }
        except Exception as e:
            st.error(f"Error extracting PDF: {e}")
            return {
                'full_text': f"PDF: {os.path.basename(pdf_path)}",
                'pages': [],
                'total_pages': 0
            }
    
    def _extract_text_from_docx(self, docx_path: str) -> Dict[str, Any]:
        """Extract text from DOCX with paragraph tracking"""
        try:
            doc = docx.Document(docx_path)
            paragraphs = []
            full_text = ""
            
            for i, paragraph in enumerate(doc.paragraphs, 1):
                if paragraph.text.strip():
                    paragraphs.append({
                        'paragraph': i,
                        'content': paragraph.text
                    })
                    full_text += f"{paragraph.text}\n"
            
            return {
                'full_text': full_text.strip(),
                'paragraphs': paragraphs,
                'total_paragraphs': len(paragraphs)
            }
        except Exception as e:
            st.error(f"Error extracting DOCX: {e}")
            return {
                'full_text': f"DOCX: {os.path.basename(docx_path)}",
                'paragraphs': [],
                'total_paragraphs': 0
            }
    
    def add_text(self, text: str, metadata: Dict = None) -> str:
        """Add text document with enhanced metadata"""
        doc_id = f"text_{int(time.time() * 1000)}_{self.collection.count()}"
        meta = metadata or {}
        meta.update({
            'type': 'text',
            'added_at': datetime.now().isoformat(),
            'content_preview': text[:200]
        })
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._embed(text)],
            documents=[text],
            metadatas=[meta]
        )
        return doc_id
    
    def add_image(self, path: str, metadata: Dict = None) -> Tuple[str, str]:
        """Add image with comprehensive description and metadata"""
        doc_id = f"image_{int(time.time() * 1000)}_{self.collection.count()}"
        
        description = self._describe_image(path)
        
        # Get image metadata
        try:
            img = Image.open(path)
            img_metadata = {
                'dimensions': f"{img.width}x{img.height}",
                'format': img.format
            }
        except:
            img_metadata = {}
        
        meta = metadata or {}
        meta.update({
            'type': 'image',
            'path': path,
            'filename': os.path.basename(path),
            'added_at': datetime.now().isoformat(),
            'description_preview': description[:200],
            **img_metadata
        })
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._embed(description)],
            documents=[description],
            metadatas=[meta]
        )
        return doc_id, description
    
    def add_audio(self, path: str, metadata: Dict = None) -> Tuple[str, str, List[Dict]]:
        """Add audio with transcription and timestamps"""
        doc_id = f"audio_{int(time.time() * 1000)}_{self.collection.count()}"
        
        transcription_data = self._transcribe_audio(path)
        transcription = transcription_data['text']
        segments = transcription_data['segments']
        
        meta = metadata or {}
        meta.update({
            'type': 'audio',
            'path': path,
            'filename': os.path.basename(path),
            'added_at': datetime.now().isoformat(),
            'transcription_preview': transcription[:200],
            'has_timestamps': len(segments) > 0
        })
        
        # Store segments as JSON string
        if segments:
            meta['segments_json'] = json.dumps(segments[:10])  # Store first 10 segments
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._embed(transcription)],
            documents=[transcription],
            metadatas=[meta]
        )
        return doc_id, transcription, segments
    
    def add_pdf(self, path: str, metadata: Dict = None) -> Tuple[str, str]:
        """Add PDF with page-level tracking"""
        pdf_data = self._extract_text_from_pdf(path)
        
        doc_id = f"pdf_{int(time.time() * 1000)}_{self.collection.count()}"
        meta = metadata or {}
        meta.update({
            'type': 'pdf',
            'path': path,
            'filename': os.path.basename(path),
            'added_at': datetime.now().isoformat(),
            'total_pages': pdf_data['total_pages'],
            'content_preview': pdf_data['full_text'][:200]
        })
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._embed(pdf_data['full_text'])],
            documents=[pdf_data['full_text']],
            metadatas=[meta]
        )
        return doc_id, pdf_data['full_text']
    
    def add_docx(self, path: str, metadata: Dict = None) -> Tuple[str, str]:
        """Add DOCX with paragraph-level tracking"""
        docx_data = self._extract_text_from_docx(path)
        
        doc_id = f"docx_{int(time.time() * 1000)}_{self.collection.count()}"
        meta = metadata or {}
        meta.update({
            'type': 'docx',
            'path': path,
            'filename': os.path.basename(path),
            'added_at': datetime.now().isoformat(),
            'total_paragraphs': docx_data['total_paragraphs'],
            'content_preview': docx_data['full_text'][:200]
        })
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._embed(docx_data['full_text'])],
            documents=[docx_data['full_text']],
            metadatas=[meta]
        )
        return doc_id, docx_data['full_text']
    
    def search_by_text(self, query: str, n: int = 5) -> List[Dict]:
        """Text-to-multimodal search"""
        if self.collection.count() == 0:
            return []
        
        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(n, self.collection.count())
        )
        
        return self._format_results(results)
    
    def search_by_image(self, image_path: str, n: int = 5) -> List[Dict]:
        """Image-to-multimodal search"""
        if self.collection.count() == 0:
            return []
        
        # Get image description
        description = self._describe_image(image_path)
        
        results = self.collection.query(
            query_embeddings=[self._embed(description)],
            n_results=min(n, self.collection.count())
        )
        
        return self._format_results(results)
    
    def search_by_audio(self, audio_path: str, n: int = 5) -> List[Dict]:
        """Audio-to-multimodal search"""
        if self.collection.count() == 0:
            return []
        
        # Get audio transcription
        transcription_data = self._transcribe_audio(audio_path)
        transcription = transcription_data['text']
        
        results = self.collection.query(
            query_embeddings=[self._embed(transcription)],
            n_results=min(n, self.collection.count())
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results consistently"""
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': results['distances'][0][i],
                    'relevance': 1 / (1 + results['distances'][0][i])
                })
        return formatted
    
    def query(self, question: str, n: int = 5, search_mode: str = "text") -> Tuple[str, List[Dict]]:
        """
        Enhanced query with citation transparency
        search_mode: 'text', 'image', or 'audio'
        """
        # Perform search based on mode
        if search_mode == "text":
            results = self.search_by_text(question, n)
        else:
            results = []
        
        if not results:
            return "No relevant documents found in the database. Please add documents first.", []
        
        # Build context with numbered citations
        context_parts = []
        for i, r in enumerate(results, 1):
            doc_type = r['metadata']['type'].upper()
            filename = r['metadata'].get('filename', 'Unknown')
            
            citation = f"[Citation {i}: {doc_type} - {filename}]"
            
            # Add specific location info if available
            if 'total_pages' in r['metadata']:
                citation += f" (Pages: {r['metadata']['total_pages']})"
            elif 'has_timestamps' in r['metadata'] and r['metadata']['has_timestamps']:
                citation += " (Timestamped audio)"
            
            context_parts.append(f"{citation}\n{r['content'][:500]}...")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant for a document retrieval system. Answer the question using the provided sources.

IMPORTANT INSTRUCTIONS:
1. Use numbered citations [1], [2], etc. when referencing information from sources
2. Be specific about which source you're using
3. If information comes from multiple sources, cite all relevant ones
4. Keep your answer clear and concise
5. If the sources don't contain enough information, say so

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""
        
        response = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        content = response['message']['content']
        return remove_think_tags(content), results
    
    def get_stats(self) -> Dict:
        """Get detailed database statistics"""
        all_docs = self.collection.get()
        
        stats = {
            'total': self.collection.count(),
            'text': 0,
            'image': 0,
            'audio': 0,
            'pdf': 0,
            'docx': 0
        }
        
        if all_docs['metadatas']:
            for meta in all_docs['metadatas']:
                doc_type = meta.get('type', 'unknown')
                if doc_type in stats:
                    stats[doc_type] += 1
        
        return stats
    
    def get_document_details(self, doc_id: str) -> Optional[Dict]:
        """Get full details of a specific document"""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except:
            pass
        return None
    
    def list_all_documents(self) -> List[Dict]:
        """List all documents with metadata"""
        all_docs = self.collection.get()
        documents = []
        
        if all_docs['ids']:
            for i in range(len(all_docs['ids'])):
                documents.append({
                    'id': all_docs['ids'][i],
                    'type': all_docs['metadatas'][i].get('type', 'unknown'),
                    'filename': all_docs['metadatas'][i].get('filename', 'Unknown'),
                    'added_at': all_docs['metadatas'][i].get('added_at', 'Unknown'),
                    'preview': all_docs['documents'][i][:100] + "..."
                })
        
        return documents
    
    def delete_all(self):
        """Delete all documents"""
        self.client.delete_collection("multimodal_docs")
        self.collection = self.client.get_or_create_collection(
            name="multimodal_docs",
            metadata={"description": "Unified semantic space for all modalities"}
        )


def preload_documents(rag: MultimodalRAG) -> Dict[str, int]:
    """Pre-load documents from specified directories"""
    loaded = {'text': 0, 'pdf': 0, 'docx': 0, 'image': 0, 'audio': 0}
    
    # Create directories
    Path(PRELOAD_TEXT_DIR).mkdir(parents=True, exist_ok=True)
    Path(PRELOAD_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    Path(PRELOAD_AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load text files
    text_dir = Path(PRELOAD_TEXT_DIR)
    if text_dir.exists():
        for txt_file in text_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    rag.add_text(content, {'source': 'preloaded', 'filename': txt_file.name})
                    loaded['text'] += 1
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
        # Load PDFs
        for pdf_file in text_dir.glob("*.pdf"):
            try:
                doc_id, content = rag.add_pdf(str(pdf_file), {'source': 'preloaded'})
                if not content.startswith("PDF:"):
                    loaded['pdf'] += 1
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
        
        # Load DOCX
        for docx_file in text_dir.glob("*.docx"):
            try:
                doc_id, content = rag.add_docx(str(docx_file), {'source': 'preloaded'})
                if not content.startswith("DOCX:"):
                    loaded['docx'] += 1
            except Exception as e:
                print(f"Error loading {docx_file}: {e}")
    
    # Load images
    image_dir = Path(PRELOAD_IMAGE_DIR)
    if image_dir.exists():
        for img_file in image_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                try:
                    rag.add_image(str(img_file), {'source': 'preloaded'})
                    loaded['image'] += 1
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
    
    # Load audio
    audio_dir = Path(PRELOAD_AUDIO_DIR)
    if audio_dir.exists() and WHISPER_AVAILABLE:
        for audio_file in audio_dir.glob("*"):
            if audio_file.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac']:
                try:
                    rag.add_audio(str(audio_file), {'source': 'preloaded'})
                    loaded['audio'] += 1
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
    
    return loaded


def setup_page():
    """Setup Streamlit page"""
    st.set_page_config(
        page_title="Multimodal RAG System - SIH",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main {padding: 1rem;}
        .stButton>button {width: 100%;}
        .citation-box {
            padding: 1rem;
            border-left: 4px solid #4CAF50;
            background-color: #f0f8f0;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
        .source-card {
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 0.5rem 0;
            background-color: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'rag' not in st.session_state:
        st.session_state.rag = MultimodalRAG()
        
        if 'preload_done' not in st.session_state:
            if st.session_state.rag.get_stats()['total'] == 0:
                with st.spinner("🔄 Pre-loading documents..."):
                    loaded_counts = preload_documents(st.session_state.rag)
                    total_loaded = sum(loaded_counts.values())
                    
                    if total_loaded > 0:
                        st.success(
                            f"✅ Pre-loaded {total_loaded} documents: "
                            f"{loaded_counts['text']} text, "
                            f"{loaded_counts['pdf']} PDFs, "
                            f"{loaded_counts['docx']} DOCX, "
                            f"{loaded_counts['image']} images, "
                            f"{loaded_counts['audio']} audio"
                        )
            st.session_state.preload_done = True
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = 'text'
    
    if 'data_dir' not in st.session_state:
        Path("data/images").mkdir(parents=True, exist_ok=True)
        Path("data/audio").mkdir(parents=True, exist_ok=True)
        Path("data/documents").mkdir(parents=True, exist_ok=True)
        st.session_state.data_dir = "data"


def sidebar():
    """Enhanced sidebar with all features"""
    with st.sidebar:
        st.title("🔍 Multimodal RAG System")
        st.caption("Smart Intelligence Centre - SIH 2024")
        
        # Model info
        with st.expander("🔧 System Configuration", expanded=False):
            st.code(f"Chat: {OLLAMA_CHAT_MODEL}", language="text")
            st.code(f"Vision: {OLLAMA_VISION_MODEL}", language="text")
            st.code(f"Whisper: {WHISPER_MODEL_SIZE}", language="text")
        
        st.markdown("---")
        
        # Enhanced statistics
        st.subheader("📊 Database Statistics")
        stats = st.session_state.rag.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", stats['total'])
            st.metric("PDF Files", stats['pdf'])
            st.metric("Audio Files", stats['audio'])
        with col2:
            st.metric("Text Files", stats['text'])
            st.metric("DOCX Files", stats['docx'])
            st.metric("Images", stats['image'])
        
        st.markdown("---")
        
        # Upload section with tabs
        st.subheader("📤 Add Documents")
        
        tab1, tab2, tab3 = st.tabs(["📄 Documents", "🖼️ Images", "🎵 Audio"])
        
        with tab1:
            # Text input
            text_input = st.text_area("Direct text input:", height=100, key="text_input_area")
            
            # File upload
            uploaded_doc = st.file_uploader(
                "Upload document",
                type=['txt', 'pdf', 'docx'],
                key='doc_uploader'
            )
            
            if st.button("➕ Add Document", type="primary", key="add_doc_btn"):
                content_added = False
                
                if uploaded_doc:
                    temp_path = Path("data/documents") / uploaded_doc.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_doc.getbuffer())
                    
                    with st.spinner("Processing document..."):
                        if uploaded_doc.type == "application/pdf":
                            doc_id, content = st.session_state.rag.add_pdf(
                                str(temp_path),
                                {'source': 'user_upload'}
                            )
                            content_added = True
                        elif uploaded_doc.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            doc_id, content = st.session_state.rag.add_docx(
                                str(temp_path),
                                {'source': 'user_upload'}
                            )
                            content_added = True
                        else:  # txt
                            content = uploaded_doc.read().decode('utf-8')
                            doc_id = st.session_state.rag.add_text(
                                content,
                                {'source': 'user_upload', 'filename': uploaded_doc.name}
                            )
                            content_added = True
                    
                    if content_added:
                        st.success(f"✅ Added: {doc_id}")
                        st.rerun()
                
                elif text_input.strip():
                    with st.spinner("Adding text..."):
                        doc_id = st.session_state.rag.add_text(
                            text_input,
                            {'source': 'user_upload', 'filename': 'direct_input.txt'}
                        )
                        st.success(f"✅ Added: {doc_id}")
                        st.rerun()
                else:
                    st.warning("Please provide text or upload a file")
        
        with tab2:
            uploaded_img = st.file_uploader(
                "Upload image",
                type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
                key='img_uploader'
            )
            
            if uploaded_img:
                st.image(uploaded_img, use_container_width=True)
                
                if st.button("➕ Add Image", type="primary", key="add_img_btn"):
                    save_path = Path("data/images") / uploaded_img.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_img.getbuffer())
                    
                    with st.spinner("Processing image..."):
                        doc_id, description = st.session_state.rag.add_image(
                            str(save_path),
                            {'source': 'user_upload'}
                        )
                        st.success(f"✅ Added: {doc_id}")
                        with st.expander("📝 Image Description"):
                            st.write(description)
                        st.rerun()
        
        with tab3:
            if not WHISPER_AVAILABLE:
                st.warning("⚠️ Whisper not installed. Install with: pip install openai-whisper")
            
            uploaded_audio = st.file_uploader(
                "Upload audio",
                type=['mp3', 'wav', 'm4a', 'flac'],
                key='audio_uploader'
            )
            
            if uploaded_audio:
                st.audio(uploaded_audio)
                
                if st.button("➕ Add Audio", type="primary", key="add_audio_btn"):
                    save_path = Path("data/audio") / uploaded_audio.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                    
                    with st.spinner("Transcribing audio..."):
                        doc_id, transcription, segments = st.session_state.rag.add_audio(
                            str(save_path),
                            {'source': 'user_upload'}
                        )
                        st.success(f"✅ Added: {doc_id}")
                        with st.expander("📝 Transcription"):
                            st.write(transcription)
                        st.rerun()
        
        st.markdown("---")
        
        # Document browser
        if st.button("📚 Browse All Documents"):
            st.session_state.show_docs = not st.session_state.get('show_docs', False)
        
        if st.session_state.get('show_docs', False):
            docs = st.session_state.rag.list_all_documents()
            for doc in docs[:5]:  # Show first 5
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    st.caption(f"Type: {doc['type'].upper()}")
                    st.caption(f"ID: {doc['id']}")
                    st.text(doc['preview'])
        
        st.markdown("---")
        
        # Management
        st.subheader("⚙️ Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Demo Data"):
                with st.spinner("Adding demo data..."):
                    st.session_state.rag.add_text(
                        "International Development Report 2024: This report discusses global development initiatives and their impact on emerging economies.",
                        {'source': 'demo', 'filename': 'dev_report_2024.txt'}
                    )
                    st.session_state.rag.add_text(
                        "Email screenshot captured at 14:32 showing project approval from the director.",
                        {'source': 'demo', 'filename': 'email_14_32.txt'}
                    )
                    st.success("✅ Demo data added!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All"):
                if st.session_state.rag.get_stats()['total'] > 0:
                    st.session_state.rag.delete_all()
                    st.session_state.chat_history = []
                    st.success("✅ Database cleared!")
                    st.rerun()


def render_citation_card(source, index):
    content = source.get("content", "")
    random_suffix = random.randint(100000, 999999)
    st.text_area(
        label=f"Citation {index + 1}",
        value=content,
        height=150,
        key=f"citation_{index}_{random_suffix}",
        disabled=True
    )
        
def main_interface():
    """Enhanced main interface with all search modes"""
    st.title("💬 Unified Query Interface")
    st.caption("Ask questions in natural language or upload files to search")
    
    # Check if database has documents
    stats = st.session_state.rag.get_stats()
    if stats['total'] == 0:
        st.info("👈 **Get Started:** Upload documents using the sidebar")
        st.info(f"💡 **Quick Start:** Place documents in `{PRELOAD_TEXT_DIR}`, `{PRELOAD_IMAGE_DIR}`, or `{PRELOAD_AUDIO_DIR}` folders and restart")
        
        # Show example queries
        with st.expander("📝 Example Queries"):
            st.markdown("""
            - "Show me the report about international development in 2024"
            - "Find the email screenshot taken at 14:32"
            - "What documents discuss project approval?"
            - Upload an image to find related documents
            - Upload audio to find matching transcripts
            """)
        return
    
    # Search mode selector
    st.markdown("### 🔍 Search Mode")
    search_cols = st.columns([1, 1, 1, 2])
    
    with search_cols[0]:
        text_mode = st.button("💬 Text Query", use_container_width=True, 
                             type="primary" if st.session_state.search_mode == 'text' else "secondary")
        if text_mode:
            st.session_state.search_mode = 'text'
    
    with search_cols[1]:
        image_mode = st.button("🖼️ Image Search", use_container_width=True,
                              type="primary" if st.session_state.search_mode == 'image' else "secondary")
        if image_mode:
            st.session_state.search_mode = 'image'
    
    with search_cols[2]:
        audio_mode = st.button("🎵 Audio Search", use_container_width=True,
                              type="primary" if st.session_state.search_mode == 'audio' else "secondary")
        if audio_mode:
            st.session_state.search_mode = 'audio'
    
    st.markdown("---")
    
    # Display mode-specific interface
    if st.session_state.search_mode == 'text':
        render_text_search()
    elif st.session_state.search_mode == 'image':
        render_image_search()
    elif st.session_state.search_mode == 'audio':
        render_audio_search()


def render_text_search():
    """Text-to-multimodal search interface"""
    st.markdown("### 💬 Natural Language Query")
    st.caption("Ask questions about your documents in plain language")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(message['content'])
                
                # Show citations
                if 'sources' in message and message['sources']:
                    st.markdown("---")
                    st.markdown("#### 📚 Sources & Citations")
                    for j, source in enumerate(message['sources'], 1):
                        render_citation_card(source, j)
    
    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and analyzing..."):
                response, sources = st.session_state.rag.query(user_query, n=5)
                st.markdown(response)
                
                # Display citations
                if sources:
                    st.markdown("---")
                    st.markdown("#### 📚 Sources & Citations")
                    for i, source in enumerate(sources, 1):
                        render_citation_card(source, i)
        
        # Add assistant message
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'sources': sources
        })
        
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def render_image_search():
    """Image-to-multimodal search interface"""
    st.markdown("### 🖼️ Image-Based Search")
    st.caption("Upload an image to find related documents, text, and audio")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_image = st.file_uploader(
            "Upload image to search",
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp'],
            key='search_image_uploader'
        )
        
        if search_image:
            st.image(search_image, caption="Query Image", use_container_width=True)
            
            if st.button("🔍 Search with this Image", type="primary", key="search_img_btn"):
                # Save temporarily
                temp_path = Path("data/temp_search.jpg")
                with open(temp_path, "wb") as f:
                    f.write(search_image.getbuffer())
                
                with st.spinner("🔍 Analyzing image and searching..."):
                    results = st.session_state.rag.search_by_image(str(temp_path), n=5)
                    st.session_state.image_search_results = results
                    st.rerun()
    
    with col2:
        if 'image_search_results' in st.session_state and st.session_state.image_search_results:
            st.markdown("#### 📊 Search Results")
            st.success(f"Found {len(st.session_state.image_search_results)} relevant documents")
            
            for i, result in enumerate(st.session_state.image_search_results, 1):
                render_citation_card(result, i)
        else:
            st.info("👆 Upload an image above to start searching")


def render_audio_search():
    """Audio-to-multimodal search interface"""
    st.markdown("### 🎵 Audio-Based Search")
    st.caption("Upload audio to find related documents and transcripts")
    
    if not WHISPER_AVAILABLE:
        st.error("⚠️ Whisper not installed. Please install: `pip install openai-whisper`")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_audio = st.file_uploader(
            "Upload audio to search",
            type=['mp3', 'wav', 'm4a', 'flac'],
            key='search_audio_uploader'
        )
        
        if search_audio:
            st.audio(search_audio)
            
            if st.button("🔍 Search with this Audio", type="primary", key="search_audio_btn"):
                # Save temporarily
                temp_path = Path("data/temp_search_audio.mp3")
                with open(temp_path, "wb") as f:
                    f.write(search_audio.getbuffer())
                
                with st.spinner("🎤 Transcribing and searching..."):
                    results = st.session_state.rag.search_by_audio(str(temp_path), n=5)
                    st.session_state.audio_search_results = results
                    st.rerun()
    
    with col2:
        if 'audio_search_results' in st.session_state and st.session_state.audio_search_results:
            st.markdown("#### 📊 Search Results")
            st.success(f"Found {len(st.session_state.audio_search_results)} relevant documents")
            
            for i, result in enumerate(st.session_state.audio_search_results, 1):
                render_citation_card(result, i)
        else:
            st.info("👆 Upload audio above to start searching")


def render_full_document_viewer():
    """Render full document viewer if a document is selected"""
    if 'selected_doc' in st.session_state:
        with st.expander("📄 Full Document View", expanded=True):
            doc = st.session_state.selected_doc
            
            st.markdown(f"**Document ID:** `{doc['id']}`")
            st.markdown(f"**Type:** {doc['metadata']['type'].upper()}")
            st.markdown(f"**Filename:** {doc['metadata'].get('filename', 'Unknown')}")
            
            st.markdown("---")
            st.markdown("**Full Content:**")
            st.text_area("", doc['content'], height=400, disabled=True, key="full_doc_view")
            
            if st.button("✖️ Close"):
                del st.session_state.selected_doc
                st.rerun()


def main():
    """Main application"""
    setup_page()
    init_session_state()
    sidebar()
    main_interface()
    render_full_document_viewer()
    
    # Footer
    st.markdown("---")
    st.caption("🔍 Multimodal RAG System | Smart Intelligence Centre | SIH 2024")
    st.caption("💡 Supports: PDF, DOCX, TXT, Images (JPG, PNG), Audio (MP3, WAV, M4A)")


if __name__ == "__main__":
    main()