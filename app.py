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
import requests

# ========== MODEL CONFIGURATION ==========
OLLAMA_CHAT_MODEL = "deepseek-r1:1.5b"
OLLAMA_VISION_MODEL = "llama3.2-vision"
WHISPER_MODEL_SIZE = "base"
OLLAMA_URL = "https://unrent-tess-histoid.ngrok-free.dev"
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
        Enhanced query with citation transparency.
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

        # Construct prompt for Ollama
        prompt = f"""You are a helpful assistant for a document retrieval system. 
Answer the question using the provided sources.

IMPORTANT INSTRUCTIONS:
1. Use numbered citations [1], [2], etc. when referencing information from sources.
2. Be specific about which source you're using.
3. If information comes from multiple sources, cite all relevant ones.
4. Keep your answer clear and concise.
5. If the sources don't contain enough information, say so.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""

        # Call Ollama through ngrok tunnel using requests
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("response", "") or data.get("message", {}).get("content", "")
        except Exception as e:
            content = f"Error connecting to Ollama: {e}"

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
