"""
Streamlit App for AI Comment Moderation Agent
Deploy on Streamlit Cloud: https://share.streamlit.io/
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import os
from typing import Dict, List, Tuple
from enum import Enum
import nltk
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import subprocess
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine
from pydantic import BaseModel
from typing import List as TypingList

# LangChain imports for cloud LLM
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

# ============================================
# HUGGINGFACE HUB SETUP
# ============================================

from getpass import getpass

# Setup HuggingFace Hub API token for cloud models
if not os.getenv('HUGGINGFACE_API_TOKEN'):
    try:
        HUGGINGFACE_API_TOKEN = getpass("Enter your HuggingFace Hub API token: ")
        os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN
        print("✅ HuggingFace Hub API token set successfully")
    except Exception as e:
        print(f"❌ Error setting API token: {e}")
        print("Please set HUGGINGFACE_API_TOKEN environment variable manually")


# ============================================
# CONFIGURATION
# ============================================

class FrameType(Enum):
    ECONOMIC = "Economic"
    MORALITY = "Morality"
    FAIRNESS_EQUALITY = "Fairness and Equality"
    LEGALITY_CRIME = "Legality and Crime"
    POLITICAL_POLICIES = "Political and Policies"
    SECURITY_DEFENSE = "Security and Defense"
    HEALTH_SAFETY = "Health and Safety"
    CULTURAL_IDENTITY = "Cultural Identity"
    PUBLIC_OPINION = "Public Opinion"
    OTHER = "None/Other"

class ReframingType(Enum):
    RETENTION = "Frame Retention"
    SELECTIVE = "Selective Reframe"
    COMPLETE = "Complete Reframe"

class LLMInterventionResponse(BaseModel):
    risk_level: str
    suggestions: TypingList[str]
    allow_post: bool

# ============================================
# NEWS SCRAPING INTEGRATION
# ============================================

class NewsScraperClient:
    """
    Scrape real news articles from various news websites without API keys
    """

    def __init__(self):
        self.article_cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # News sources to scrape (no API keys needed)
        self.news_sources = {
            'bbc': {
                'url': 'https://www.bbc.com/news',
                'article_selector': 'a[href*="/news/"]:not([href*="live"]):not([href*="video"])',
                'title_selector': 'h1',
                'content_selector': '[data-component="text-block"], .article-body p, .ssrcss-uf6wea-RichTextComponentWrapper p, p[data-reactroot], .article__body p',
                'base_url': 'https://www.bbc.com'
            },
        }

    def fetch_articles(self, categories: List[str] = None, limit: int = 10) -> List[Dict]:
        """Fetch recent news articles by scraping news websites"""
        try:
            articles = self._scrape_bbc_articles()
        except Exception as e:
            st.error(f"Error scraping articles: {e}")
            articles = self._get_demo_articles()

        # Simple category filter
        if categories:
            category_keywords = {
                'politics': ['politics', 'government', 'election', 'policy'],
                'health': ['health', 'medical', 'covid', 'disease'],
                'science': ['science', 'research', 'study', 'discovery'],
                'environment': ['climate', 'environment', 'green', 'sustainability'],
                'economy': ['economy', 'economic', 'business', 'finance']
            }
            filtered_articles = []
            for article in articles:
                text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
                for cat in categories:
                    if any(keyword in text for keyword in category_keywords.get(cat, [cat])):
                        filtered_articles.append(article)
                        break
            articles = filtered_articles

        # Cache articles
        for article in articles:
            self.article_cache[article['id']] = article

        return articles[:limit]

    def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for articles by scraping search results"""
        try:
            articles = self._scrape_bbc_search(query)
        except Exception as e:
            st.error(f"Error scraping search results: {e}")
            articles = self._get_demo_articles()
            query_lower = query.lower()
            articles = [a for a in articles if query_lower in a['title'].lower() or query_lower in a['content'].lower()]

        # Cache articles
        for article in articles:
            self.article_cache[article['id']] = article

        return articles[:limit]

    def get_article(self, article_id: str) -> Dict:
        """Get article from cache"""
        return self.article_cache.get(article_id)

    def _scrape_bbc_articles(self) -> List[Dict]:
        """Scrape articles from BBC News"""
        try:
            response = self.session.get('https://www.bbc.com/news')
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            articles = []
            article_links = soup.select('a[href*="/news/"]:not([href*="live"]):not([href*="video"])')

            for link in article_links[:20]:
                href = link.get('href')
                if not href.startswith('http'):
                    href = urljoin('https://www.bbc.com', href)

                # Skip if already processed
                article_id = href.split('/')[-1] if '/' in href else str(hash(href))
                if article_id in self.article_cache:
                    continue

                try:
                    # Fetch individual article
                    article_response = self.session.get(href)
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.content, 'lxml')

                    title = article_soup.select_one('h1')
                    title = title.get_text().strip() if title else "No Title"

                    content_divs = article_soup.select('[data-component="text-block"], .article-body p, .ssrcss-uf6wea-RichTextComponentWrapper p, p[data-reactroot], .article__body p')
                    content = ' '.join([div.get_text().strip() for div in content_divs if div.get_text().strip()])

                    if content and len(content) > 100:
                        article = {
                            'id': article_id,
                            'title': title,
                            'content': content,
                            'source': 'BBC News',
                            'url': href,
                            'published_at': datetime.now().isoformat(),
                            'categories': []
                        }
                        articles.append(article)

                except Exception as e:
                    st.warning(f"Error scraping article {href}: {e}")
                    continue

            return articles

        except Exception as e:
            st.error(f"Error scraping BBC: {e}")
            raise

    def _scrape_bbc_search(self, query: str) -> List[Dict]:
        """Scrape BBC search results for a query - optimized for 3 articles"""
        try:
            search_url = f'https://www.bbc.co.uk/search?q={query}&filter=news'
            response = self.session.get(search_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            articles = []
            search_results = soup.select('a[href*="/news/"]:not([href*="live"]):not([href*="video"])')[:6]

            for link in search_results:
                if len(articles) >= 3:
                    break

                href = link.get('href')
                if not href.startswith('http'):
                    href = urljoin('https://www.bbc.co.uk', href)

                article_id = href.split('/')[-1] if '/' in href else str(hash(href))
                if article_id in self.article_cache:
                    continue

                try:
                    article_response = self.session.get(href, timeout=10)
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.content, 'lxml')

                    title = article_soup.select_one('h1')
                    title = title.get_text().strip() if title else "No Title"

                    content_divs = article_soup.select('[data-component="text-block"], .article-body p, .ssrcss-uf6wea-RichTextComponentWrapper p, p[data-reactroot], .article__body p')
                    content = ' '.join([div.get_text().strip() for div in content_divs if div.get_text().strip()])

                    if content and len(content) > 200:
                        article = {
                            'id': article_id,
                            'title': title,
                            'content': content,
                            'source': 'BBC News',
                            'url': href,
                            'published_at': datetime.now().isoformat(),
                            'categories': []
                        }
                        articles.append(article)

                except Exception as e:
                    st.warning(f"Error scraping search result article {href}: {e}")
                    continue

            return articles[:3]

        except Exception as e:
            st.error(f"Error scraping BBC search: {e}")
            raise

    def _get_demo_articles(self) -> List[Dict]:
        """Demo articles for fallback when scraping fails"""
        return [
            {
                'id': 'demo1',
                'title': 'Sample News Article 1',
                'content': 'This is a sample article content for testing purposes. It contains some text about current events and political discussions.',
                'source': 'Demo News',
                'url': 'https://example.com/demo1',
                'published_at': datetime.now().isoformat(),
                'categories': ['politics']
            },
            {
                'id': 'demo2',
                'title': 'Sample News Article 2',
                'content': 'Another sample article discussing economic policies and their impact on society. This includes various viewpoints on fiscal matters.',
                'source': 'Demo News',
                'url': 'https://example.com/demo2',
                'published_at': datetime.now().isoformat(),
                'categories': ['economy']
            }
        ]

# ============================================
# AI AGENT COMPONENTS
# ============================================

class AIAgent:
    """
    The AI agent with toxicity detection and LLM-based constructive suggestions
    """
    FRAME_MODEL = "mattdr/sentence-frame-classifier"

    def __init__(self):
        self.perspective_api_key = os.getenv('PERSPECTIVE_API_KEY', 'AIzaSyC2GXct-IoB8Rw43rp4G3yROnb5TSreS8I')

        # Hugging Face Model setup - detect best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            st.info("Using CUDA GPU acceleration")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            st.info("Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            st.info("Using CPU (no GPU acceleration available)")

        # Add try-except for robust loading
        try:
            st.info(f"Loading frame classification model: {self.FRAME_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.FRAME_MODEL)

            # Load model with explicit configuration to avoid meta tensor issues
            st.info("Loading model configuration...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.FRAME_MODEL,
                num_labels=len(FrameType),
                torch_dtype=torch.float32,  # Explicitly set dtype
                low_cpu_mem_usage=True
            )

            # Move to device more carefully
            st.info(f"Moving model to device: {self.device}")
            try:
                if hasattr(self.model, 'to_empty') and self.device.type != 'cpu':
                    self.model = self.model.to_empty(device=self.device)
                else:
                    self.model = self.model.to(self.device)
            except Exception as device_error:
                st.warning(f"Device placement failed: {device_error}. Using CPU.")
                self.model = self.model.to(torch.device("cpu"))
                self.device = torch.device("cpu")

            self.model.eval()
            st.success("✅ Frame classification model loaded successfully")

        except Exception as e:
            st.error(f"ERROR LOADING HF MODEL: {e}. Frame classification will be mocked.")
            st.error(f"Model: {self.FRAME_MODEL}")
            st.error(f"Device: {self.device}")
            st.error(f"Available devices: CUDA={torch.cuda.is_available()}, MPS={torch.backends.mps.is_available()}")
            self.tokenizer = None
            self.model = None

        # Map model labels to FrameType (Crucial for a real model, mocked/guessed here)
        self.label_map = {i: frame.value for i, frame in enumerate(FrameType)}

        if self.model and self.model.config.id2label:
            hf_labels = list(self.model.config.id2label.values())
            # Create a more robust mapping that handles mismatched label counts
            num_labels = min(len(hf_labels), len(FrameType))
            self.label_map = {}
            for i in range(num_labels):
                hf_label = hf_labels[i]
                # Try to match with FrameType values
                frame_match = next((ft.value for ft in FrameType if ft.value.lower().startswith(hf_label.split('_')[0].lower())), None)
                if frame_match:
                    self.label_map[i] = frame_match
                else:
                    self.label_map[i] = FrameType.OTHER.value

            # Fill remaining slots with OTHER if model has more labels than FrameType
            for i in range(num_labels, len(hf_labels)):
                self.label_map[i] = FrameType.OTHER.value

            st.info(f"Using HuggingFace label map: {self.label_map}")

        # Initialize LangChain for cloud LLM
        try:
            # Try multiple models in order of preference for conversational tasks
            models_to_try = [
                #"facebook/blenderbot-400M-distill",
                "google/flan-t5-base"
            ]

            self.llm = None
            for model_id in models_to_try:
                try:
                    self.llm = HuggingFaceEndpoint(
                        repo_id=model_id,
                        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN'),
                        temperature=0.7,
                        max_new_tokens=512,
                        task="text-generation"  # Use text-generation for these models
                    )
                    st.success(f"LLM {model_id} initialized successfully")
                    break
                except Exception as model_error:
                    st.warning(f"Failed to load {model_id}: {model_error}")
                    continue

            if not self.llm:
                # Fallback to a simpler initialization without task specification
                try:
                    self.llm = HuggingFaceEndpoint(
                        repo_id="google/flan-t5-base",
                        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN'),
                        temperature=0.7,
                        max_new_tokens=512
                    )
                    st.success("LLM initialized successfully (fallback mode)")
                except Exception as fallback_error:
                    st.warning(f"All LLM initialization attempts failed: {fallback_error}. Check HUGGINGFACE_API_TOKEN.")
                    self.llm = None

        except Exception as e:
            st.warning(f"LLM initialization failed: {e}. Check HUGGINGFACE_API_TOKEN.")
            self.llm = None


    def _classify_frames(self, text_list: List[str]) -> List[List[Tuple[str, float]]]:
        """Classify frames for a list of sentences"""
        if not self.model or not self.tokenizer:
            st.warning("Frame classification model not available, using fallback")
            return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]

        try:
            # Validate inputs
            if not text_list or len(text_list) == 0:
                return [[(FrameType.OTHER.value, 1.0)]]

            # Clean and prepare text
            clean_texts = [str(text).strip() for text in text_list if str(text).strip()]
            if not clean_texts:
                return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]

            st.info(f"Classifying {len(clean_texts)} sentences with frame model")

            with torch.no_grad():
                # Tokenize with error handling
                try:
                    inputs = self.tokenizer(
                        clean_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                    )
                except Exception as tokenize_error:
                    st.error(f"Tokenization failed: {tokenize_error}")
                    return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]

                # Move inputs to device
                if self.device.type != 'cpu':
                    try:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    except Exception as device_error:
                        st.warning(f"Device transfer failed: {device_error}. Using CPU.")
                        inputs = {k: v.to('cpu') for k, v in inputs.items()}

                # Model inference with detailed error handling
                try:
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                except Exception as model_error:
                    st.error(f"Model inference failed: {model_error}")
                    st.error(f"Input shapes: {[v.shape for v in inputs.values()]}")
                    st.error(f"Model device: {next(self.model.parameters()).device}")
                    return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]

                # Handle logits
                try:
                    if hasattr(logits, 'cpu'):
                        logits = logits.cpu()
                    elif self.device.type != 'cpu':
                        logits = logits.to('cpu')
                except Exception as logits_error:
                    st.warning(f"Logits processing failed: {logits_error}")

                # Calculate probabilities
                try:
                    probs = torch.softmax(logits, dim=-1)
                except Exception as softmax_error:
                    st.error(f"Softmax failed: {softmax_error}")
                    return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]

                # Process results
                results = []
                for i, prob in enumerate(probs):
                    try:
                        # Get top k predictions
                        top_k = torch.topk(prob, k=min(2, len(prob)))
                        sentence_frames = []

                        for score, label_index in zip(top_k.values.tolist(), top_k.indices.tolist()):
                            # Validate label_index
                            if 0 <= label_index < len(self.label_map):
                                frame_label = self.label_map.get(label_index, FrameType.OTHER.value)
                            else:
                                st.warning(f"Label index {label_index} out of range [0, {len(self.label_map)})")
                                frame_label = FrameType.OTHER.value

                            if score > 0.15:
                                sentence_frames.append((frame_label, score))

                        if not sentence_frames:
                            sentence_frames.append((FrameType.OTHER.value, 1.0))

                        results.append(sentence_frames)

                    except Exception as prob_error:
                        st.error(f"Processing probabilities failed for sentence {i}: {prob_error}")
                        results.append([(FrameType.OTHER.value, 1.0)])

                st.success(f"✅ Frame classification completed for {len(results)} sentences")
                return results

        except Exception as e:
            st.error(f"Frame classification failed: {e}. Using fallback.")
            st.error(f"Text list length: {len(text_list)}")
            st.error(f"Device: {self.device}")
            return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]

    def detect_toxicity(self, text: str) -> Dict:
        """Use Perspective API to detect toxicity"""
        if not self.perspective_api_key:
            raise RuntimeError("Perspective API key not configured")

        url = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze'
        params = {'key': self.perspective_api_key}

        data = {
            'comment': {'text': text},
            'languages': ['en'],
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'INSULT': {},
                'PROFANITY': {}
            }
        }

        response = requests.post(url, params=params, json=data)
        response.raise_for_status()
        result = response.json()

        scores = {}
        for attr in data['requestedAttributes'].keys():
            scores[attr] = result['attributeScores'][attr]['summaryScore']['value']

        return scores

    def _call_llm(self, prompt: str) -> str:
        """Call HuggingFace cloud LLM for suggestions"""
        if not self.llm:
            st.error("LLM not initialized. Please set HUGGINGFACE_API_TOKEN environment variable.")
            return ""

        # Check if API token is available
        token = os.getenv('HUGGINGFACE_API_TOKEN')
        if not token:
            st.error("HUGGINGFACE_API_TOKEN environment variable is not set.")
            return ""

        try:
            # Log for debugging (remove in production)
            st.info(f"Calling LLM with prompt length: {len(prompt)} characters")

            response = self.llm.invoke(prompt)

            if not response or response.strip() == "":
                st.warning("LLM returned empty response")
                return ""

            return response

        except Exception as e:
            st.error(f"LLM call failed: {str(e)}")
            st.error("This might be due to: 1) Invalid API token, 2) Model unavailable, 3) Rate limiting, or 4) Network issues")
            return ""

    def analyze_article(self, article_text: str) -> Dict:
        """Perform sentence-level frame classification for an article"""
        # Preprocess text to improve sentence tokenization
        article_text = re.sub(r'\.([A-Za-z])', r'. \1', article_text)

        sentences = sent_tokenize(article_text)
        if not sentences:
            return {'sentence_analysis': [], 'article_frames': []}

        sentence_frames_raw = self._classify_frames(sentences)

        sentence_analysis = []
        all_frames_counts = {}
        for i, (sent, frames) in enumerate(zip(sentences, sentence_frames_raw)):
            sentence_analysis.append({
                'sentence': sent,
                'frames': [{'label': f[0], 'confidence': f[1]} for f in frames]
            })
            for frame, score in frames:
                all_frames_counts[frame] = all_frames_counts.get(frame, 0) + score

        article_frames_list = sorted([(f, s) for f, s in all_frames_counts.items() if f != FrameType.OTHER.value], key=lambda item: item[1], reverse=True)[:5]
        article_frames = [{'label': f[0], 'score': f[1]} for f in article_frames_list]

        return {
            'sentence_analysis': sentence_analysis,
            'article_frames': article_frames
        }

    def analyze_comment_with_context(self, comment_text: str, article_analysis: Dict) -> Dict:
        """
        Perform frame classification for the comment and compare with article frames.
        """
        comment_sentences = sent_tokenize(comment_text)
        if not comment_sentences:
             return {
                'comment_frames': [],
                'reframing_type': ReframingType.RETENTION.value,
                'sentence_matches': []
            }

        comment_frames_raw = self._classify_frames(comment_sentences)

        all_comment_frames = set()
        for frames in comment_frames_raw:
            for frame, _ in frames:
                all_comment_frames.add(frame)

        article_frame_labels = {f['label'] for f in article_analysis['article_frames']}
        comment_frame_labels = {f for f in all_comment_frames if f != FrameType.OTHER.value}

        reframing_type = self._compare_frames(article_frame_labels, comment_frame_labels)

        comment_frame_counts = {}
        for frames in comment_frames_raw:
            for frame, score in frames:
                if frame != FrameType.OTHER.value:
                    comment_frame_counts[frame] = comment_frame_counts.get(frame, 0) + score

        sorted_comment_frames = sorted(comment_frame_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        comment_frames_list = [{'label': f[0], 'score': f[1]} for f in sorted_comment_frames]

        return {
            'comment_frames': comment_frames_list,
            'reframing_type': reframing_type.value,
            'sentence_matches': []
        }

    def _compare_frames(self, article_frames: set, comment_frames: set) -> ReframingType:
        """Determine reframing type"""
        if not comment_frames:
            return ReframingType.RETENTION

        overlap = article_frames.intersection(comment_frames)

        if len(overlap) == len(comment_frames) and len(overlap) > 0:
            return ReframingType.RETENTION
        elif len(overlap) > 0:
            return ReframingType.SELECTIVE
        else:
            return ReframingType.COMPLETE

    def generate_intervention(self, toxicity_scores: Dict, article_text: str, comment_text: str, analysis_results: Dict) -> Dict:
        """Generate constructive suggestions using LLM only when toxic or reframing detected"""
        toxicity = toxicity_scores.get('TOXICITY', 0)

        reframing_type = analysis_results.get('reframing_type', ReframingType.RETENTION.value)
        detected_frames = [f['label'] for f in analysis_results.get('comment_frames', []) if f['label'] != FrameType.OTHER.value]

        needs_intervention = (
            toxicity > 0.7 or
            (toxicity > 0.4 and reframing_type in [ReframingType.COMPLETE.value, ReframingType.SELECTIVE.value]) or
            reframing_type == ReframingType.COMPLETE.value
        )

        risk_level = 'low'
        if toxicity > 0.7 or (toxicity > 0.5 and reframing_type == ReframingType.COMPLETE.value):
            risk_level = 'high'
        elif toxicity > 0.4 or reframing_type in [ReframingType.SELECTIVE.value, ReframingType.COMPLETE.value]:
            risk_level = 'medium'

        allow_post = toxicity < 0.8 and reframing_type != ReframingType.COMPLETE.value

        if not needs_intervention:
            return {
                'risk_level': risk_level,
                'toxicity_score': round(toxicity, 2),
                'reframing_type': reframing_type,
                'detected_frames': [(f['label'], f['score']) for f in analysis_results.get('comment_frames', []) if f['label'] != FrameType.OTHER.value],
                'suggestions': [],
                'allow_post': allow_post,
                'intervention_reason': 'Comment is not toxic and maintains article frame alignment'
            }

        context_lines = [
            f"Toxicity Score: {toxicity:.2f}",
            f"Reframing Type: {reframing_type}",
            f"Detected Frames in Comment: {', '.join(detected_frames) if detected_frames else 'None'}",
            f"Intervention Trigger: {'High toxicity' if toxicity > 0.7 else 'Complete reframing' if reframing_type == ReframingType.COMPLETE.value else 'Moderate toxicity with reframing'}"
        ]

        json_template = json.dumps({
            "risk_level": "low|medium|high",
            "suggestions": ["suggestion1", "suggestion2"],
            "allow_post": "true|false"
        }, indent=2)

        prompt = f"""You are an AI comment moderator. Analyze this comment for toxicity and frame transfer (reframing). Provide constructive suggestions only when the comment is toxic or uses a completely different perspective from the article.

CONTEXT:
{chr(10).join(context_lines)}

Article Excerpt: {article_text[:500]}...

Comment to Analyze: {comment_text}

This comment requires intervention due to: {'high toxicity' if toxicity > 0.7 else 'complete reframing' if reframing_type == ReframingType.COMPLETE.value else 'moderate toxicity with reframing'}

Based on toxicity and frame transfer analysis:
1. Confirm the risk level (low, medium, high) based on toxicity and reframing.
2. Provide 2-3 specific, constructive reformulations that:
   - Reduce toxicity if present
   - Help align the comment with article frames if reframing is detected
   - Maintain the core message while improving constructiveness
3. Determine if the original comment should be allowed to be posted.

Provide a JSON response with the following structure:
{json_template}"""

        response = self._call_llm(prompt)
        if not response:
            raise RuntimeError("LLM returned empty response")

        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            final_risk_level = result.get('risk_level', risk_level)
            final_suggestions = result.get('suggestions', [])
            final_allow_post = result.get('allow_post', allow_post)
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"LLM response parsing failed: {e}")
            final_risk_level = risk_level
            final_suggestions = []
            final_allow_post = allow_post

        return {
            'risk_level': final_risk_level,
            'toxicity_score': round(toxicity, 2),
            'reframing_type': reframing_type,
            'detected_frames': [(f['label'], f['score']) for f in analysis_results.get('comment_frames', []) if f['label'] != FrameType.OTHER.value],
            'suggestions': [{'type': 'Constructive Rephrase', 'message': s} for s in final_suggestions],
            'allow_post': final_allow_post,
            'intervention_reason': f"Intervention due to: {'high toxicity' if toxicity > 0.7 else 'complete reframing' if reframing_type == ReframingType.COMPLETE.value else 'moderate toxicity with reframing'}"
        }

# ============================================
# STREAMLIT APP
# ============================================

def main():
    st.set_page_config(
        page_title="🤖 Frame-Based Moderation Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'news_client' not in st.session_state:
        st.session_state.news_client = NewsScraperClient()
    if 'agent' not in st.session_state:
        st.session_state.agent = AIAgent()
    if 'articles' not in st.session_state:
        st.session_state.articles = []
    if 'selected_article' not in st.session_state:
        st.session_state.selected_article = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    # Header
    st.title("🤖 Frame-Based Moderation Agent")
    st.markdown("*Toxicity detection + LLM-based constructive suggestions*")

    # Sidebar for navigation
    with st.sidebar:
        st.header("📰 News Browser")

        # Search functionality
        search_query = st.text_input("Search articles:", placeholder="e.g., Trump, Climate, AI...")
        if st.button("🔍 Search", use_container_width=True):
            if search_query:
                with st.spinner("Searching articles..."):
                    st.session_state.articles = st.session_state.news_client.search_articles(search_query, limit=5)
                    if st.session_state.articles:
                        st.session_state.selected_article = st.session_state.articles[0]
                    st.rerun()

        # Load recent articles
        if st.button("📡 Load Recent Articles", use_container_width=True):
            with st.spinner("Fetching articles..."):
                st.session_state.articles = st.session_state.news_client.fetch_articles(limit=10)
                if st.session_state.articles:
                    st.session_state.selected_article = st.session_state.articles[0]
                st.rerun()

        # Article selection
        if st.session_state.articles:
            st.subheader("Articles")
            article_titles = [f"{i+1}. {article['title'][:50]}..." for i, article in enumerate(st.session_state.articles)]
            selected_idx = st.selectbox(
                "Select article:",
                range(len(st.session_state.articles)),
                format_func=lambda x: article_titles[x],
                index=0 if st.session_state.selected_article else 0
            )
            if selected_idx is not None:
                st.session_state.selected_article = st.session_state.articles[selected_idx]

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📖 Article Content")

        if st.session_state.selected_article:
            article = st.session_state.selected_article

            st.subheader(article['title'])
            st.caption(f"Source: {article['source']} • {datetime.fromisoformat(article['published_at']).strftime('%Y-%m-%d %H:%M')}")

            # Analyze article frames if not already done
            if 'frame_analysis' not in article:
                with st.spinner("Analyzing article frames..."):
                    article['frame_analysis'] = st.session_state.agent.analyze_article(article.get('content', ''))

            # Display frames
            if article.get('frame_analysis', {}).get('article_frames'):
                st.subheader("🎯 Core Article Frames")
                frames_html = ""
                for frame in article['frame_analysis']['article_frames'][:3]:
                    frames_html += f'<span style="background:#e3f2fd; color:#1565c0; padding:4px 8px; border-radius:12px; margin:2px; display:inline-block;">{frame["label"]}</span>'
                st.markdown(frames_html, unsafe_allow_html=True)

            # Article content
            st.text_area(
                "Article Text:",
                article.get('content', 'No content available'),
                height=300,
                disabled=True
            )

            # Article URL
            if article.get('url'):
                st.markdown(f"[Read full article]({article['url']})")
        else:
            st.info("👈 Select or search for articles in the sidebar")

    with col2:
        st.header("💬 Comment Analysis")

        if st.session_state.selected_article:
            # Comment input
            comment_text = st.text_area(
                "Enter your comment:",
                placeholder="Share your thoughts on this article...",
                height=150
            )

            if st.button("🔍 Analyze Comment", use_container_width=True, type="primary"):
                if comment_text.strip():
                    with st.spinner("Analyzing comment for toxicity and frame alignment..."):
                        try:
                            # Detect toxicity
                            toxicity_scores = st.session_state.agent.detect_toxicity(comment_text)

                            # Analyze comment with context
                            comment_analysis = st.session_state.agent.analyze_comment_with_context(
                                comment_text,
                                st.session_state.selected_article['frame_analysis']
                            )

                            # Generate intervention
                            intervention = st.session_state.agent.generate_intervention(
                                toxicity_scores,
                                st.session_state.selected_article.get('content', ''),
                                comment_text,
                                comment_analysis
                            )

                            # Combine results
                            st.session_state.analysis_result = {**comment_analysis, **intervention}
                            st.session_state.analysis_result['detected_frames'] = [
                                (f['label'], f['score']) for f in st.session_state.analysis_result['comment_frames']
                            ]
                            if 'comment_frames' in st.session_state.analysis_result:
                                del st.session_state.analysis_result['comment_frames']

                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                else:
                    st.warning("Please enter a comment to analyze")

            # Display results
            if st.session_state.analysis_result:
                result = st.session_state.analysis_result

                # Risk level badge
                risk_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
                st.markdown(f"""
                <div style="background:{'lightgreen' if result['risk_level'] == 'low' else 'lightyellow' if result['risk_level'] == 'medium' else 'lightcoral'};
                     padding:15px; border-radius:10px; border-left:5px solid {risk_colors[result['risk_level']]};">
                    <h3 style="margin:0; color:{risk_colors[result['risk_level']]};">⚠️ RISK LEVEL: {result['risk_level'].upper()}</h3>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Toxicity Score", f"{int(result['toxicity_score'] * 100)}%")
                with col_b:
                    st.metric("Reframing Type", result['reframing_type'])
                with col_c:
                    status = "✅ ALLOWED" if result['allow_post'] else "❌ REFUSED"
                    st.metric("AI Recommendation", status)

                # Detected frames
                if result.get('detected_frames'):
                    st.subheader("🎯 Detected Frames in Comment")
                    frame_html = ""
                    for frame_label, score in result['detected_frames']:
                        frame_html += f'<span style="background:#f3e5f5; color:#6a1b9a; padding:4px 8px; border-radius:12px; margin:2px; display:inline-block;">{frame_label} ({score:.1%})</span>'
                    st.markdown(frame_html, unsafe_allow_html=True)

                # Suggestions
                if result.get('suggestions'):
                    st.subheader("💡 AI Reformulation Suggestions")
                    for suggestion in result['suggestions']:
                        st.info(f"**{suggestion['type']}:** {suggestion['message']}")
                else:
                    st.success("✅ No reformulation needed - comment is appropriate!")

                # Intervention reason
                if result.get('intervention_reason'):
                    st.caption(f"*{result['intervention_reason']}*")
        else:
            st.info("Select an article first to analyze comments")

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit • Powered by AI for constructive discourse")

if __name__ == "__main__":
    main()
