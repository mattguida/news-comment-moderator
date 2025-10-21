"""
Interactive AI Agent Demo for Comment Moderation
Toxicity detection + LLM-based constructive suggestions

Requirements:
pip install flask flask-cors requests nltk beautifulsoup4 lxml torch transformers scipy scikit-learn
# Also install Ollama separately: https://ollama.ai/
# Run: ollama pull mistral
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
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
from scipy.spatial.distance import cosine # For frame vector comparison (conceptually)
from pydantic import BaseModel
from typing import List as TypingList

# LangChain imports for cloud LLM
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Download sentence tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize


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
    OTHER = "None/Other" # Catch-all for low confidence

class ReframingType(Enum):
    RETENTION = "Frame Retention" # Comment frames are a subset of article frames
    SELECTIVE = "Selective Reframe" # Comment introduces some new frames while keeping some article frames
    COMPLETE = "Complete Reframe" # Comment frames are completely different from article frames

class LLMInterventionResponse(BaseModel):
    risk_level: str
    suggestions: TypingList[str]
    allow_post: bool


app = Flask(__name__)
CORS(app)

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
            # Removed other scrappers for brevity - using BBC/Demo for the example's core logic
        }

    def fetch_articles(self, categories: List[str] = None, limit: int = 10) -> List[Dict]:
        """Fetch recent news articles by scraping news websites"""
        try:
            articles = self._scrape_bbc_articles()
        except Exception as e:
            print(f"Error scraping articles: {e}")
            raise RuntimeError("Failed to fetch articles from news sources")

        # Simple category filter
        if categories:
            # For now, just filter by title/content keywords
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
            print(f"Error scraping search results: {e}")
            # Fallback to demo articles if scraping fails
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

            for link in article_links[:20]:  # Limit to first 20 links
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

                    if content and len(content) > 100:  # Only include substantial articles
                        article = {
                            'id': article_id,
                            'title': title,
                            'content': content,
                            'source': 'BBC News',
                            'url': href,
                            'published_at': datetime.now().isoformat(),
                            'categories': []  # Could be inferred from URL or content
                        }
                        articles.append(article)

                except Exception as e:
                    print(f"Error scraping article {href}: {e}")
                    continue

            return articles

        except Exception as e:
            print(f"Error scraping BBC: {e}")
            raise RuntimeError("Failed to scrape BBC News")

    def _scrape_bbc_search(self, query: str) -> List[Dict]:
        """Scrape BBC search results for a query - optimized for 3 articles"""
        try:
            search_url = f'https://www.bbc.co.uk/search?q={query}&filter=news'
            response = self.session.get(search_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            articles = []
            # BBC search results - get top 3 most relevant links
            search_results = soup.select('a[href*="/news/"]:not([href*="live"]):not([href*="video"])')[:6]  # Get 6 to account for potential failures

            for link in search_results:
                if len(articles) >= 3:  # Stop once we have 3 articles
                    break

                href = link.get('href')
                if not href.startswith('http'):
                    href = urljoin('https://www.bbc.co.uk', href)

                # Skip if already processed
                article_id = href.split('/')[-1] if '/' in href else str(hash(href))
                if article_id in self.article_cache:
                    continue

                try:
                    # Fetch individual article
                    article_response = self.session.get(href, timeout=10)  # Add timeout
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.content, 'lxml')

                    title = article_soup.select_one('h1')
                    title = title.get_text().strip() if title else "No Title"

                    content_divs = article_soup.select('[data-component="text-block"], .article-body p, .ssrcss-uf6wea-RichTextComponentWrapper p, p[data-reactroot], .article__body p')
                    content = ' '.join([div.get_text().strip() for div in content_divs if div.get_text().strip()])

                    if content and len(content) > 200:  # Higher threshold for quality
                        article = {
                            'id': article_id,
                            'title': title,
                            'content': content,  # Limit content length for efficiency
                            'source': 'BBC News',
                            'url': href,
                            'published_at': datetime.now().isoformat(),
                            'categories': []  # Could be inferred from URL or content
                        }
                        articles.append(article)

                except Exception as e:
                    print(f"Error scraping search result article {href}: {e}")
                    continue

            return articles[:3]  # Ensure we return at most 3 articles

        except Exception as e:
            print(f"Error scraping BBC search: {e}")
            raise RuntimeError("Failed to scrape BBC search results")

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
        self.perspective_api_key = 'AIzaSyC2GXct-IoB8Rw43rp4G3yROnb5TSreS8I'
        self.ollama_url = os.getenv('OLLAMA_URL', "http://localhost:11434/api/generate")
        self.model_name = os.getenv('OLLAMA_MODEL', "gemma3:1b")

        # Check and start Ollama if needed
        self._ensure_ollama_running()

        # Hugging Face Model setup - detect best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU acceleration")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (no GPU acceleration available)")

        # Add try-except for robust loading
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.FRAME_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.FRAME_MODEL, num_labels=len(FrameType))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"ERROR LOADING HF MODEL: {e}. Frame classification will be mocked.")
            self.tokenizer = None
            self.model = None

        # Map model labels to FrameType (Crucial for a real model, mocked/guessed here)
        self.label_map = {i: frame.value for i, frame in enumerate(FrameType)}

        if self.model and self.model.config.id2label:
             # Match model's labels to the FrameType Enum by name similarity
            hf_labels = list(self.model.config.id2label.values())
            self.label_map = {i: next((ft.value for ft in FrameType if ft.value.lower().startswith(hf_labels[i].split('_')[0].lower())), FrameType.OTHER.value) for i in range(len(hf_labels))}
            print(f"Using HuggingFace label map: {self.label_map}")

    def _ensure_ollama_running(self):
        """Check if Ollama is running and try to start it if not"""
        print("Checking Ollama status...")

        # First check if Ollama is already running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is already running")
                # Check if configured model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                print(f"DEBUG: Available models: {model_names}")
                print(f"DEBUG: Looking for: {self.model_name}")
                if any(name == self.model_name or name.startswith(self.model_name) for name in model_names):
                    print(f"✅ {self.model_name} model is available")
                    return
                else:
                    print(f"⚠️  {self.model_name} model not found, attempting to pull...")
                    print(f"   Available models: {model_names}")
                    self._pull_model(self.model_name)
                    return
        except requests.exceptions.RequestException:
            print("❌ Ollama is not running, attempting to start...")

        # Try to start Ollama
        try:
            print("Starting Ollama server...")
            # Start Ollama in background
            process = subprocess.Popen(['ollama', 'serve'],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     start_new_session=True)

            # Wait a bit for it to start
            time.sleep(3)

            # Check if it's now running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama started successfully")
                    self._pull_model(self.model_name)
                    return
            except requests.exceptions.RequestException:
                print("❌ Failed to start Ollama automatically")

        except FileNotFoundError:
            print("❌ Ollama command not found. Please install Ollama: https://ollama.ai/")
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")

        print("⚠️  Ollama setup incomplete. LLM features will not work until Ollama is running with mistral model.")

    def _pull_model(self, model_name: str):
        """Pull the specified model if not available"""
        try:
            print(f"Pulling {model_name} model...")
            process = subprocess.run(['ollama', 'pull', model_name],
                                   capture_output=True, text=True, timeout=300)
            if process.returncode == 0:
                print(f"✅ {model_name} model pulled successfully")
            else:
                print(f"❌ Failed to pull {model_name} model: {process.stderr}")
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout pulling {model_name} model")
        except FileNotFoundError:
            print("❌ Ollama command not found")
        except Exception as e:
            print(f"❌ Error pulling {model_name} model: {e}")

    def _pull_mistral_model(self):
        """Pull the mistral model if not available"""
        try:
            print("Pulling mistral model...")
            process = subprocess.run(['ollama', 'pull', 'mistral'],
                                   capture_output=True, text=True, timeout=300)
            if process.returncode == 0:
                print("✅ Mistral model pulled successfully")
            else:
                print(f"❌ Failed to pull mistral model: {process.stderr}")
        except subprocess.TimeoutExpired:
            print("❌ Timeout pulling mistral model")
        except FileNotFoundError:
            print("❌ Ollama command not found")
        except Exception as e:
            print(f"❌ Error pulling mistral model: {e}")
            
    def _classify_frames(self, text_list: List[str]) -> List[List[Tuple[str, float]]]:
        """Classify frames for a list of sentences"""
        if not self.model:
            raise RuntimeError("Frame classification model not loaded")
        if not text_list:
            return []
            
        with torch.no_grad():
            inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # Apply sigmoid/softmax and get top 2 frames
            probs = torch.softmax(outputs.logits, dim=-1)
            
            results = []
            for prob in probs:
                # Get the top 2 indices and their probabilities
                top_k = torch.topk(prob, k=2)
                
                sentence_frames = []
                for score, label_index in zip(top_k.values.tolist(), top_k.indices.tolist()):
                    # Map the index to the FrameType string
                    frame_label = self.label_map.get(label_index, FrameType.OTHER.value)
                    if score > 0.15: # Confidence threshold
                        sentence_frames.append((frame_label, score))
                
                # If no strong frames, assign 'OTHER'
                if not sentence_frames:
                    sentence_frames.append((FrameType.OTHER.value, 1.0))
                    
                results.append(sentence_frames)
                
        return results

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

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for LLM-based suggestions with structured output"""
        print(f"DEBUG: Calling Ollama at {self.ollama_url} with model {self.model_name}")
        try:
            # Use structured output format
            format_schema = {
                "type": "object",
                "properties": {
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "allow_post": {"type": "boolean"}
                },
                "required": ["risk_level", "suggestions", "allow_post"]
            }

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "format": format_schema,
                "stream": False
            }
            print(f"DEBUG: Payload: {payload}")
            response = requests.post(self.ollama_url, json=payload, timeout=45)
            print(f"DEBUG: Response status: {response.status_code}")
            print(f"DEBUG: Response headers: {dict(response.headers)}")
            if response.status_code == 200:
                result = response.json()
                print(f"DEBUG: Response JSON: {result}")
                return result.get('response', '')
            else:
                print(f"Ollama error: {response.status_code} - {response.text}")
                return ""
        except requests.exceptions.ConnectionError as e:
            print(f"Ollama connection error: {e}")
            print("Make sure Ollama is running: ollama serve")
            return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""
    
    def analyze_article(self, article_text: str) -> Dict:
        """Perform sentence-level frame classification for an article"""
        # Preprocess text to improve sentence tokenization
        # Add space after periods that don't have spaces (e.g., "word1.word2" -> "word1. word2")
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
                
        # Calculate article-level frames (Top 5 frames by total score, exclude OTHER)
        article_frames_list = sorted([(f, s) for f, s in all_frames_counts.items() if f != FrameType.OTHER.value], key=lambda item: item[1], reverse=True)[:5]
        article_frames = [{'label': f[0], 'score': f[1]} for f in article_frames_list]
        
        return {
            'sentence_analysis': sentence_analysis,
            'article_frames': article_frames # Total frames of the article (string, score)
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
        
        # 1. Classify comment sentences
        comment_frames_raw = self._classify_frames(comment_sentences)
        
        # Aggregate comment frames
        all_comment_frames = set()
        for frames in comment_frames_raw:
            for frame, _ in frames:
                all_comment_frames.add(frame)
        
        # 2. Compare frames (Reframing type)
        article_frame_labels = {f['label'] for f in article_analysis['article_frames']}
        comment_frame_labels = {f for f in all_comment_frames if f != FrameType.OTHER.value}
        
        reframing_type = self._compare_frames(article_frame_labels, comment_frame_labels)
        
        # Top 5 comment frames by total score/count (exclude OTHER)
        comment_frames_list = []
        comment_frame_counts = {}
        for frames in comment_frames_raw:
            for frame, score in frames:
                if frame != FrameType.OTHER.value:  # Exclude OTHER frames
                    comment_frame_counts[frame] = comment_frame_counts.get(frame, 0) + score

        sorted_comment_frames = sorted(comment_frame_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        comment_frames_list = [{'label': f[0], 'score': f[1]} for f in sorted_comment_frames]

        return {
            'comment_frames': comment_frames_list,
            'reframing_type': reframing_type.value,
            'sentence_matches': []  # Will be filled by LLM
        }
    
    def _compare_frames(self, article_frames: set, comment_frames: set) -> ReframingType:
        """Determine reframing type"""
        if not comment_frames:
            return ReframingType.RETENTION
        
        overlap = article_frames.intersection(comment_frames)
        
        if len(overlap) == len(comment_frames) and len(overlap) > 0:
            # Comment frames are a subset (or equal) of article frames
            return ReframingType.RETENTION
        elif len(overlap) > 0:
            # Some overlap, but new frames introduced
            return ReframingType.SELECTIVE
        else:
            # No overlap
            return ReframingType.COMPLETE
    
    def generate_intervention(self, toxicity_scores: Dict, article_text: str, comment_text: str, analysis_results: Dict) -> Dict:
        """Generate constructive suggestions using LLM only when toxic or reframing detected, tracing both toxicity and frame transfer"""
        toxicity = toxicity_scores.get('TOXICITY', 0)

        reframing_type = analysis_results.get('reframing_type', ReframingType.RETENTION.value)
        detected_frames = [f['label'] for f in analysis_results.get('comment_frames', []) if f['label'] != FrameType.OTHER.value]

        # Determine if intervention is needed based on toxicity and reframing
        needs_intervention = (
            toxicity > 0.7 or  # High toxicity
            (toxicity > 0.4 and reframing_type in [ReframingType.COMPLETE.value, ReframingType.SELECTIVE.value]) or  # Medium toxicity with reframing
            reframing_type == ReframingType.COMPLETE.value  # Complete reframing (at risk even with low toxicity)
        )

        # Determine risk level based on toxicity and reframing
        risk_level = 'low'
        if toxicity > 0.7 or (toxicity > 0.5 and reframing_type == ReframingType.COMPLETE.value):
            risk_level = 'high'
        elif toxicity > 0.4 or reframing_type in [ReframingType.SELECTIVE.value, ReframingType.COMPLETE.value]:
            risk_level = 'medium'

        # Determine allow_post based on toxicity and reframing
        allow_post = toxicity < 0.8 and reframing_type != ReframingType.COMPLETE.value

        # If no intervention needed, return basic analysis without suggestions
        if not needs_intervention:
            return {
                'risk_level': risk_level,
                'toxicity_score': round(toxicity, 2),
                'reframing_type': reframing_type,
                'detected_frames': [(f['label'], f['score']) for f in analysis_results.get('comment_frames', []) if f['label'] != FrameType.OTHER.value],
                'suggestions': [],  # No suggestions for low-risk comments
                'allow_post': allow_post,
                'intervention_reason': 'Comment is not toxic and maintains article frame alignment'
            }

        # Detailed context for the LLM when intervention is needed
        context_lines = [
            f"Toxicity Score: {toxicity:.2f}",
            f"Reframing Type: {reframing_type}",
            f"Detected Frames in Comment: {', '.join(detected_frames) if detected_frames else 'None'}",
            f"Intervention Trigger: {'High toxicity' if toxicity > 0.7 else 'Complete reframing' if reframing_type == ReframingType.COMPLETE.value else 'Moderate toxicity with reframing'}"
        ]

        # Define the JSON template using json.dumps to handle escaping correctly
        json_template = json.dumps({
            "risk_level": "low|medium|high",
            "suggestions": ["suggestion1", "suggestion2"],
            "allow_post": "true|false"
        }, indent=2)

        # Create prompt for LLM focusing on toxicity and frame transfer
        prompt = """You are an AI comment moderator. Analyze this comment for toxicity and frame transfer (reframing). Provide constructive suggestions only when the comment is toxic or uses a completely different perspective from the article.

CONTEXT:
{context}

Article Excerpt: {article}...

Comment to Analyze: {comment}

This comment requires intervention due to: {intervention_trigger}

Based on toxicity and frame transfer analysis:
1. Confirm the risk level (low, medium, high) based on toxicity and reframing.
2. Provide 2-3 specific, constructive reformulations that:
   - Reduce toxicity if present
   - Help align the comment with article frames if reframing is detected
   - Maintain the core message while improving constructiveness
3. Determine if the original comment should be allowed to be posted.

Provide a JSON response with the following structure:
{template}""".format(
            context='\n'.join(context_lines),
            article=article_text,
            comment=comment_text,
            intervention_trigger='high toxicity' if toxicity > 0.7 else 'complete reframing' if reframing_type == ReframingType.COMPLETE.value else 'moderate toxicity with reframing',
            template=json_template
        )

        # Call Ollama
        response = self._call_ollama(prompt)
        print("LLM Response:", response)
        # Parse response
        if not response:
            raise RuntimeError("LLM returned empty response - ensure Ollama is running with mistral model")

        # Clean response - remove markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]  # Remove ```
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            # Ensure the LLM doesn't override critical logic (e.g., if we hard-set risk/allow based on a policy)
            final_risk_level = result.get('risk_level', risk_level)
            final_suggestions = result.get('suggestions', [])
            final_allow_post = result.get('allow_post', allow_post)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"LLM response parsing failed: {e}")

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
# GLOBAL INSTANCES
# ============================================

news_client = NewsScraperClient()
agent = AIAgent()


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def index():
    """Serve the demo interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/articles', methods=['GET'])
def get_articles():
    """Get recent news articles"""
    categories = request.args.get('categories')
    limit = int(request.args.get('limit', 10))
    
    category_list = categories.split(',') if categories else None
    articles = news_client.fetch_articles(categories=category_list, limit=limit)
    
    # Analyze frames for each article
    for article in articles:
        if article.get('content'):
            analysis = agent.analyze_article(article['content'])
            # Only store top frames for list display
            article['frames'] = [f['label'] for f in analysis['article_frames'][:3]] 
            article['frame_analysis'] = analysis  # Store for later use
        else:
            article['frames'] = []
    
    return jsonify({'articles': articles})


@app.route('/api/articles/<article_id>', methods=['GET'])
def get_article(article_id):
    """Get specific article with sentence-level frame analysis"""
    article = news_client.get_article(article_id)
    
    if not article:
        return jsonify({'error': 'Article not found'}), 404
    
    # Analyze if not already done
    if 'frame_analysis' not in article and article.get('content'):
        analysis = agent.analyze_article(article['content'])
        article['frame_analysis'] = analysis
        article['frames'] = [f['label'] for f in analysis['article_frames'][:3]]
    
    # Prepare response data (ensure we send frame analysis only once)
    response_article = article.copy()
    response_article['article_frames_detailed'] = response_article.get('frame_analysis', {}).get('article_frames', [])
    response_article['sentence_analysis'] = response_article.get('frame_analysis', {}).get('sentence_analysis', [])
    
    if 'frame_analysis' in response_article:
        del response_article['frame_analysis']
    
    return jsonify(response_article)


@app.route('/api/search', methods=['GET'])
def search_articles():
    """Search for articles by keyword - optimized for 3 results"""
    query = request.args.get('q')
    limit = int(request.args.get('limit', 3))  # Default to 3 articles for efficiency
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    articles = news_client.search_articles(query, limit=limit)
    
    for article in articles:
        if article.get('content'):
            analysis = agent.analyze_article(article['content'])
            article['frames'] = [f['label'] for f in analysis['article_frames'][:3]]
            article['frame_analysis'] = analysis
        else:
            article['frames'] = []
    
    return jsonify({'articles': articles})


@app.route('/api/analyze', methods=['POST'])
def analyze_comment():
    """
    Main endpoint: Analyze comment with toxicity, frame analysis, and intervention.
    """
    data = request.json
    comment_text = data.get('comment', '')
    article_id = data.get('article_id')
    
    if not comment_text:
        return jsonify({'error': 'No comment provided'}), 400
    
    # Get article from cache
    article = news_client.get_article(article_id)
    if not article:
        return jsonify({'error': 'Article not found'}), 404
    
    # Ensure article is analyzed (should be done in get_articles/get_article, but safety check)
    if 'frame_analysis' not in article or not article['frame_analysis']:
        article['frame_analysis'] = agent.analyze_article(article.get('content', ''))
    
    # 1. Detect toxicity
    toxicity_scores = agent.detect_toxicity(comment_text)
    
    # 2. Analyze comment with frame classification and context comparison
    comment_analysis = agent.analyze_comment_with_context(
        comment_text,
        article['frame_analysis']
    )
    
    # 3. Generate intervention
    intervention = agent.generate_intervention(
        toxicity_scores,
        article.get('content', 'No content available'),
        comment_text,
        comment_analysis
    )
    
    # Merge results for final JSON
    final_result = {**comment_analysis, **intervention}
    
    # Reformat comment frames for interface (list of tuples for easier JS parsing)
    final_result['detected_frames'] = [(f['label'], f['score']) for f in final_result['comment_frames']]
    del final_result['comment_frames']
    
    return jsonify(final_result)


# ============================================
# HTML TEMPLATE (Modified for visual appeal)
# ============================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent: Frame-Based Moderation</title>
    <style>
        /* Base Styles and Layout */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%); /* Deep Blue/Indigo */
            min-height: 100vh;
            padding: 30px;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 2fr; /* News list on left, Detail/Comment on right */
            gap: 25px;
        }

        /* Welcome/Landing Page */
        .welcome-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100vh;
            background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.5s ease, transform 0.5s ease;
        }

        .welcome-container.hidden {
            opacity: 0;
            transform: scale(0.95);
            pointer-events: none;
        }

        .welcome-card {
            background: white;
            border-radius: 20px;
            padding: 50px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            max-width: 600px;
            width: 90%;
        }

        .welcome-card h1 {
            font-size: 3em;
            color: #1e3a8a;
            margin-bottom: 20px;
            font-weight: 800;
        }

        .welcome-card p {
            font-size: 1.3em;
            color: #6b7280;
            margin-bottom: 40px;
            line-height: 1.6;
        }

        .welcome-search {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }

        .welcome-search input {
            flex: 1;
            padding: 15px 20px;
            font-size: 1.1em;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-family: inherit;
        }

        .welcome-search input:focus {
            outline: none;
            border-color: #3b82f6;
        }

        .welcome-search button {
            background: linear-gradient(to right, #3b82f6, #1d4ed8);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 1.1em;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s;
        }

        .welcome-search button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(29, 78, 216, 0.4);
        }

        .welcome-examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }

        .example-tag {
            background: #f3f4f6;
            color: #374151;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
            border: 1px solid #e5e7eb;
        }

        .example-tag:hover {
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }
        
        .header {
            grid-column: 1 / -1;
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.8em;
            margin-bottom: 8px;
            font-weight: 800;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.8;
        }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 12px 25px rgba(0,0,0,0.15);
        }

        /* News List Section (Left Column) */
        .article-list-section {
            grid-column: 1 / 2;
        }
        
        .article-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 75vh;
            overflow-y: auto;
            padding-right: 5px; /* for scrollbar spacing */
        }
        
        .article-item {
            padding: 15px;
            background: #f7f9fc;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
            position: relative;
        }
        
        .article-item:hover {
            background: #eef1f6;
            border-color: #3b82f6; /* Blue */
        }
        
        .article-item.selected {
            background: #e0f2fe; /* Light Blue */
            border-color: #1d4ed8; /* Darker Blue */
            box-shadow: 0 4px 10px rgba(29, 78, 216, 0.2);
        }
        
        .article-item h3 {
            margin-bottom: 5px;
            color: #1e3a8a;
            font-size: 1.1em;
            line-height: 1.4;
        }
        
        .article-item .source {
            font-size: 0.8em;
            color: #6b7280;
        }

        /* Article Detail and Comment Section (Right Column) */
        .detail-comment-section {
            grid-column: 2 / 3;
            display: flex;
            flex-direction: column;
            gap: 25px;
        }
        
        .article-content h2 {
            color: #1e3a8a;
            margin-bottom: 10px;
            font-size: 1.8em;
        }
        
        .article-text {
            color: #4b5563;
            line-height: 1.6;
            margin-bottom: 15px;
            min-height: 200px;
            padding-right: 10px;
        }

        .article-sentence {
            transition: background-color 0.2s ease;
        }

        .article-sentence.highlight {
            background-color: #fef3c7; /* Light yellow */
            border-radius: 3px;
            padding: 2px 4px;
            box-shadow: 0 0 0 2px #f59e0b; /* Orange border */
        }

        /* Frames and Tags */
        .frames-display, .detected-frames {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .frame-tag, .frame-item {
            background: #bfdbfe; /* Light Blue */
            color: #1d4ed8; /* Dark Blue */
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .frame-item {
            background: #3b82f6;
            color: white;
        }
        
        .frame-confidence {
            opacity: 0.8;
            font-size: 0.9em;
        }

        /* Comment Input */
        textarea {
            width: 100%;
            min-height: 100px;
            padding: 15px;
            font-size: 1em;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-family: inherit;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #3b82f6;
        }
        
        .analyze-btn {
            background: linear-gradient(to right, #3b82f6, #1d4ed8);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1.1em;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s, opacity 0.2s;
            font-weight: 600;
        }
        
        .analyze-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(29, 78, 216, 0.4);
        }
        
        .analyze-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Results Display */
        .results {
            margin-top: 20px;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid transparent;
            box-shadow: 0 6px 15px rgba(0,0,0,0.05);
        }
        
        .results.low { background: #dcfce7; border-left: 5px solid #10b981; } /* Green */
        .results.medium { background: #fffbeb; border-left: 5px solid #f59e0b; } /* Yellow */
        .results.high { background: #fee2e2; border-left: 5px solid #ef4444; } /* Red */
        
        .risk-badge {
            display: inline-block;
            padding: 6px 15px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 15px;
            color: white;
        }
        
        .risk-badge.low { background: #10b981; }
        .risk-badge.medium { background: #f59e0b; }
        .risk-badge.high { background: #ef4444; }
        
        .metrics {
            display: flex;
            gap: 15px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        
        .metric {
            background: #f7f9fc;
            padding: 15px;
            border-radius: 8px;
            flex: 1;
            min-width: 150px;
        }
        
        .metric-label {
            font-size: 0.85em;
            color: #6b7280;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.6em;
            font-weight: 700;
            color: #1e3a8a;
        }

        .suggestion {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .suggestion-type {
            font-weight: bold;
            color: #1d4ed8;
            margin-bottom: 5px;
            text-transform: uppercase;
            font-size: 0.7em;
        }
        
        /* Sentence Matching Visuals */
        .sentence-matches {
            margin-top: 20px;
            padding: 15px;
            background: #f7f9fc;
            border-radius: 10px;
            border: 1px dashed #bfdbfe;
        }
        
        .match-item {
            margin-bottom: 15px;
            padding: 12px;
            background: white;
            border-radius: 8px;
            border-left: 3px solid #3b82f6;
        }
        
        .comment-sentence {
            font-weight: 600;
            color: #1e3a8a;
            margin-bottom: 8px;
        }
        
        .article-match {
            margin-top: 8px;
            padding: 10px;
            background: #e0f2fe; /* Very light blue */
            border-radius: 6px;
            font-size: 0.9em;
            border: 1px solid #bfdbfe;
        }
        
        .overlap-score {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            margin-left: 10px;
            font-weight: bold;
        }

        /* Scrollbar styles for aesthetics */
        .article-list::-webkit-scrollbar, .article-text::-webkit-scrollbar {
            width: 8px;
        }
        .article-list::-webkit-scrollbar-thumb, .article-text::-webkit-scrollbar-thumb {
            background-color: #93c5fd;
            border-radius: 4px;
        }
        .article-list::-webkit-scrollbar-track, .article-text::-webkit-scrollbar-track {
            background: #e0f2fe;
        }

        /* Responsive adjustments */
        @media (max-width: 1000px) {
            .container {
                grid-template-columns: 1fr;
            }
            .article-list-section, .detail-comment-section {
                grid-column: 1 / -1;
            }
        }
    </style>
</head>
<body>
    <!-- Welcome/Landing Page -->
    <div class="welcome-container" id="welcome-container">
        <div class="welcome-card">
            <h1>🤖 Frame-Based Moderation Agent</h1>
            <p>Discover and analyze news articles with AI-powered frame analysis and comment moderation. Enter a topic to explore current events and understand different perspectives.</p>

            <div class="welcome-search">
                <input type="text" id="welcome-search-input" placeholder="e.g. Trump, Climate Change, AI...">
                <button id="welcome-search-btn">Explore Topic</button>
            </div>

            <div class="welcome-examples">
                <span class="example-tag" data-topic="Trump">Trump</span>
                <span class="example-tag" data-topic="Climate Change">Climate Change</span>
                <span class="example-tag" data-topic="AI">AI</span>
                <span class="example-tag" data-topic="Election">Election</span>
                <span class="example-tag" data-topic="Economy">Economy</span>
                <span class="example-tag" data-topic="Health">Health</span>
            </div>
        </div>
    </div>

    <div class="container" id="main-container" style="display: none;">
        <div class="header">
            <h1>🤖 Frame-Based Moderation Agent</h1>
            <p>Toxicity, frame analysis, and constructive reformulation suggestions.</p>
        </div>
        
        <div class="card article-list-section">
            <h3 style="color: #1e3a8a; margin-bottom: 15px;">🔍 News Article Browser</h3>
            
            <div class="controls" style="margin-bottom: 15px;">
                <input type="text" id="search-input" placeholder="Search articles...">
                <button id="search-btn">Search</button>
            </div>
            
            <div class="article-list" id="article-list">
                <div style="text-align: center; padding: 30px; color: #6b7280;">
                    Loading articles...
                </div>
            </div>
        </div>
        
        <div class="detail-comment-section">
            
            <div class="card" id="article-detail" style="display:none;">
                <div class="article-content">
                    <h2 id="article-title"></h2>
                    <div class="source" id="article-source"></div>
                    
                    <div style="margin-top: 15px;">
                        <strong>Core Article Frames:</strong>
                        <div class="frames-display" id="article-frames"></div>
                    </div>
                    
                    <p class="article-text" id="article-text" style="margin-top: 15px;"></p>
                </div>
            </div>
            
            <div class="card comment-section" id="comment-card" style="display:none;">
                <h3 style="color: #1e3a8a; margin-bottom: 15px;">✍️ Submit a Comment</h3>
                <textarea id="comment-input" placeholder="Share your thoughts on this article..."></textarea>
                <button class="analyze-btn" id="analyze-btn">Analyze & Moderate Comment</button>
            </div>
            
            <div id="results-container"></div>
        </div>
    </div>
    
    <script>
        const API_BASE = '';
        let currentArticleId = null;
        let articlesCache = {};

        // Start exploration with a topic
        async function startExploration(query) {
            // Hide welcome screen
            document.getElementById('welcome-container').classList.add('hidden');

            // Show main interface
            document.getElementById('main-container').style.display = 'grid';

            // Perform search
            await searchArticles(query);
        }

        // Load articles on start
        async function loadArticles() {
            const list = document.getElementById('article-list');
            list.innerHTML = '<div style="text-align: center; padding: 30px; color: #6b7280;">Fetching articles...</div>';

            try {
                const response = await fetch(`${API_BASE}/api/articles?limit=10`);
                const data = await response.json();

                // Cache articles and display
                data.articles.forEach(article => articlesCache[article.id] = article);
                displayArticles(Object.values(articlesCache));

                // Automatically select the first article
                if (data.articles.length > 0) {
                    selectArticle(data.articles[0].id);
                    document.querySelector('.article-item').classList.add('selected');
                }
            } catch (error) {
                list.innerHTML = '<div style="text-align: center; padding: 30px; color: #ef4444;">Error loading articles. Check server/Ollama status.</div>';
            }
        }
        
        // Display articles in list
        function displayArticles(articles) {
            const list = document.getElementById('article-list');
            list.innerHTML = articles.map(article => `
                <div class="article-item" data-id="${article.id}">
                    <h3>${article.title}</h3>
                    <div class="source">${article.source || 'News'} • ${new Date(article.published_at).toLocaleDateString()}</div>
                    <div class="frames-display">
                        ${(article.frames || []).map(f => `<span class="frame-tag">${f}</span>`).join('')}
                    </div>
                </div>
            `).join('');
            
            // Add click handlers
            document.querySelectorAll('.article-item').forEach(item => {
                item.addEventListener('click', () => {
                    const id = item.dataset.id;
                    selectArticle(id);
                    document.querySelectorAll('.article-item').forEach(i => i.classList.remove('selected'));
                    item.classList.add('selected');
                });
            });
        }
        
        // Search articles
        async function searchArticles(query) {
            const list = document.getElementById('article-list');
            list.innerHTML = '<div style="text-align: center; padding: 30px; color: #6b7280;">Searching...</div>';
            
            const response = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            data.articles.forEach(article => articlesCache[article.id] = article);
            displayArticles(data.articles);
            
            // Auto-select first result
            if (data.articles.length > 0) {
                selectArticle(data.articles[0].id);
                document.querySelector('.article-item').classList.add('selected');
            } else {
                document.getElementById('article-detail').style.display = 'none';
                document.getElementById('comment-card').style.display = 'none';
            }
        }
        
        // Select and display article
        async function selectArticle(articleId) {
            currentArticleId = articleId;
            document.getElementById('results-container').innerHTML = '';
            document.getElementById('comment-input').value = '';

            // Fetch detail (includes sentence analysis)
            const response = await fetch(`${API_BASE}/api/articles/${articleId}`);
            const article = await response.json();

            document.getElementById('article-title').textContent = article.title;
            document.getElementById('article-source').textContent = `${article.source || 'News'} • ${new Date(article.published_at).toLocaleDateString()}`;

            // Render article text with sentence-level frame highlighting
            const articleTextDiv = document.getElementById('article-text');
            if (article.sentence_analysis && article.sentence_analysis.length > 0) {
                articleTextDiv.innerHTML = article.sentence_analysis.map((sent_data, index) =>
                    `<span class="article-sentence" data-frames="${sent_data.frames.map(f => f.label).join(',')}" data-index="${index}">${sent_data.sentence}</span>`
                ).join(' ');
            } else {
                articleTextDiv.textContent = article.content || 'No content available.';
            }

            const framesDiv = document.getElementById('article-frames');
            framesDiv.innerHTML = (article.article_frames_detailed || []).map(f =>
                `<span class="frame-tag hoverable-frame" data-frame="${f.label}">${f.label}</span>`
            ).join('') || '<span class="frame-tag">No core frames detected</span>';

            // Add hover event listeners for frame highlighting
            setupFrameHover();

            document.getElementById('article-detail').style.display = 'block';
            document.getElementById('comment-card').style.display = 'block';
        }

        // Setup frame hover highlighting and click scrolling
        function setupFrameHover() {
            const frameTags = document.querySelectorAll('.hoverable-frame');
            const sentences = document.querySelectorAll('.article-sentence');

            frameTags.forEach(tag => {
                tag.addEventListener('mouseenter', () => {
                    const frameLabel = tag.dataset.frame;
                    sentences.forEach(sentence => {
                        const sentenceFrames = sentence.dataset.frames.split(',');
                        if (sentenceFrames.includes(frameLabel)) {
                            sentence.classList.add('highlight');
                        }
                    });
                });

                tag.addEventListener('mouseleave', () => {
                    sentences.forEach(sentence => {
                        sentence.classList.remove('highlight');
                    });
                });

                // Add click functionality to scroll to highlighted text
                tag.addEventListener('click', () => {
                    const frameLabel = tag.dataset.frame;
                    const highlightedSentences = Array.from(sentences).filter(sentence =>
                        sentence.dataset.frames.split(',').includes(frameLabel)
                    );

                    if (highlightedSentences.length > 0) {
                        // Scroll to the first highlighted sentence
                        highlightedSentences[0].scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        });

                        // Temporarily highlight all matching sentences
                        highlightedSentences.forEach(sentence => {
                            sentence.classList.add('highlight');
                        });

                        // Remove highlight after 3 seconds
                        setTimeout(() => {
                            highlightedSentences.forEach(sentence => {
                                sentence.classList.remove('highlight');
                            });
                        }, 3000);
                    }
                });
            });
        }
        
        // Analyze comment
        async function analyzeComment() {
            const comment = document.getElementById('comment-input').value.trim();
            
            if (!comment || !currentArticleId) {
                alert('Please write a comment and ensure an article is selected.');
                return;
            }
            
            const btn = document.getElementById('analyze-btn');
            btn.disabled = true;
            btn.textContent = 'Analyzing Frames & Toxicity...';
            
            try {
                const response = await fetch(`${API_BASE}/api/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        comment: comment,
                        article_id: currentArticleId
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Server error during analysis.');
                }
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                alert('Error analyzing comment: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze & Moderate Comment';
            }
        }
        
        // Display results
        function displayResults(result) {
            const container = document.getElementById('results-container');
            
            let suggestionsHTML = result.suggestions.map(s => `
                <div class="suggestion">
                    <div class="suggestion-type">${s.type}</div>
                    <div>${s.message}</div>
                </div>
            `).join('');
            
            let framesHTML = (result.detected_frames || []).map(([frame, conf]) => `
                <span class="frame-item">
                    ${frame} <span class="frame-confidence">(${(conf * 100).toFixed(0)}%)</span>
                </span>
            `).join('');
            
            container.innerHTML = `
                <div class="results ${result.risk_level}">
                    <span class="risk-badge ${result.risk_level}">
                        RISK LEVEL: ${result.risk_level.toUpperCase()}
                    </span>

                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Toxicity Score (0-100)</div>
                            <div class="metric-value">${(result.toxicity_score * 100).toFixed(0)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Frame Reframing Type</div>
                            <div class="metric-value">${result.reframing_type}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">AI Recommended Post Status</div>
                            <div class="metric-value" style="color: ${result.allow_post ? '#10b981' : '#ef4444'};">
                                ${result.allow_post ? '✓ ALLOWED' : '✗ REFUSED'}
                            </div>
                        </div>
                    </div>

                    <div class="detected-frames" style="margin-bottom: 20px;">
                        <strong>Detected Frames in Comment:</strong><br>
                        ${framesHTML || '<em>No strong frames detected</em>'}
                    </div>

                    ${suggestionsHTML ? `
                        <div style="margin-top: 20px;">
                            <strong style="color: #1e3a8a;">💡 AI Reformulation Suggestions:</strong>
                            ${suggestionsHTML}
                        </div>
                    ` : `
                        <div style="margin-top: 20px; padding: 15px; background: #f0f9ff; border-radius: 8px; border: 1px solid #bfdbfe;">
                            <strong style="color: #1e3a8a;">✅ No Reformulation Needed</strong><br>
                            <span style="color: #6b7280; font-size: 0.9em;">${result.intervention_reason || 'Comment is appropriate and aligns with article frames.'}</span>
                        </div>
                    `}
                </div>
            `;
            
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        // Welcome screen event listeners
        document.getElementById('welcome-search-btn').addEventListener('click', () => {
            const query = document.getElementById('welcome-search-input').value.trim();
            if (query) {
                startExploration(query);
            }
        });

        document.getElementById('welcome-search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const query = e.target.value.trim();
                if (query) {
                    startExploration(query);
                }
            }
        });

        // Example tags
        document.querySelectorAll('.example-tag').forEach(tag => {
            tag.addEventListener('click', () => {
                const topic = tag.dataset.topic;
                document.getElementById('welcome-search-input').value = topic;
                startExploration(topic);
            });
        });

        // Main interface event listeners
        document.getElementById('search-btn').addEventListener('click', () => {
            const query = document.getElementById('search-input').value.trim();
            if (query) {
                searchArticles(query);
            }
        });

        document.getElementById('search-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const query = e.target.value.trim();
                if (query) {
                    searchArticles(query);
                }
            }
        });

        document.getElementById('analyze-btn').addEventListener('click', analyzeComment);

        // No automatic initialization - wait for user input
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("=" * 60)
    print("AI COMMENT MODERATION AGENT - FRAME ANALYSIS EDITION")
    print("=" * 60)
    print("\nStarting server...")
    print("\n" + "=" * 60 + "\n")
    
    # For local testing
    app.run(debug=True, port=5000, use_reloader=False)
