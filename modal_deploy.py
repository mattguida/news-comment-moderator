"""
Modal-wrapped AI Agent with FREE LLM (Mistral/Gemma via vLLM)
Run with: modal deploy modal_agent.py

Uses vLLM to run open-source models efficiently on Modal GPUs
"""
import modal

app = modal.App("comment-moderation-agent")

# Image with all dependencies including vLLM for fast inference
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",  # Let it pick the latest compatible version
        "flask",
        "flask-cors",
        "requests",
        "nltk",
        "beautifulsoup4",
        "lxml",
        "scipy",
        "scikit-learn",
        "pydantic",
    )
    .run_commands(
        "python -c 'import nltk; nltk.download(\"punkt\")'",
        "python -c 'import nltk; nltk.download(\"punkt_tab\")'"
    )
)

# Volume to cache models (important for large LLMs!)
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# Choose your free model - uncomment one:
# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Mistral 7B
LLM_MODEL = "google/gemma-2b-it"  # Gemma 2B (smaller, faster)
# LLM_MODEL = "google/gemma-7b-it"  # Gemma 7B
# LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # Phi-3 (very efficient)

@app.cls(
    image=image,
    gpu="A10G",  # A10G or A100 for better performance with 7B models
    secrets=[modal.Secret.from_name("perspective-api-key")],
    volumes={"/cache": volume},
    timeout=1800,
    container_idle_timeout=600,
    allow_concurrent_inputs=10,
)
class Model:
    """
    vLLM-powered LLM inference class
    """
    @modal.enter()
    def load_model(self):
        """Load model once when container starts"""
        import os
        from vllm import LLM, SamplingParams
        
        os.environ['TRANSFORMERS_CACHE'] = '/cache'
        os.environ['HF_HOME'] = '/cache'
        os.environ['HF_HUB_CACHE'] = '/cache'
        
        print(f"Loading {LLM_MODEL}...")
        self.llm = LLM(
            model=LLM_MODEL,
            download_dir="/cache",
            dtype="auto",  # auto-detect best dtype
            gpu_memory_utilization=0.85,
            max_model_len=2048,  # Limit context for faster inference
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["</s>", "[/INST]"],  # Stop tokens for Mistral
        )
        print("Model loaded successfully!")
    
    @modal.method()
    def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("perspective-api-key")],
    gpu="T4",  # Separate GPU for transformer models
    timeout=900,
    volumes={"/cache": volume},
    keep_warm=1,
    container_idle_timeout=300,
)
@modal.wsgi_app()
def web_app():
    import os
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import requests
    from typing import Dict, List, Tuple
    from enum import Enum
    import nltk
    import json
    import re
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    from datetime import datetime
    from pydantic import BaseModel
    from typing import List as TypingList
    from nltk.tokenize import sent_tokenize
    
    os.environ['TRANSFORMERS_CACHE'] = '/cache'
    os.environ['HF_HOME'] = '/cache'
    
    perspective_key = os.environ.get("PERSPECTIVE_API_KEY")
    
    # ============================================
    # ENUMS
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
    # NEWS SCRAPER
    # ============================================
    class NewsScraperClient:
        def __init__(self):
            self.article_cache = {}
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        
        def fetch_articles(self, categories: List[str] = None, limit: int = 10) -> List[Dict]:
            try:
                articles = self._scrape_bbc_articles()
            except Exception as e:
                print(f"Error scraping: {e}")
                articles = self._get_demo_articles()
            
            for article in articles:
                self.article_cache[article['id']] = article
            return articles[:limit]
        
        def search_articles(self, query: str, limit: int = 10) -> List[Dict]:
            try:
                articles = self._scrape_bbc_search(query)
            except Exception as e:
                print(f"Error: {e}")
                articles = self._get_demo_articles()
            
            for article in articles:
                self.article_cache[article['id']] = article
            return articles[:limit]
        
        def get_article(self, article_id: str) -> Dict:
            return self.article_cache.get(article_id)
        
        def _scrape_bbc_articles(self) -> List[Dict]:
            response = self.session.get('https://www.bbc.com/news')
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            
            articles = []
            links = soup.select('a[href*="/news/"]:not([href*="live"]):not([href*="video"])')
            
            for link in links[:10]:
                href = link.get('href')
                if not href.startswith('http'):
                    href = urljoin('https://www.bbc.com', href)
                
                article_id = href.split('/')[-1]
                if article_id in self.article_cache:
                    continue
                
                try:
                    r = self.session.get(href, timeout=10)
                    r.raise_for_status()
                    s = BeautifulSoup(r.content, 'lxml')
                    
                    title = s.select_one('h1')
                    title = title.get_text().strip() if title else "No Title"
                    
                    content_divs = s.select('[data-component="text-block"], .article-body p')
                    content = ' '.join([d.get_text().strip() for d in content_divs])
                    
                    if content and len(content) > 100:
                        articles.append({
                            'id': article_id,
                            'title': title,
                            'content': content,
                            'source': 'BBC News',
                            'url': href,
                            'published_at': datetime.now().isoformat(),
                            'categories': []
                        })
                except Exception as e:
                    print(f"Error scraping article: {e}")
                    continue
            
            return articles
        
        def _scrape_bbc_search(self, query: str) -> List[Dict]:
            url = f'https://www.bbc.co.uk/search?q={query}&filter=news'
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            
            articles = []
            links = soup.select('a[href*="/news/"]')[:6]
            
            for link in links:
                if len(articles) >= 3:
                    break
                
                href = link.get('href')
                if not href.startswith('http'):
                    href = urljoin('https://www.bbc.co.uk', href)
                
                article_id = href.split('/')[-1]
                if article_id in self.article_cache:
                    continue
                
                try:
                    r = self.session.get(href, timeout=10)
                    r.raise_for_status()
                    s = BeautifulSoup(r.content, 'lxml')
                    
                    title = s.select_one('h1')
                    title = title.get_text().strip() if title else "No Title"
                    
                    content_divs = s.select('[data-component="text-block"]')
                    content = ' '.join([d.get_text().strip() for d in content_divs])
                    
                    if content and len(content) > 200:
                        articles.append({
                            'id': article_id,
                            'title': title,
                            'content': content,
                            'source': 'BBC News',
                            'url': href,
                            'published_at': datetime.now().isoformat(),
                            'categories': []
                        })
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            return articles[:3]
        
        def _get_demo_articles(self) -> List[Dict]:
            return [{
                'id': 'demo1',
                'title': 'Sample News Article',
                'content': 'This is sample content for testing purposes.',
                'source': 'Demo News',
                'url': 'https://example.com/demo1',
                'published_at': datetime.now().isoformat(),
                'categories': []
            }]

    # ============================================
    # AI AGENT WITH FREE LLM
    # ============================================
    class AIAgent:
        FRAME_MODEL = "mattdr/sentence-frame-classifier"
        
        def __init__(self):
            self.perspective_api_key = perspective_key
            
            # Initialize LLM model reference
            self.llm_model = Model()
            
            # Detect device for frame classifier
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA GPU for frame classifier")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for frame classifier")
            
            # Load frame classifier
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.FRAME_MODEL)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.FRAME_MODEL,
                    num_labels=len(FrameType)
                )
                self.model.to(self.device)
                self.model.eval()
                print("Frame classifier loaded successfully")
            except Exception as e:
                print(f"Error loading frame classifier: {e}")
                self.tokenizer = None
                self.model = None
            
            self.label_map = {i: frame.value for i, frame in enumerate(FrameType)}
        
        def detect_toxicity(self, text: str) -> Dict:
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
            """Call Modal-hosted free LLM (Mistral/Gemma)"""
            try:
                # Call the Modal LLM class
                response = self.llm_model.generate.remote(prompt)
                return response
            except Exception as e:
                print(f"LLM error: {e}")
                return ""
        
        def _classify_frames(self, text_list: List[str]) -> List[List[Tuple[str, float]]]:
            if not self.model:
                return [[(FrameType.OTHER.value, 1.0)] for _ in text_list]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    text_list,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                results = []
                for prob in probs:
                    top_k = torch.topk(prob, k=2)
                    sentence_frames = []
                    
                    for score, label_index in zip(top_k.values.tolist(), top_k.indices.tolist()):
                        frame_label = self.label_map.get(label_index, FrameType.OTHER.value)
                        if score > 0.15:
                            sentence_frames.append((frame_label, score))
                    
                    if not sentence_frames:
                        sentence_frames.append((FrameType.OTHER.value, 1.0))
                    
                    results.append(sentence_frames)
                
                return results
        
        def analyze_article(self, article_text: str) -> Dict:
            article_text = re.sub(r'\.([A-Za-z])', r'. \1', article_text)
            sentences = sent_tokenize(article_text)
            
            if not sentences:
                return {'sentence_analysis': [], 'article_frames': []}
            
            sentence_frames_raw = self._classify_frames(sentences)
            
            sentence_analysis = []
            all_frames_counts = {}
            
            for sent, frames in zip(sentences, sentence_frames_raw):
                sentence_analysis.append({
                    'sentence': sent,
                    'frames': [{'label': f[0], 'confidence': f[1]} for f in frames]
                })
                for frame, score in frames:
                    all_frames_counts[frame] = all_frames_counts.get(frame, 0) + score
            
            article_frames_list = sorted(
                [(f, s) for f, s in all_frames_counts.items() if f != FrameType.OTHER.value],
                key=lambda item: item[1],
                reverse=True
            )[:5]
            
            article_frames = [{'label': f[0], 'score': f[1]} for f in article_frames_list]
            
            return {
                'sentence_analysis': sentence_analysis,
                'article_frames': article_frames
            }
        
        def analyze_comment_with_context(self, comment_text: str, article_analysis: Dict) -> Dict:
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
            
            sorted_frames = sorted(
                comment_frame_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )[:5]
            
            comment_frames_list = [{'label': f[0], 'score': f[1]} for f in sorted_frames]
            
            return {
                'comment_frames': comment_frames_list,
                'reframing_type': reframing_type.value,
                'sentence_matches': []
            }
        
        def _compare_frames(self, article_frames: set, comment_frames: set) -> ReframingType:
            if not comment_frames:
                return ReframingType.RETENTION
            
            overlap = article_frames.intersection(comment_frames)
            
            if len(overlap) == len(comment_frames) and len(overlap) > 0:
                return ReframingType.RETENTION
            elif len(overlap) > 0:
                return ReframingType.SELECTIVE
            else:
                return ReframingType.COMPLETE
        
        def generate_intervention(self, toxicity_scores: Dict, article_text: str,
                                 comment_text: str, analysis_results: Dict) -> Dict:
            toxicity = toxicity_scores.get('TOXICITY', 0)
            reframing_type = analysis_results.get('reframing_type', ReframingType.RETENTION.value)
            detected_frames = [f['label'] for f in analysis_results.get('comment_frames', [])]
            
            # Determine if intervention needed
            needs_intervention = (
                toxicity > 0.7 or
                (toxicity > 0.4 and reframing_type in [ReframingType.COMPLETE.value, ReframingType.SELECTIVE.value]) or
                reframing_type == ReframingType.COMPLETE.value
            )
            
            # Determine risk
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
                    'detected_frames': [(f['label'], f['score']) for f in analysis_results.get('comment_frames', [])],
                    'suggestions': [],
                    'allow_post': allow_post,
                    'intervention_reason': 'Comment is not toxic and maintains article frame alignment'
                }
            
            # Build LLM prompt
            context_lines = [
                f"Toxicity Score: {toxicity:.2f}",
                f"Reframing Type: {reframing_type}",
                f"Detected Frames: {', '.join(detected_frames) if detected_frames else 'None'}",
            ]
            
            json_template = json.dumps({
                "risk_level": "low|medium|high",
                "suggestions": ["suggestion1", "suggestion2"],
                "allow_post": "true|false"
            }, indent=2)
            
            prompt = f"""You are an AI comment moderator. Analyze this comment for toxicity and frame transfer.

CONTEXT:
{chr(10).join(context_lines)}

Article: {article_text}...

Comment: {comment_text}

Provide 2-3 specific constructive reformulations that:
1. Reduce toxicity if present
2. Help align with article frames if reframing detected
3. Maintain core message

Respond ONLY with valid JSON in this exact format:
{json_template}"""
            
            # Call FREE LLM (Mistral/Gemma via vLLM)
            response = self._call_llm(prompt)
            
            if not response:
                raise RuntimeError("LLM returned empty response")
            
            # Parse JSON response
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            try:
                result = json.loads(cleaned)
                final_risk = result.get('risk_level', risk_level)
                final_suggestions = result.get('suggestions', [])
                final_allow = result.get('allow_post', allow_post)
            except (json.JSONDecodeError, ValueError) as e:
                raise RuntimeError(f"LLM response parsing failed: {e}")
            
            return {
                'risk_level': final_risk,
                'toxicity_score': round(toxicity, 2),
                'reframing_type': reframing_type,
                'detected_frames': [(f['label'], f['score']) for f in analysis_results.get('comment_frames', [])],
                'suggestions': [{'type': 'Constructive Rephrase', 'message': s} for s in final_suggestions],
                'allow_post': final_allow,
                'intervention_reason': f"Intervention due to: {'high toxicity' if toxicity > 0.7 else 'reframing detected'}"
            }

    # ============================================
    # FLASK APP INITIALIZATION
    # ============================================
    flask_app = Flask(__name__)
    CORS(flask_app)
    
    print("Initializing components...")
    try:
        news_client = NewsScraperClient()
        agent = AIAgent()
        print("✅ Components initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        raise
    
    # Full HTML template
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent: Frame-Based Moderation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
            min-height: 100vh;
            padding: 30px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 25px;
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
        .card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 12px 25px rgba(0,0,0,0.15);
        }
        .article-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 75vh;
            overflow-y: auto;
        }
        .article-item {
            padding: 15px;
            background: #f7f9fc;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
        }
        .article-item:hover {
            background: #eef1f6;
            border-color: #3b82f6;
        }
        .article-item.selected {
            background: #e0f2fe;
            border-color: #1d4ed8;
            box-shadow: 0 4px 10px rgba(29, 78, 216, 0.2);
        }
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
        .analyze-btn {
            background: linear-gradient(to right, #3b82f6, #1d4ed8);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1.1em;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 15px;
            font-weight: 600;
        }
        .results {
            margin-top: 20px;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.05);
        }
        .results.low { background: #dcfce7; }
        .results.medium { background: #fffbeb; }
        .results.high { background: #fee2e2; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Frame-Based Moderation Agent</h1>
            <p>Toxicity & frame analysis powered by FREE LLM on Modal</p>
        </div>
        
        <div class="card">
            <h3>📰 News Articles</h3>
            <input type="text" id="search-input" placeholder="Search...">
            <button id="search-btn">Search</button>
            <div class="article-list" id="article-list">Loading...</div>
        </div>
        
        <div>
            <div class="card" id="article-detail" style="display:none;">
                <h2 id="article-title"></h2>
                <p id="article-text"></p>
            </div>
            
            <div class="card" id="comment-card" style="display:none;">
                <h3>✍️ Comment</h3>
                <textarea id="comment-input" placeholder="Your thoughts..."></textarea>
                <button class="analyze-btn" id="analyze-btn">Analyze</button>
            </div>
            
            <div id="results-container"></div>
        </div>
    </div>
    
    <script>
        let currentArticleId = null;
        let articlesCache = {};
        
        async function loadArticles() {
            const list = document.getElementById('article-list');
            try {
                const response = await fetch('/api/articles?limit=10');
                const data = await response.json();
                data.articles.forEach(a => articlesCache[a.id] = a);
                displayArticles(data.articles);
                if (data.articles.length > 0) {
                    selectArticle(data.articles[0].id);
                }
            } catch (error) {
                list.innerHTML = '<div>Error loading articles</div>';
            }
        }
        
        function displayArticles(articles) {
            const list = document.getElementById('article-list');
            list.innerHTML = articles.map(a => `
                <div class="article-item" data-id="${a.id}">
                    <h3>${a.title}</h3>
                    <div>${a.source}</div>
                </div>
            `).join('');
            
            document.querySelectorAll('.article-item').forEach(item => {
                item.addEventListener('click', () => {
                    selectArticle(item.dataset.id);
                    document.querySelectorAll('.article-item').forEach(i => i.classList.remove('selected'));
                    item.classList.add('selected');
                });
            });
        }
        
        async function selectArticle(articleId) {
            currentArticleId = articleId;
            const response = await fetch(`/api/articles/${articleId}`);
            const article = await response.json();
            
            document.getElementById('article-title').textContent = article.title;
            document.getElementById('article-text').textContent = article.content;
            document.getElementById('article-detail').style.display = 'block';
            document.getElementById('comment-card').style.display = 'block';
        }
        
        async function analyzeComment() {
            const comment = document.getElementById('comment-input').value.trim();
            if (!comment || !currentArticleId) {
                alert('Please write a comment');
                return;
            }
            
            const btn = document.getElementById('analyze-btn');
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({comment, article_id: currentArticleId})
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze';
            }
        }
        
        function displayResults(result) {
            const container = document.getElementById('results-container');
            container.innerHTML = `
                <div class="results ${result.risk_level}">
                    <h3>Risk: ${result.risk_level.toUpperCase()}</h3>
                    <p>Toxicity: ${(result.toxicity_score * 100).toFixed(0)}%</p>
                    <p>Reframing: ${result.reframing_type}</p>
                    ${result.suggestions.length > 0 ? `
                        <div><strong>Suggestions:</strong>
                        ${result.suggestions.map(s => `<p>• ${s.message}</p>`).join('')}
                        </div>
                    ` : '<p>✅ No issues detected</p>'}
                </div>
            `;
        }
        
        document.getElementById('analyze-btn').addEventListener('click', analyzeComment);
        document.getElementById('search-btn').addEventListener('click', () => {
            const query = document.getElementById('search-input').value.trim();
            if (query) searchArticles(query);
        });
        
        async function searchArticles(query) {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            data.articles.forEach(a => articlesCache[a.id] = a);
            displayArticles(data.articles);
        }
        
        loadArticles();
    </script>
</body>
</html>
"""
    
    @flask_app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @flask_app.route('/api/articles', methods=['GET'])
    def get_articles():
        categories = request.args.get('categories')
        limit = int(request.args.get('limit', 10))
        
        category_list = categories.split(',') if categories else None
        articles = news_client.fetch_articles(categories=category_list, limit=limit)
        
        for article in articles:
            if article.get('content'):
                analysis = agent.analyze_article(article['content'])
                article['frames'] = [f['label'] for f in analysis['article_frames'][:3]]
                article['frame_analysis'] = analysis
            else:
                article['frames'] = []
        
        return jsonify({'articles': articles})
    
    @flask_app.route('/api/articles/<article_id>', methods=['GET'])
    def get_article(article_id):
        article = news_client.get_article(article_id)
        if not article:
            return jsonify({'error': 'Not found'}), 404
        
        if 'frame_analysis' not in article and article.get('content'):
            analysis = agent.analyze_article(article['content'])
            article['frame_analysis'] = analysis
            article['frames'] = [f['label'] for f in analysis['article_frames'][:3]]
        
        response_article = article.copy()
        response_article['article_frames_detailed'] = response_article.get('frame_analysis', {}).get('article_frames', [])
        response_article['sentence_analysis'] = response_article.get('frame_analysis', {}).get('sentence_analysis', [])
        
        if 'frame_analysis' in response_article:
            del response_article['frame_analysis']
        
        return jsonify(response_article)
    
    @flask_app.route('/api/search', methods=['GET'])
    def search_articles():
        query = request.args.get('q')
        limit = int(request.args.get('limit', 3))
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        articles = news_client.search_articles(query, limit=limit)
        
        for article in articles:
            if article.get('content'):
                analysis = agent.analyze_article(article['content'])
                article['frames'] = [f['label'] for f in analysis['article_frames'][:3]]
                article['frame_analysis'] = analysis
            else:
                article['frames'] = []
        
        return jsonify({'articles': articles})
    
    @flask_app.route('/api/analyze', methods=['POST'])
    def analyze_comment():
        data = request.json
        comment_text = data.get('comment', '')
        article_id = data.get('article_id')
        
        if not comment_text:
            return jsonify({'error': 'No comment'}), 400
        
        article = news_client.get_article(article_id)
        if not article:
            return jsonify({'error': 'Article not found'}), 404
        
        if 'frame_analysis' not in article:
            article['frame_analysis'] = agent.analyze_article(article.get('content', ''))
        
        toxicity_scores = agent.detect_toxicity(comment_text)
        comment_analysis = agent.analyze_comment_with_context(comment_text, article['frame_analysis'])
        intervention = agent.generate_intervention(toxicity_scores, article.get('content', ''), comment_text, comment_analysis)
        
        final_result = {**comment_analysis, **intervention}
        final_result['detected_frames'] = [(f['label'], f['score']) for f in final_result['comment_frames']]
        del final_result['comment_frames']
        
        return jsonify(final_result)
    
    return flask_app