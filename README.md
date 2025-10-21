# 🤖 Frame-Based Comment Moderation Agent

An AI-powered system for analyzing news article comments using frame analysis, toxicity detection, and LLM-based constructive suggestions.

## Features

- 📰 **News Article Scraping**: Fetches real news articles from BBC News
- 🎯 **Frame Analysis**: Classifies article and comment frames using HuggingFace transformers
- ⚠️ **Toxicity Detection**: Uses Google Perspective API for toxicity analysis
- 🤖 **LLM Suggestions**: Provides constructive reformulations using cloud LLMs
- 🌐 **Web Interface**: Clean Streamlit interface for easy interaction

## Deployment on Streamlit Cloud

### Prerequisites

1. **GitHub Repository**: Your code must be in a public GitHub repository
2. **API Keys**: Set up the following environment variables:
   - `PERSPECTIVE_API_KEY`: Google Perspective API key (for toxicity detection)
   - `HUGGINGFACE_API_TOKEN`: HuggingFace API token (for cloud LLM)

### Steps to Deploy

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit for Streamlit deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository

3. **Configure Deployment**:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.9 or higher
   - **Requirements file**: `requirements.txt`

4. **Set Environment Variables**:
   In Streamlit Cloud dashboard, add these secrets:
   - `PERSPECTIVE_API_KEY`: Your Google Perspective API key
   - `HUGGINGFACE_API_TOKEN`: Your HuggingFace API token
   - `STREAMLIT_CLOUD`: `true` (to skip local Ollama setup)

5. **Deploy**: Click "Deploy" and wait for the build to complete

### Local Development

For local testing before deployment:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PERSPECTIVE_API_KEY="your_perspective_key"
export HUGGINGFACE_API_TOKEN="your_huggingface_token"

# Run the app
streamlit run streamlit_app.py
```

## API Keys Setup

### Google Perspective API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Perspective API
4. Create credentials (API key)
5. Copy the API key for deployment

### HuggingFace API Token
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token for deployment

## Architecture

- **Frontend**: Streamlit web interface
- **News Scraping**: BeautifulSoup for BBC News articles
- **Frame Analysis**: HuggingFace transformers model
- **Toxicity Detection**: Google Perspective API
- **LLM Suggestions**: HuggingFace Inference API (Mistral-7B)

## File Structure

```
├── streamlit_app.py      # Main Streamlit application
├── agent.py              # Original Flask version (for reference)
├── requirements.txt      # Python dependencies
├── Procfile             # Heroku deployment (legacy)
├── modal_deploy.py      # Modal deployment (legacy)
└── README.md            # This file
```

## Usage

1. **Browse Articles**: Search for news topics or load recent articles
2. **Select Article**: Choose an article to analyze
3. **Enter Comment**: Type a comment you'd like to analyze
4. **Analyze**: Click "Analyze Comment" to get:
   - Toxicity score
   - Frame alignment analysis
   - Risk assessment
   - Constructive suggestions (if needed)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
