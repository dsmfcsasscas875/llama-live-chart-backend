import json
import httpx
import feedparser
import urllib.parse
from typing import List, Dict
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
from app.core.config import settings

class SentimentService:
    def __init__(self):
        # Groq Cloud API Anahtarınız
        self.api_key = settings.GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile" 
        self._cache = {} # {symbol: {data: result, timestamp: datetime}}
        
    def get_google_news(self, symbol: str) -> List[Dict]:
        """Google News RSS üzerinden en güncel haberleri çeker"""
        try:
            query = urllib.parse.quote(f"{symbol} stock news market analysis")
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            news_items = []
            
            for entry in feed.entries[:10]:
                news_items.append({
                    'title': entry.title,
                    'link': entry.link,
                    'publisher': entry.source.get('title', 'Google News'),
                    'published': getattr(entry, 'published', 'Today')
                })
            return news_items
        except Exception as e:
            print(f"Haber çekme hatası (Google News): {e}")
            return []

    async def analyze_news(self, symbol: str = "BTC") -> List[Dict]:
        """Profesyonel haberleri çeker ve Groq ile analiz eder (Caching dahil)"""
        
        # 1. Check Cache
        if symbol in self._cache:
            cache_entry = self._cache[symbol]
            if datetime.now() - cache_entry['timestamp'] < timedelta(hours=1):
                return cache_entry['data']

        # Get news from Google News
        news_list = self.get_google_news(symbol)
            
        if not news_list:
            return []

        all_titles = "\n".join([f"{i+1}. {item['title']}" for i, item in enumerate(news_list)])
        
        # Daha güçlü ve yönlendirmeli bir System Prompt
        system_prompt = (
            "You are a professional High-Frequency Trading Analyst. Analyze the financial impact of each news headline. "
            "Your goal is to classify news as 'Positive' (Bullish), 'Negative' (Bearish), or 'Neutral' (only if no market impact). "
            "Be decisive; avoid using 'Neutral' for news that clearly affects investor sentiment. "
            "Assign a 'score' (0.0 to 1.0) representing confidence/strength of the move. "
            "Return JSON in this EXACT format: {\"results\": [{\"label\": \"Positive\", \"score\": 0.85}, ...]} "
            "Maintain the exact order and length of the inputs."
        )
        
        user_prompt = f"Analyze these {symbol} headlines and return sentiment JSON:\n{all_titles}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1 # Düşük sıcaklık daha istikrarlı finansal yargı sağlar
                    },
                    timeout=20.0
                )
                
                ai_data = response.json()
                raw_content = ai_data['choices'][0]['message']['content']
                json_data = json.loads(raw_content)
                
                # 'results' yoksa direkt liste olarak dönmüş mü kontrol et
                sentiment_results = json_data.get('results', []) if isinstance(json_data, dict) else json_data

                for i, news in enumerate(news_list):
                    if i < len(sentiment_results):
                        news['sentiment'] = sentiment_results[i].get('label', 'Neutral')
                        news['sentiment_score'] = sentiment_results[i].get('score', 0.5)
                        
                        # Probabilities for UI visualization (vibrant bars)
                        score = news['sentiment_score']
                        if news['sentiment'] == 'Positive':
                            news['sentiment_probs'] = {'positive': score, 'negative': round((1-score)*0.2, 2), 'neutral': round((1-score)*0.8, 2)}
                        elif news['sentiment'] == 'Negative':
                            news['sentiment_probs'] = {'positive': round((1-score)*0.2, 2), 'negative': score, 'neutral': round((1-score)*0.8, 2)}
                        else:
                            news['sentiment_probs'] = {'positive': 0.1, 'negative': 0.1, 'neutral': 0.8}
                    else:
                        news['sentiment'] = 'Neutral'
                        news['sentiment_score'] = 0.5
                        news['sentiment_probs'] = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                
                self._cache[symbol] = {
                    'data': news_list,
                    'timestamp': datetime.now()
                }
                return news_list

        except Exception as e:
            print(f"Groq Sentiment Analysis Error: {e}")
            for news in news_list:
                news['sentiment'] = 'Neutral'
                news['sentiment_score'] = 0.5
                news['sentiment_probs'] = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            return news_list

sentiment_service = SentimentService()
