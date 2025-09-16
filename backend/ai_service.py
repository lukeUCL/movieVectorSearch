import openai
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .config import OPENAI_API_KEY, GEMINI_API_KEY, AI_PROVIDERS

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self, provider='openai'):
        self.provider = provider.lower()
        self.config = AI_PROVIDERS[self.provider]
        
        if self.provider == 'openai':
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == 'gemini':
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai
        
        logger.info(f"🤖 AI Service initialized with {provider}")
    
    def create_embedding(self, text):
        try:
            if self.provider == 'openai':
                response = self.client.embeddings.create(
                    model=self.config['embedding_model'],
                    input=text
                )
                return response.data[0].embedding
            
            elif self.provider == 'gemini':
                response = genai.embed_content(
                    model=self.config['embedding_model'],
                    content=text,
                    task_type="retrieval_query"
                )
                return response['embedding']
                
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return None
    
    def compute_similarity(self, query_embedding, movie_embeddings):
        try:
            if not query_embedding or not movie_embeddings:
                return []
            
            query_arr = np.array(query_embedding).reshape(1, -1)
            movie_arrs = np.array(movie_embeddings)
            
            similarities = cosine_similarity(query_arr, movie_arrs)[0]
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return []
    
    def manual_vector_search(self, query, movies):
        query_embedding = self.create_embedding(query)
        if not query_embedding:
            return movies
        
        enriched_movies = [m for m in movies if m.get('embedding')]
        
        if not enriched_movies:
            logger.info("No movies with embeddings for vector search")
            return movies
        
        embeddings = [m['embedding'] for m in enriched_movies]
        similarities = self.compute_similarity(query_embedding, embeddings)
        
        for i, movie in enumerate(enriched_movies):
            movie['similarity'] = similarities[i] if i < len(similarities) else 0.0
        
        enriched_movies.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        non_enriched = [m for m in movies if not m.get('embedding')]
        
        return enriched_movies + non_enriched
