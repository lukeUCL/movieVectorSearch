import re
from datetime import datetime
import logging
from .database import Database
from .ai_service import AIService
from .config import SEARCH_CONFIG

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, ai_provider='openai'):
        self.db = Database()
        self.ai = AIService(ai_provider)
    
    def search_movies(self, query, limit=None):
        limit = min(limit or SEARCH_CONFIG['default_limit'], SEARCH_CONFIG['max_limit'])
        
        base_filter = {
            'processing_status': {'$in': ['enriched', 'llm_generated']}
        }
        
        if not query:
            movies = self.db.search_movies(base_filter, limit=limit)
            return [self._format_movie_result(movie) for movie in movies]
        
        # Pure vector search for all queries
        movies = self.db.search_movies(base_filter, limit=limit*3)
        if movies:
            movies = self.ai.manual_vector_search(query, movies)[:limit]
        else:
            movies = []
        
        return [self._format_movie_result(movie) for movie in movies]
    
    def _format_movie_result(self, movie):
        formatted = {
            'id': movie.get('id', str(movie.get('_id', ''))),
            'title': movie.get('title', ''),
            'year': movie.get('year'),
            'director': movie.get('director') or (movie.get('directors', [None])[0] if movie.get('directors') else None),
            'genres': movie.get('genres', []),
            'cast': movie.get('cast', []),
            'plot': movie.get('plot') or movie.get('description', ''),
            'chatgpt_description': movie.get('enrichment_response') or movie.get('analysis', ''),
            'structured_enrichment': movie.get('structured_enrichment'),
            'similarity': movie.get('similarity', 0.0),
            'poster_url': movie.get('poster_url', ''),
            'source': movie.get('source', 'unknown')
        }
        
        return {k: v for k, v in formatted.items() if v is not None}
    
    def load_local_profile(self, profile_id):
        import os
        profile_path = os.path.join(os.getcwd(), 'sample_profile.json')
        
        try:
            import json
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load local profile: {e}")
            return {'error': 'Profile not found'}
