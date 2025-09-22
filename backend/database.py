from pymongo import MongoClient
import logging
from .config import MONGODB_URI, MONGODB_DB_NAME

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client[MONGODB_DB_NAME]
            self.collection = self.db['films']
            self.profiles = self.db['profiles']
            
            self._ensure_vector_search_index()
            
            try:
                total_movies = self.collection.count_documents({})
                enriched_movies = self.collection.count_documents({'processing_status': 'enriched'})
                logger.info(f"database: {total_movies:,} total movies, {enriched_movies:,} enriched")
            except Exception as e:
                logger.warning(f"Database connection failed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _ensure_vector_search_index(self):
        try:
            # Check if vector search index exists by trying to use it
            test_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_search_index",
                        "path": "embedding",
                        "queryVector": [0.0] * 1536,  
                        "numCandidates": 1,
                        "limit": 1
                    }
                }
            ]
            
            list(self.collection.aggregate(test_pipeline))
            logger.info(" Vector search index exists")
            
        except Exception as e:
            logger.warning(f"Vector search index not found: {e}")
    
    def search_movies(self, query_filter, limit=20, skip=0):
        try:
            return list(self.collection.find(query_filter).limit(limit).skip(skip))
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def count_movies(self, query_filter=None):
        try:
            return self.collection.count_documents(query_filter or {})
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
    
    def aggregate_movies(self, pipeline):
        try:
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return []
    
    def get_profile(self, profile_id):
        try:
            return self.profiles.find_one({'_id': profile_id})
        except Exception as e:
            logger.error(f"Profile fetch failed: {e}")
            return None
