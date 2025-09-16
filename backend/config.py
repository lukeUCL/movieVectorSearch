import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'movies')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

AI_PROVIDERS = {
    'openai': {
        'embedding_model': 'text-embedding-ada-002',
        'chat_model': 'gpt-4o-mini',
        'embedding_dims': 1536
    },
    'gemini': {
        'embedding_model': 'models/embedding-001',
        'chat_model': 'gemini-1.5-flash',
        'embedding_dims': 768
    }
}

SEARCH_CONFIG = {
    'default_limit': 20,
    'max_limit': 100,
    'vector_search_threshold': 0.5
}
