
import logging
import argparse
from flask import Flask
from flask_cors import CORS
from backend.routes import register_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(ai_provider='openai'):
    app = Flask(__name__)
    CORS(app)
    
    register_routes(app, ai_provider)
    
    logger.info(f"Movie Search API ready with {ai_provider} provider")
    return app

def main():
    parser = argparse.ArgumentParser(description='Movie Search API')
    parser.add_argument('--provider', choices=['openai', 'gemini'], default='openai', help='AI provider')
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    app = create_app(args.provider)
    
    logger.info(f"Starting server on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
