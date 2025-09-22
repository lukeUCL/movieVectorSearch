from flask import request, jsonify
import logging
from datetime import datetime
from .search import SearchService
from ai_clustering_service import ai_clustering_service
from poster_cache_service import get_poster_url

logger = logging.getLogger(__name__)

def register_routes(app, ai_provider='openai'):
    search_service = SearchService(ai_provider)
    
    @app.route('/api/search', methods=['POST'])
    def search():
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = data.get('limit', 20)
                
        try:
            movies = search_service.search_movies(query, limit)
            
            logger.info(f"Found {len(movies)} movies")
            return jsonify({
                'results': movies, 
                'total': len(movies),
                'query': query,
                'provider': ai_provider
            })
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/poster', methods=['POST'])
    def get_poster():
        data = request.get_json()
        title = data.get('title')
        year = data.get('year')
        
        if not title:
            return jsonify({'error': 'Title required'}), 400
        
        try:
            poster_url = get_poster_url(title, year)
            return jsonify({'poster_url': poster_url})
        except Exception as e:
            logger.error(f"Poster fetch failed: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analyze-with-prompt', methods=['POST'])
    def analyze_with_prompt():
        try:
            data = request.get_json()
            prompt = data.get('prompt', '').strip()
            movies = data.get('movies', []) 
            
            if not movies:
                return jsonify({'error': 'Movies required'}), 400
            
            logger.info(f"Personalized analysis request for: '{prompt}' with {len(movies)} movies")
            
            # Load user profile for personalized analysis
            profile_data = search_service.load_local_profile('movie_lover_001')
            
            # Build user context for AI service
            user_context = {
                'userReviews': profile_data.get('reviews', []),
                'watchedMovies': profile_data.get('watched_movies', []),
                'likedMovies': profile_data.get('liked_movies', []),
                'favoriteGenres': profile_data.get('favorite_genres', [])
            }
            
            logger.info(f"📝 Profile loaded: {len(user_context.get('userReviews', []))} reviews")
            
            # Use AI clustering service for personalized analysis
            analysis = ai_clustering_service.generate_personalized_analysis(
                movies, prompt, user_context
            )
            
            # Extract metadata for compatibility
            years = [movie.get('year', 0) for movie in movies if movie.get('year')]
            genres = []
            for movie in movies:
                genres.extend(movie.get('genres', []))
            
            year_range = f"{min(years)}-{max(years)}" if years else "various years"
            common_genres = list(set(genres))[:3]
            
            return jsonify({
                'analysis': analysis,
                'movieCount': len(movies),
                'yearRange': year_range,
                'commonGenres': common_genres
            })
            
        except Exception as e:
            logger.error(f"Prompt analysis error: {e}")
            return jsonify({'error': 'Analysis failed'}), 500
    
    @app.route('/api/analyze-subtopics', methods=['POST'])
    def analyze_subtopics():
        """Analyze movies for subtopic grouping"""
        try:
            data = request.json
            movies = data.get('movies', [])
            main_theme = data.get('mainTheme', '')
            min_groups = data.get('minGroups', 2)
            max_groups = data.get('maxGroups', 3)
            
            if len(movies) < min_groups:
                return jsonify({
                    'hasNaturalSplit': False,
                    'groups': [],
                    'reason': 'Not enough movies for grouping'
                })
            
            genres_map = {}
            for movie in movies:
                for genre in movie.get('genres', []):
                    if genre not in genres_map:
                        genres_map[genre] = []
                    genres_map[genre].append(movie)
            
            # Create groups from most common genres
            groups = []
            for genre, genre_movies in sorted(genres_map.items(), key=lambda x: len(x[1]), reverse=True):
                if len(groups) >= max_groups:
                    break
                if len(genre_movies) >= 2:  
                    groups.append({
                        'subtheme': f"{genre} films",
                        'movies': [movie.get('title', '') for movie in genre_movies],
                        'confidence': len(genre_movies) / len(movies)
                    })
            
            return jsonify({
                'hasNaturalSplit': len(groups) >= min_groups,
                'groups': groups,
                'mainTheme': main_theme
            })
            
        except Exception as e:
            logger.error(f"Subtopics analysis error: {e}")
            return jsonify({'error': 'Subtopics analysis failed'}), 500

    @app.route('/api/analyze-prompt-chain', methods=['POST'])
    def analyze_prompt_chain():
        """Analyze a chain of prompts for themes"""
        try:
            data = request.json
            prompt_chain = data.get('promptChain', [])
            movies = data.get('movies', [])
            
            if not prompt_chain:
                return jsonify({'error': 'Prompt chain required'}), 400
            
            # Simple analysis of prompt evolution
            original_topic = prompt_chain[0] if prompt_chain else ''
            expansions = prompt_chain[1:] if len(prompt_chain) > 1 else []
            
            # Generate subtheme based on the chain
            if expansions:
                subtheme = f"{original_topic} exploring {' and '.join(expansions)}"
            else:
                subtheme = original_topic
            
            return jsonify({
                'subtheme': subtheme,
                'confidence': 0.8,  # Static confidence for now
                'originalTopic': original_topic,
                'expansions': expansions,
                'analysis': f"The exploration evolved from '{original_topic}' through {len(expansions)} thematic expansions."
            })
            
        except Exception as e:
            logger.error(f"Prompt chain analysis error: {e}")
            return jsonify({'error': 'Chain analysis failed'}), 500
    
    @app.route('/api/subcluster-insight', methods=['POST'])
    def subcluster_insight():
        try:
            data = request.get_json()
            movies = data.get('movies', [])
            main_prompt = data.get('mainPrompt', '')
            subtheme = data.get('subtheme', '')
            user_engagement = data.get('userEngagement', {})
            
            # Use AI clustering service for insight generation
            insight = ai_clustering_service.generate_subcluster_insight(
                movies, main_prompt, subtheme, user_engagement
            )
            
            return jsonify({
                'insight': insight,
                'subtheme': subtheme,
                'engagement': user_engagement
            })
            
        except Exception as e:
            logger.error(f"Subcluster insight error: {e}")
            return jsonify({'error': 'Failed to generate insight'}), 500
    
    @app.route('/api/profile/dummy', methods=['GET'])
    def get_dummy_profile():
        try:
            profile_data = search_service.load_local_profile('movie_lover_001')
            logger.info(f"📝 Serving profile: {len(profile_data.get('reviews', []))} reviews")
            return jsonify(profile_data)
        except Exception as e:
            logger.error(f"Profile loading error: {e}")
            return jsonify({'error': 'Failed to load profile'}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'ai_provider': ai_provider,
            'database_connected': True
        })
    
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            'service': '🎬 Movie Search API',
            'status': 'running',
            'endpoints': ['/api/search', '/api/poster', '/api/analyze-with-prompt']
        })
