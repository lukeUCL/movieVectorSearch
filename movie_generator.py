#!/usr/bin/env python3
"""
Generate, enrich, embed and push movies to db
"""

import os
import json
import time
import argparse
import re
import signal
import sys
import random
from datetime import datetime
from pymongo import MongoClient, UpdateOne
import openai
from dotenv import load_dotenv
import logging
from typing import List, Dict
from tqdm import tqdm
from difflib import SequenceMatcher
import pickle

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlinedMovieCurator:
    def __init__(self, ai_provider='openai', session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.progress_file = f"curator_progress_{self.session_id}.json"
        self.checkpoint_file = f"curator_checkpoint_{self.session_id}.pkl"
        
        self.client = MongoClient(
            os.getenv('MONGODB_URI'),
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000
        )
        db_name = os.getenv('MONGODB_DB_NAME', 'movies')
        self.db = self.client[db_name]
        self.collection = self.db['films']
        
        self.ai_provider = ai_provider.lower()
        if self.ai_provider == 'openai':
            self.ai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            raise ValueError("Only OpenAI supported in streamlined version")
        
        # Progress tracking
        self.total_cost = 0.0
        self.completed_movies = []
        self.failed_movies = []
        self.current_category_index = 0
        self.movies_in_current_category = 0
        
        # Load existing movies for deduplication
        self.seen_movies = self.load_existing_movies()
        self.duplicate_attempts = 0  
        
        # Try to resume from checkpoint
        self.load_checkpoint()
        
        logger.info(f"Streamlined curator ready with {ai_provider}")
        logger.info(f"Found {len(self.seen_movies)} existing movies for deduplication")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Progress: {len(self.completed_movies)} movies completed")

    def load_existing_movies(self) -> set:
        try:
            existing = self.collection.find({}, {'title': 1, 'year': 1})
            seen = set()
            for movie in existing:
                title = movie.get('title', '').lower().strip()
                year = movie.get('year', 0)
                if title and year:
                    seen.add((title, year))
            return seen
        except Exception as e:
            logger.warning(f"Could not load existing movies: {e}")
            return set()

    def is_similar_movie(self, title: str, year: int, threshold: float = 0.85) -> bool:
        title_clean = title.lower().strip()
        
        for existing_title, existing_year in self.seen_movies:
            # Same year check
            if abs(year - existing_year) <= 1:  # Allow 1 year difference
                similarity = SequenceMatcher(None, title_clean, existing_title).ratio()
                if similarity >= threshold:
                    logger.debug(f"Similar movie found: '{title}' vs '{existing_title}' (similarity: {similarity:.2f})")
                    return True
        
        return False

    def refresh_seen_movies(self):
        try:
            old_count = len(self.seen_movies)
            self.seen_movies = self.load_existing_movies()
            new_count = len(self.seen_movies)
            if new_count > old_count:
                logger.info(f"🔄 Refreshed seen movies: {old_count} → {new_count}")
        except Exception as e:
            logger.warning(f"Could not refresh seen movies: {e}")

    def save_checkpoint(self):
        try:
            checkpoint = {
                'completed_movies': self.completed_movies,
                'failed_movies': self.failed_movies,
                'current_category_index': self.current_category_index,
                'movies_in_current_category': self.movies_in_current_category,
                'total_cost': self.total_cost,
                'duplicate_attempts': self.duplicate_attempts,
                'seen_movies': list(self.seen_movies),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
                
            progress = {
                'session_id': self.session_id,
                'completed_count': len(self.completed_movies),
                'failed_count': len(self.failed_movies),
                'total_cost': self.total_cost,
                'duplicate_attempts': self.duplicate_attempts,
                'last_updated': datetime.now().isoformat(),
                'completed_movies': [
                    f"{movie['title']} ({movie.get('year', 'N/A')})" 
                    for movie in self.completed_movies[-10:]  
                ]
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                
            logger.debug(f"💾 Checkpoint saved: {len(self.completed_movies)} movies")
            
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

    def load_checkpoint(self):
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                self.completed_movies = checkpoint.get('completed_movies', [])
                self.failed_movies = checkpoint.get('failed_movies', [])
                self.current_category_index = checkpoint.get('current_category_index', 0)
                self.movies_in_current_category = checkpoint.get('movies_in_current_category', 0)
                self.total_cost = checkpoint.get('total_cost', 0.0)
                self.duplicate_attempts = checkpoint.get('duplicate_attempts', 0)
                
                # Restore seen movies
                seen_list = checkpoint.get('seen_movies', [])
                self.seen_movies.update(seen_list)
                
                timestamp = checkpoint.get('timestamp', 'unknown')
                logger.info(f" Resumed from checkpoint: {len(self.completed_movies)} movies completed")
                logger.info(f"Last saved: {timestamp}")
                return True
                
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
        
        return False

    def log_movie_completion(self, movie: Dict):
        self.completed_movies.append({
            'title': movie['title'],
            'year': movie.get('year'),
            'director': movie.get('director'),
            'timestamp': datetime.now().isoformat()
        })
        
        # Save checkpoint every 5 movies
        if len(self.completed_movies) % 5 == 0:
            self.save_checkpoint()

    def generate_and_process_movie(self, prompt: str) -> Dict:
        max_retries = 8 
        
        for attempt in range(max_retries):
            try:
                # if too many duplicates, try randomizing category
                if attempt >= 4: 
                    random_categories = ["drama", "comedy", "thriller", "sci-fi", "romance", "horror", "documentary", "foreign", "indie", "classic"]
                    original_prompt = prompt
                    prompt = random.choice(random_categories)
                
                basic_movie = self.generate_single_movie(prompt)
                if not basic_movie:
                    continue
                
                movie_title = basic_movie['title']
                movie_year = basic_movie.get('year', 0)
                movie_key = (movie_title.lower().strip(), movie_year)
                
                if movie_key in self.seen_movies:
                    self.duplicate_attempts += 1
                    logger.info(f"🔄 Exact duplicate: {movie_title} ({movie_year}) - Retrying ({attempt + 1}/{max_retries})")
                    continue
                
                # Check fuzzy duplicates
                if self.is_similar_movie(movie_title, movie_year):
                    self.duplicate_attempts += 1
                    logger.info(f"🔄 Similar duplicate: {movie_title} ({movie_year}) - Retrying ({attempt + 1}/{max_retries})")
                    continue
                
                self.seen_movies.add(movie_key)
                
                analysis = self.enrich_movie(basic_movie)
                basic_movie['enrichment_response'] = analysis  # Frontend expects this field
                basic_movie['description'] = basic_movie['plot']  # API expects this field
                
                basic_movie['structured_enrichment'] = {
                    'themes': basic_movie.get('genres', [])[:3],
                    'significance': f"A {basic_movie.get('year')} film by {basic_movie.get('director')}"
                }
                
                embedding = self.create_embedding(basic_movie, analysis)
                basic_movie['embedding'] = embedding
                
                self.store_movie(basic_movie)
                
                self.log_movie_completion(basic_movie)
                
                logger.info(f"✅ Completed: {basic_movie['title']} ({basic_movie.get('year')})")
                return basic_movie
                
            except Exception as e:
                logger.error(f" Attempt {attempt + 1} failed: {e}")
                continue
        
        logger.warning(f" Failed to generate unique movie after {max_retries} attempts")
        return None

    def generate_single_movie(self, category: str) -> Dict:
        
        recent_seen = list(self.seen_movies)[-100:] if len(self.seen_movies) > 100 else list(self.seen_movies)
        exclusion_text = ""
        if recent_seen:
            exclusion_movies = [f"{title.title()} ({year})" for title, year in recent_seen]
            exclusion_text = f"""
            ABSOLUTELY FORBIDDEN - DO NOT SUGGEST ANY OF THESE {len(exclusion_movies)} MOVIES:
            {', '.join(exclusion_movies[:50])}
            {"... and " + str(len(exclusion_movies) - 50) + " more are also banned" if len(exclusion_movies) > 50 else ""}

            ESPECIALLY AVOID: The Fall, Timecrimes, Coherence, The Man from Earth, Stalker
            PICK SOMETHING COMPLETELY DIFFERENT AND UNIQUE
            TRY INTERNATIONAL CINEMA, DOCUMENTARIES, OR OBSCURE INDIE FILMS
            """
        
        # Add randomness to the prompt to encourage variety
        variety_prompts = [
            "Focus on international cinema from Asia, Europe, or South America",
            "Choose from documentaries, indie films, or art house cinema", 
            "Pick from 1970s-1990s hidden gems or cult classics",
            "Select from Criterion Collection or festival circuit films",
            "Choose experimental, avant-garde, or foreign language films",
            "Pick from underrated directors or debut features"
        ]
        variety_hint = random.choice(variety_prompts)
        
        time_periods = ["1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
        time_hint = random.choice(time_periods)
        
        prompt = f"""Generate 1 {category} movie with this EXACT format - NO MARKDOWN, NO ASTERISKS, NO BOLD TEXT:

        {exclusion_text}
        🎨 VARIETY FOCUS: {variety_hint}
        ⏰ CONSIDER: Films from the {time_hint}

        TITLE: [Movie Title]
        YEAR: [Release Year]  
        DIRECTOR: [Director Name]
        CAST: [Actor1, Actor2, Actor3, Actor4]
        GENRES: [Genre1, Genre2, Genre3]
        PLOT: [2-3 sentence plot description]

        CRITICAL REQUIREMENTS:
        - Use PLAIN TEXT ONLY - no **, no *, no _underscores_, no "quotes"
        - Well-known, critically acclaimed film that actually exists  
        - Include main cast members (4 actors)
        - Real director name
        - Must be COMPLETELY DIFFERENT from any movies listed above
        - NO REPEATS OF PREVIOUSLY MENTIONED FILMS
        - Choose diverse, lesser-known gems from various countries/eras
        - Give exact release year (4 digits)
        - BE CREATIVE AND UNPREDICTABLE IN YOUR CHOICES

        EXAMPLE FORMAT:
        TITLE: The Bicycle Thief
        YEAR: 1948
        DIRECTOR: Vittorio De Sica
        CAST: Lamberto Maggiorani, Enzo Staiola, Lianella Carell, Gino Saltamerenda
        GENRES: Drama, Neorealism, Italian Cinema
        PLOT: A poor man searches Rome for his stolen bicycle without which he will lose his job."""

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=1.2  # high tmep to prevent getting stuck on same movies
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw LLM response:\n{content}")
            
            movie = self.parse_movie_response(content)
            
            if movie:
                logger.debug(f"Parsed movie: {movie['title']} ({movie.get('year')}) by {movie.get('director')}")
            
            # Track cost
            cost = (response.usage.prompt_tokens / 1_000_000) * 0.15 + \
                   (response.usage.completion_tokens / 1_000_000) * 0.60
            self.total_cost += cost
            
            return movie
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.replace('**', '').replace('*', '').replace('_', '')
        text = text.replace('"', '').replace("'", "")
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'^\d+\.\s*', '', text)
        
        return text

    def parse_movie_response(self, content: str) -> Dict:
        movie = {}
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('TITLE:'):
                raw_title = line.split(':', 1)[1].strip()
                movie['title'] = self.clean_text(raw_title)
            elif line.startswith('YEAR:'):
                try:
                    year_text = self.clean_text(line.split(':', 1)[1])
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                    movie['year'] = int(year_match.group()) if year_match else 2000
                except:
                    movie['year'] = 2000
            elif line.startswith('DIRECTOR:'):
                raw_director = line.split(':', 1)[1].strip()
                movie['director'] = self.clean_text(raw_director)
            elif line.startswith('CAST:'):
                cast_str = line.split(':', 1)[1].strip()
                movie['cast'] = [self.clean_text(c) for c in cast_str.split(',') if c.strip()]
            elif line.startswith('GENRES:'):
                genres_str = line.split(':', 1)[1].strip()
                movie['genres'] = [self.clean_text(g) for g in genres_str.split(',') if g.strip()]
            elif line.startswith('PLOT:'):
                raw_plot = line.split(':', 1)[1].strip()
                movie['plot'] = self.clean_text(raw_plot)
        
        # Validate we got essential fields
        if not movie.get('title') or not movie.get('title').strip():
            logger.warning(f"Invalid title parsed from LLM response: {content[:200]}...")
            return None
            
        if not movie.get('year') or movie.get('year') < 1900 or movie.get('year') > 2030:
            logger.warning(f"Invalid year for {movie['title']}: {movie.get('year')}")
            movie['year'] = 2000  
        
        # Add metadata
        movie['created_at'] = datetime.now()
        movie['processing_status'] = 'enriched'  
        movie['ai_provider'] = 'openai'
        movie['source'] = 'streamlined_generated'
        movie['id'] = f"{movie.get('title', '').lower().replace(' ', '_')}_{movie.get('year', 2000)}"
        
        return movie

    def enrich_movie(self, movie: Dict) -> str:
        prompt = f"""Analyze this movie and write a rich, detailed description:

        Movie: {movie['title']} ({movie.get('year')})
        Director: {movie.get('director')}
        Genres: {', '.join(movie.get('genres', []))}
        Plot: {movie.get('plot')}

        Write a comprehensive 150-200 word analysis covering:
        - Themes and significance
        - Style and cinematography  
        - Cultural impact
        - Why it's worth watching

        Write in an engaging, informative style."""

        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5  
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Track cost
            cost = (response.usage.prompt_tokens / 1_000_000) * 0.15 + \
                   (response.usage.completion_tokens / 1_000_000) * 0.60
            self.total_cost += cost
            
            return analysis
            
        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            return f"Analysis for {movie['title']} - A {movie.get('year')} film directed by {movie.get('director')}."

    def create_embedding(self, movie: Dict, analysis: str) -> List[float]:
        try:
            # Combine all text for embedding
            text = f"""
            Title: {movie['title']}
            Year: {movie.get('year')}
            Director: {movie.get('director')}
            Cast: {', '.join(movie.get('cast', [])[:5])}
            Genres: {', '.join(movie.get('genres', []))}
            Plot: {movie.get('plot', '')}
            Analysis: {analysis}
            """
            
            response = self.ai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text.strip()
            )
            
            cost = (1 / 1_000_000) * 0.10 
            self.total_cost += cost
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    def store_movie(self, movie: Dict):
        try:
            # Use upsert to avoid duplicates
            filter_key = {
                'title': movie['title'],
                'year': movie.get('year')
            }
            
            self.collection.update_one(
                filter_key,
                {'$set': movie},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Storage failed for {movie['title']}: {e}")

    def curate_movies(self, target_count: int = 1000):
        categories = [
            "classic Hollywood films",
            "international cinema masterpieces", 
            "modern acclaimed dramas",
            "sci-fi and fantasy essentials",
            "thriller and horror classics",
            "comedy favorites",
            "action and adventure movies",
            "animated and family films",
            "indie and arthouse cinema",
            "award-winning films"
        ]
        
        movies_per_category = target_count // len(categories)
        completed = len(self.completed_movies)  # Start from resumed count
        
        start_category = self.current_category_index
        
        logger.info(f"{'Resuming' if completed > 0 else 'Starting'} curation for {target_count} movies")
        logger.info(f"Current progress: {completed}/{target_count} movies completed")
        if start_category > 0:
            logger.info(f"Resuming from category {start_category + 1}: {categories[start_category]}")
        
        with tqdm(total=target_count, desc=" Curating movies", initial=completed) as pbar:
            for category_idx, category in enumerate(categories):
                # Skip categories we've already completed
                if category_idx < start_category:
                    continue
                    
                self.current_category_index = category_idx
                logger.info(f"📝 Processing: {category}")
                
                movies_needed = movies_per_category
                if category_idx == start_category:
                    movies_needed = movies_per_category - self.movies_in_current_category
                
                for movie_in_category in range(movies_needed):
                    if completed >= target_count:
                        break
                    
                    if completed % 50 == 0 and completed > 0:
                        self.refresh_seen_movies()
                        
                    movie = self.generate_and_process_movie(category)
                    if movie:
                        completed += 1
                        self.movies_in_current_category = (movie_in_category + 1) if category_idx == start_category else (movie_in_category + 1)
                        pbar.update(1)
                        success_rate = (completed / (completed + self.duplicate_attempts)) * 100 if completed + self.duplicate_attempts > 0 else 100
                        pbar.set_postfix({
                            'completed': completed,
                            'duplicates': self.duplicate_attempts,
                            'success': f"{success_rate:.0f}%",
                            'cost': f"${self.total_cost:.2f}",
                            'category': category[:15] + "..." if len(category) > 15 else category
                        })
                    
                    time.sleep(0.05) 
                
                self.movies_in_current_category = 0
                
                if completed >= target_count:
                    break
        
        # Final checkpoint save
        self.save_checkpoint()
        
        # Final stats
        total_in_db = self.collection.count_documents({})
        success_rate = (completed / (completed + self.duplicate_attempts)) * 100 if completed + self.duplicate_attempts > 0 else 0
        

    @staticmethod
    def list_sessions():
        progress_files = [f for f in os.listdir('.') if f.startswith('curator_progress_') and f.endswith('.json')]
        
        if not progress_files:
            print("No saved sessions found.")
            return []
        
        print("\n Available Sessions:")
        print("=" * 60)
        
        sessions = []
        for progress_file in sorted(progress_files):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                session_id = progress.get('session_id', 'unknown')
                completed = progress.get('completed_count', 0)
                cost = progress.get('total_cost', 0)
                last_updated = progress.get('last_updated', 'unknown')
                
                sessions.append(session_id)
                
            except Exception as e:
                print(f" Error reading {progress_file}: {e}")
        
        return sessions

    @staticmethod
    def clean_sessions(keep_recent=3):
        progress_files = [f for f in os.listdir('.') if f.startswith('curator_progress_') and f.endswith('.json')]
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('curator_checkpoint_') and f.endswith('.pkl')]
        
        all_files = progress_files + checkpoint_files
        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        sessions_to_keep = set()
        for f in all_files[:keep_recent * 2]:  
            if f.startswith('curator_progress_'):
                sessions_to_keep.add(f.replace('curator_progress_', '').replace('.json', ''))
            elif f.startswith('curator_checkpoint_'):
                sessions_to_keep.add(f.replace('curator_checkpoint_', '').replace('.pkl', ''))
        
        deleted_count = 0
        for f in all_files:
            session_id = None
            if f.startswith('curator_progress_'):
                session_id = f.replace('curator_progress_', '').replace('.json', '')
            elif f.startswith('curator_checkpoint_'):
                session_id = f.replace('curator_checkpoint_', '').replace('.pkl', '')
            
            if session_id and session_id not in sessions_to_keep:
                try:
                    os.remove(f)
                    deleted_count += 1
                except Exception as e:
        
        print(f" Cleaned {deleted_count} old session files, kept {len(sessions_to_keep)} recent sessions.")

    def show_sample(self):
        movies = list(self.collection.find({}).limit(3))
        for movie in movies:
            print(f"\n {movie['title']} ({movie.get('year')})")
            print(f"Director: {movie.get('director')}")
            print(f"Cast: {', '.join(movie.get('cast', [])[:3])}")
            print(f"Genres: {', '.join(movie.get('genres', []))}")
            print(f"Plot: {movie.get('plot', '')}")
            print(f"Analysis: {movie.get('enrichment_response', '')[:100]}...")
            print(f"Embedding: {len(movie.get('embedding', []))} dimensions")
            print(f"Status: {movie.get('processing_status')}")

    def show_existing_movies(self, limit: int = 0):
        total = len(self.seen_movies)
        print(f"\nExisting Movies ({total} total):")
        print("=" * 50)
        sorted_movies = sorted(list(self.seen_movies), key=lambda x: (x[1], x[0]))
        
        movies_to_show = sorted_movies[-limit:] if limit > 0 else sorted_movies
        
        for title, year in movies_to_show:
            print(f"• {title.title()} ({year})")
        
        if limit > 0 and total > limit:
            print(f"... and {total - limit} more movies")
        print("=" * 50)

    def debug_duplicates(self, check_title: str, check_year: int):
        print(f"\n Debugging duplicates for: '{check_title}' ({check_year})")
        print("=" * 60)
        
        clean_title = check_title.lower().strip()
        movie_key = (clean_title, check_year)
        
        print(f"Cleaned title: '{clean_title}'")
        print(f"Movie key: {movie_key}")
        print(f"Exact match: {movie_key in self.seen_movies}")
        print(f"Fuzzy match: {self.is_similar_movie(check_title, check_year)}")
        
        # Show similar movies
        print(f"\nSimilar movies (threshold 0.85):")
        for existing_title, existing_year in self.seen_movies:
            if abs(check_year - existing_year) <= 1:
                similarity = SequenceMatcher(None, clean_title, existing_title).ratio()
                if similarity > 0.7:  
                    print(f"  {similarity:.2f}: '{existing_title}' ({existing_year})")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Streamlined Movie Curator with Resume Support')
    parser.add_argument('--count', type=int, default=500,
                       help='Number of movies to curate (default: 500)')
    parser.add_argument('--sample', action='store_true',
                       help='Show sample of existing movies')
    parser.add_argument('--list', action='store_true',
                       help='Show all existing movies for debugging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging to see LLM responses')
    parser.add_argument('--check-dupe', nargs=2, metavar=('TITLE', 'YEAR'),
                       help='Debug duplicate detection for specific movie')
    
    # Resume/session management
    parser.add_argument('--resume', type=str, metavar='SESSION_ID',
                       help='Resume from a specific session ID')
    parser.add_argument('--sessions', action='store_true',
                       help='List all available sessions that can be resumed')
    parser.add_argument('--clean-sessions', type=int, metavar='KEEP_COUNT', nargs='?', const=3,
                       help='Clean old session files, keeping most recent N (default: 3)')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Handle session management commands
    if args.sessions:
        StreamlinedMovieCurator.list_sessions()
        return
    
    if args.clean_sessions is not None:
        StreamlinedMovieCurator.clean_sessions(args.clean_sessions)
        return
    
    # Create curator (with resume if specified)
    curator = StreamlinedMovieCurator(session_id=args.resume)
    
    if args.sample:
        curator.show_sample()
        return
    
    if args.list:
        curator.show_existing_movies()
        return
    
    if args.check_dupe:
        try:
            check_year = int(args.check_dupe[1])
            curator.debug_duplicates(args.check_dupe[0], check_year)
        except ValueError:
            print(f"❌ Invalid year: {args.check_dupe[1]}")
        return
    
    if len(curator.completed_movies) > 0:
        remaining = args.count - len(curator.completed_movies)
        if remaining > 0:
            estimated_cost = remaining * 0.08
            print(f" Target: {args.count} movies ({remaining} remaining)")
            print(f" Already completed: {len(curator.completed_movies)} movies")
            print(f" Estimated remaining cost: ${estimated_cost:.2f}")
        else:
            print(f" Target already reached! {len(curator.completed_movies)} movies completed")
            return
    else:
        # New session
        estimated_cost = args.count * 0.08
        print(f"Target: {args.count} movies")
        print(f"Estimated cost: ${estimated_cost:.2f}")
    
    print(f"Session ID: {curator.session_id}")
    print("Progress auto-saves every 5 movies. Press Ctrl+C to stop safely anytime.")
    print()
    
    response = input("Proceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted")
        return
    
    # Set up graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info(f"\n Received signal {signum}. Saving progress and shutting down gracefully...")
        curator.save_checkpoint()
        logger.info(f"Progress saved to {curator.progress_file}")
        logger.info(f"Resume with: python {sys.argv[0]} --resume {curator.session_id}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_time = time.time()
    try:
        curator.curate_movies(args.count)
        duration = time.time() - start_time
        
        print(f"\n Completed in {duration:.1f} seconds")
        if duration > 0:
            print(f" Rate: {len(curator.completed_movies)/duration:.1f} movies/second")
        print("\n Test your search:")
        print("python app.py --provider openai")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Progress has been saved.")
        logger.info(f" Session: {curator.session_id}")
        logger.info(f" Resume with: python {sys.argv[0]} --resume {curator.session_id}")
    except Exception as e:
        logger.error(f"Error during curation: {e}")
        curator.save_checkpoint()
        logger.info(f"Progress saved despite error. Resume with: python {sys.argv[0]} --resume {curator.session_id}")

if __name__ == '__main__':
    main()
