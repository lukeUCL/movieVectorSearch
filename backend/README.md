# Backend Architecture

Production-ready modular structure for the movie search API.

## Structure

```
backend/
├── config.py      # Configuration & constants
├── database.py    # MongoDB operations  
├── ai_service.py  # AI operations (OpenAI/Gemini)
├── search.py      # Search functionality
└── routes.py      # Flask endpoints
```

## Usage

```bash
# New interface (recommended)
python main.py --provider openai --port 5001

# Backward compatibility
python app.py

# Debug mode
python main.py --debug
```

## Architecture Benefits

- **Separation of concerns** - each module has a single responsibility
- **Easy testing** - modules can be tested independently  
- **Configuration management** - centralized in config.py
- **Provider flexibility** - easy to switch AI providers
- **Clean imports** - no circular dependencies
