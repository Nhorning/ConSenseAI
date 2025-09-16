# ConSenseAI Development Guide

ConSenseAI is an X/Twitter bot that provides AI-powered fact-checking by leveraging multiple LLM models (Grok, GPT, Claude) to analyze and verify claims.

## Project Structure

- `ConSenseAI_v1.1.py`: Main bot implementation containing:
  - Authentication and API client setup
  - Tweet processing and conversation context gathering
  - Multi-model fact-checking pipeline
  - Reply generation and posting

## Key Components

### Authentication Flow
- Uses Tweepy with both OAuth 1.0a (for posting) and Bearer Token (for reading)
- Credentials stored in `keys.txt` (see format in `load_keys()` docstring)
- Auto-handles token refresh and re-authentication

### AI Models Integration
- Supports parallel execution of multiple LLMs (xAI/Grok, OpenAI/GPT, Anthropic/Claude)
- Each model runs independently with its own API client
- Models are shuffled randomly to prevent bias
- Responses are combined using an additional model run

### Context Management
- Builds comprehensive thread context including:
  - Original tweets and full conversation threads
  - Ancestor chains (tweet reply hierarchies)
  - Bot's previous replies (for loop prevention)

## Development Workflow

### Setup
1. Create `keys.txt` with required API keys:
```
XAPI_key=your_x_api_key
XAPI_secret=your_x_api_secret
bearer_token=your_bearer_token
XAI_API_KEY=your_xai_api_key
CHATGPT_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Running the Bot
```bash
python ConSenseAI_v1.1.py [--username USERNAME] [--delay MINUTES] [--dryrun BOOL] [--accuracy THRESHOLD] [--fetchthread BOOL]
```

### Command Line Arguments
- `--username`: X/Twitter username to monitor (default: "consenseai")
- `--delay`: Minutes between API checks (default: prompt)
- `--dryrun`: Print responses without posting (default: False)
- `--accuracy`: Skip replies above this threshold (default: 4)
- `--fetchthread`: Fetch full conversation context (default: False)

## Project Conventions

### Error Handling
- Extensive use of try/except blocks for API resilience
- Exponential backoff on rate limits (`backoff_multiplier`)
- Automatic script restart on critical errors

### Rate Limiting
- 30-second delay between replies
- Configurable check interval with backoff
- Thread reply limit (5 replies per thread)

### Bot Response Format
- Combined analysis from multiple models
- Attribution of models used
- Character limit enforcement (450 chars)

## Integration Points

### External APIs
- X/Twitter API v1.1 via Tweepy
- xAI/Grok API with search capability
- OpenAI GPT API
- Anthropic Claude API with web search

### Data Flow
1. Monitor mentions via Twitter API
2. Gather conversation context
3. Run parallel model analysis
4. Combine responses
5. Post reply with attribution