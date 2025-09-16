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

## ConSenseAI — Copilot guidance (concise)

This repository is a single-script X (Twitter) fact-check bot. The authoritative implementation is `ConSenseAI_v1.1.py` (most logic lives in that file). Use it and `README.md` as primary references.

### Big picture
- Single-process bot: reads mentions with a Tweepy read client (Bearer token) and posts with an OAuth1 client. See `authenticate()` in `ConSenseAI_v1.1.py`.
- Multi-model pipeline: the bot runs several LLMs (xAI/Grok, OpenAI, Anthropic) via `run_model()` and merges their outputs in `fact_check()`.
- Persistent state: `keys.txt` (credentials), `bot_tweets.json` (saved replies), and `last_tweet_id_consenseai.txt` (last processed id). `load_keys()` documents expected keys and format.

### Key files and entry points
- `ConSenseAI_v1.1.py` — main script. Inspect these functions first: `load_keys()`, `authenticate()`, `run_model()`, `fact_check()`, `post_reply()`, and storage helpers `load_bot_tweets()`/`save_bot_tweet()`.
- `README.md` — historical run notes; some filenames (e.g., `autogrok_v1.5.py`) are outdated — treat `ConSenseAI_v1.1.py` as source of truth.
- `keys.txt` — must contain names used by `load_keys()` (e.g. `XAPI_key`, `XAPI_secret`, `bearer_token`, `XAI_API_KEY`, `CHATGPT_API_KEY`, `ANTHROPIC_API_KEY`, optional `access_token`/`access_token_secret`).

### Project-specific conventions
- Prompting: system prompts are embedded in `fact_check()` and enforce final reply constraints (e.g., "under 450 chars", "no Markdown"). Preserve those constraints when editing prompts.
- Models list: models are declared inline in `fact_check()` as dicts with keys `name`, `client`, and `api`. To add a model, append to that list and ensure the SDK client is created similarly to existing ones.
- Shuffling: the script randomizes the order of models (`random.shuffle`) before running — tests should mock randomness or set seeds.
- Token persistence: OAuth flow will append new `access_token`/`access_token_secret` to `keys.txt` (see `authenticate()`); avoid exposing keys in commits.
- Dry-run: use `--dryrun True` to avoid posting; `dryruncheck()` gates posting behavior. Prefer dry-run for development.

### Integration and external dependencies
- Tweepy: `read_client = tweepy.Client(bearer_token=...)` (reads). `post_client = tweepy.Client(consumer_key=..., access_token=...)` (posts). If tokens missing, script runs three-legged OAuth and opens a browser to get verifier.
- LLM SDKs: uses `xai_sdk`, `openai`, and `anthropic`. Clients are instantiated inside `fact_check()` using keys from `load_keys()`.
- Search/tools: `run_model()` uses `SearchParameters` for xAI and a `tools` entry for Anthropic web search — keep tool usage when adding models that require web search.

### Practical dev & debugging tips
- Setup keys (example names in `load_keys()` docstring). Never commit real `keys.txt` to version control.
- Run quick dry-run: `python ConSenseAI_v1.1.py --dryrun True` to exercise the pipeline without posting.
- Inspect saved replies in `bot_tweets.json` after `post_reply()` runs (the script writes full reply text there).
- To test `fact_check()` in isolation, call it with a synthetic `tweet_text` and a minimal `context` dict; mock LLM clients to return deterministic responses.

### Small examples (copy/paste locations)
- Add a model: in `fact_check()` modify `models = [...]` with an entry like `{"name": "my-model", "client": my_client, "api": "openai"}` and create `my_client` as shown for `openai.OpenAI(...)`.
- Change storage filename: edit the `TWEETS_FILE` and `LAST_TWEET_FILE` constants at the top of `ConSenseAI_v1.1.py`.

### What not to assume
- `README.md` contains historical notes and older filenames; do not treat it as authoritative for current runtime behavior.
- The bot enforces character/format constraints via prompts — changing prompt wording can change runtime behavior.

If you want, I can add a small test harness that mocks the LLM clients and a CI job that runs basic linting and the dry-run flow. Tell me which you prefer and I'll iterate on this doc.