# ConSenseAI Development Guide

ConSenseAI is an X/Twitter bot that provides AI-powered fact-checking by leveraging multiple LLM models (Grok, GPT, Claude) to analyze and verify claims.

## Project Structure

**Current implementation:** `ConSenseAI_v1.3.py` (single-file bot, ~2059 lines)

### Core Components
- Authentication and API client setup (`authenticate()`)
- Tweet processing: mentions AND search-based discovery
- Conversation context gathering with caching (`get_tweet_context()`)
- Multi-model fact-checking pipeline (`fact_check()`, `run_model()`)
- Reply generation and posting with safety checks (`post_reply()`)
- Reflection/summary generation (`post_reflection_on_recent_bot_threads()`)

### Persistent State Files
- `keys.txt` — API credentials (never commit!)
- `bot_tweets.json` — Full text of bot's posted replies (up to MAX_BOT_TWEETS=1000)
- `ancestor_chains.json` — Cached conversation hierarchies (up to MAX_ANCESTOR_CHAINS=500)
- `last_tweet_id_{username}.txt` — Last processed mention ID
- `last_search_id_{username}_{search_term}.txt` — Last processed search result ID per term
- `search_reply_count_{username}.json` — Daily search reply counts (for dynamic caps)
- `sent_reply_hashes_{username}.json` — Reply deduplication hashes
- `approval_queue_{username}.json` — Human approval queue (when enabled)
- `output.log` — Rotating log file (rotates at LOG_MAX_SIZE_MB=10MB, keeps LOG_MAX_ROTATIONS=5 files)

## Key Features

### Dual Processing Paths
1. **Mention-based** (`fetch_and_process_mentions()`): Monitors @mentions of the bot
2. **Search-based** (`fetch_and_process_search()`): Proactively searches for keywords (e.g., "fascism")
   - Dynamic daily caps that increase hourly (`get_current_search_cap()`)
   - Deduplication within configurable window (`--dedupe_window_hours`)
   - Optional human approval queue (`--enable_human_approval`)

### Context Building (`get_tweet_context()`)
- **Caching strategy**: Prioritizes `ancestor_chains.json` cache before API calls
- **bot_username parameter**: Explicitly passed to avoid implicit global dependencies
- Builds complete context including:
  - Original tweet and full conversation threads
  - Ancestor chains (tweet reply hierarchies)
  - Bot's previous replies (for loop prevention via `count_bot_replies_in_conversation()`)
  - Media attachments (photos, videos) from tweets and quoted tweets
  - Retweet resolution (detects retweets and sets `reply_target_id` to retweeter's tweet)

### AI Models Integration
- **Lower-tier models** (indices 0-2): `grok-4-fast-reasoning`, `gpt-5-mini`, `claude-3-5-haiku-latest`
- **Higher-tier models** (indices 3-5): `grok-4`, `gpt-5`, `claude-sonnet-4-5`
- **Runtime behavior**:
  - Randomly shuffles 3 lower-tier models for initial analysis
  - Randomly selects 1 higher-tier model to combine responses
  - Each model runs independently with its own API client
  - Web search enabled for Grok (SearchParameters) and Claude (web_search tool)
  - Vision support for GPT and Claude (analyzes media URLs from context)

### Safety & Rate Limiting
- **Reply thresholds**: Per-thread or per-user-per-thread limits (`--reply_threshold`, `--per_user_threshold`)
- **Search safeguards**:
  - Dynamic cap increases hourly starting at `--cap_increase_time` (default 10:00)
  - Content safety heuristics (blocks "doxx", "address", "phone", "ssn", "private")
  - Deduplication via content hashes within 24h window
- **Backoff multiplier**: Exponential backoff on rate limits and errors
- **Auto-restart**: Catches critical errors and restarts after RESTART_DELAY=10s

### Reflection Posts
- Triggered every N bot replies (`--post_interval`, default 10)
- Reviews last N threads where bot participated
- Generates engagement-optimized standalone tweet
- Uses `compute_baseline_replies_since_last_direct_post()` to track post counts

## Development Workflow

### Setup
Create `keys.txt` with required API keys:
```
XAPI_key=your_x_api_key
XAPI_secret=your_x_api_secret
bearer_token=your_bearer_token
XAI_API_KEY=your_xai_api_key
CHATGPT_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TWITTERAPIIO_KEY=your_twitterapiio_key
access_token=your_access_token
access_token_secret=your_access_token_secret
```

### Running the Bot
```bash
python ConSenseAI_v1.3.py \
  --username consenseai \
  --delay 3 \
  --dryrun False \
  --search_term "fascism" \
  --search_daily_cap 12 \
  --search_cap_interval_hours 1 \
  --reply_threshold 10 \
  --per_user_threshold True \
  --post_interval 10
```

### Command Line Arguments
- `--username`: X/Twitter username to monitor (default: "ConSenseAI")
- `--delay`: Minutes between API checks (default: prompt)
- `--dryrun`: Print responses without posting (default: False)
- `--fetchthread`: Fetch full conversation context (default: True)
- `--reply_threshold`: Max replies per thread or per-user-per-thread (default: 5)
- `--per_user_threshold`: If True, enforce threshold per user; if False, per thread (default: True)
- `--search_term`: Keyword to search for proactive responses (default: None)
- `--search_max_results`: Max results per search query (default: 10)
- `--search_daily_cap`: Max automated search replies per day (default: 5)
- `--search_cap_interval_hours`: Hours between cap increases (default: 2)
- `--cap_increase_time`: Earliest time for cap increases, HH:MM format (default: "10:00")
- `--dedupe_window_hours`: Deduplication window in hours (default: 24.0)
- `--enable_human_approval`: Queue replies for approval instead of auto-posting (default: False)
- `--post_interval`: Number of replies between reflection posts (default: 10)

## Architecture & Flow

### Authentication Flow
- Uses Tweepy with both OAuth 1.0a (for posting) and Bearer Token (for reading)
- `read_client`: Bearer token (app-only, basic-tier app)
- `post_client`: OAuth 1.0a (posts as @ConSenseAI, free tier)
- Auto-handles token refresh via three-legged OAuth if tokens missing/expired
- Caches `BOT_USER_ID` to avoid repeated API calls

### Main Loop (in `main()`)
1. Authenticate and initialize state (search caps, deduplication, reflection baseline)
2. **Each cycle:**
   - Call `fetch_and_process_mentions()` — check for new @mentions
   - Call `fetch_and_process_search()` (if `--search_term` provided) — proactive discovery
   - Check if reflection post should be triggered (based on `--post_interval`)
   - Sleep for `delay * backoff_multiplier` minutes
3. **On critical error:** Auto-restart after RESTART_DELAY

### Context Building Flow (`get_tweet_context()`)
1. Check `ancestor_chains.json` cache for conversation_id
2. If cache hit: load ancestor chain, thread_tweets, bot_replies, media from cache
3. If cache miss or incomplete:
   - Detect retweets: resolve original tweet and set `reply_target_id` to retweeter's tweet
   - Build ancestor chain: walk up reply hierarchy via `in_reply_to_user_id`
   - Extract media from all tweets in chain (including quoted tweets)
   - Fetch bot's prior replies: search `conversation_id:{conv_id} from:{bot_username}`
4. Return context dict with `ancestor_chain`, `thread_tweets`, `bot_replies_in_thread`, `media`, `reply_target_id`, `mention_full_text`

### Fact-Check Pipeline (`fact_check()`)
1. Construct context string from `ancestor_chain`, `thread_tweets`, `quoted_tweets`, `media`
2. Try `get_full_text_twitterapiio()` for full tweet text (falls back to standard Tweepy text)
3. Initialize LLM clients (xai_sdk, openai, anthropic)
4. **First pass:** Run 3 randomized lower-tier models via `run_model()`
5. **Second pass:** Combine responses using 1 random higher-tier model
6. Append model attribution to final response
7. If `generate_only=True`: return text only (for search pipeline)
8. Else: post reply via `post_reply()` (respects dryrun)

### Post Reply Flow (`post_reply()`)
1. Create tweet via `post_client.create_tweet()`
2. Store full reply text in `bot_tweets.json` via `save_bot_tweet()`
3. Append reply to `ancestor_chains.json` cache via `append_reply_to_ancestor_chain()`
4. Return `'done!'` on success or `'delay!'` on 429 rate limit

## Common Issues & Lessons Learned

### 1. Missing Ancestor Chains in Prompts
**Symptoms:** Logs show "Ancestor chain doesn't seem to be appearing" in fact-check prompts.

**Root causes identified:**
- **Dynamic search cap skipping runs:** When `get_current_search_cap()` returns a cap ≤ current daily count, search processing is skipped entirely (check logs for `"[Search] Current cap reached (...), skipping processing"`).
- **External full-text provider 401 errors:** twitterapi.io returning `401 Unauthorized` causes fallback to shorter Tweepy text, which may be truncated for retweets.
- **Implicit global `username` dependency:** `get_tweet_context()` originally used `from:{username}` without explicit parameter passing, causing inconsistent bot-replies fetching between mention and search flows.

**Fixes applied:**
- Added explicit `bot_username` parameter to `get_tweet_context()` (passed from both `fetch_and_process_mentions()` and `fetch_and_process_search()`)
- Ensured both callers pass `bot_username=username` to avoid implicit global reliance

**Recommended next steps:**
- Add debug log inside `get_tweet_context()` showing which `bot_username` was used and how many bot_replies were found
- Run dry-run test for search flow and print generated context to verify ancestor_chain population

### 2. Retweet Handling
**Symptoms:** "When the search function replies to a retweet, it is not getting the whole retweet text and is replying to the original tweet."

**Root cause:** When a tweet is a retweet, the API may return truncated text and the wrong target for replies.

**Fix applied:** `get_tweet_context()` now detects retweets via `referenced_tweets` (type="retweeted"), resolves the original tweet, extracts full text, and sets `context['reply_target_id']` to the retweeter's tweet (not the original). The full text is stored in `context['mention_full_text']` for use in `fact_check()`.

### 3. Provider 401 Errors
**Symptoms:** Repeated `"Error fetching tweet {id} from twitterapi.io: 401 Client Error: Unauthorized"` in logs.

**Context:** twitterapi.io is an **optional** full-text provider. When it fails, the bot falls back to Tweepy's standard text (which may be truncated).

**Current behavior:** The bot continues to function; shorter text may reduce context quality but does not break the pipeline.

**Recommended next steps:**
- Verify `TWITTERAPIIO_KEY` is valid and not expired
- Consider reordering fallbacks to prefer Tweepy includes before external provider
- Add explicit logging showing which source provided final text for each mention

## Project Conventions

### Code Style
- **Minimal edits preferred:** User requests focused, safe changes over large refactors
- **Defensive coding:** Use `getattr()`, `hasattr()`, `isinstance()` checks for Tweepy objects (which may be dicts or objects depending on context)
- **Explicit parameters:** Avoid implicit globals; pass required values as function parameters (e.g., `bot_username` in `get_tweet_context()`)

### Error Handling
- Extensive use of try/except blocks for API resilience
- Exponential backoff on rate limits (`backoff_multiplier`)
- Automatic script restart on critical errors (ConnectionError, TweepyException, generic Exception)

### Caching Strategy
- **Ancestor chains:** Stored in `ancestor_chains.json` with conversation_id as key
  - Format: `{"conv_id": {"chain": [...], "thread_tweets": [...], "bot_replies": [...]}}` or legacy `{"conv_id": [...]}`
- **Bot tweets:** Stored in `bot_tweets.json` with tweet_id as key
  - Format: `{"tweet_id": "full_tweet_text", ...}`
- **Pruning:** When cache exceeds MAX_ANCESTOR_CHAINS or MAX_BOT_TWEETS, oldest entries are removed

### Prompting Constraints
- System prompts embedded in `fact_check()` enforce:
  - "under 450 chars" (do not mention character length in response)
  - "no Markdown formatting"
  - "Respond in same language as the post"
- **Critical:** Preserve these constraints when editing prompts; changing wording affects runtime behavior

## Integration Points

### External APIs
- **X/Twitter API v1.1** via Tweepy: `search_recent_tweets()`, `get_users_mentions()`, `get_tweet()`, `create_tweet()`
- **xAI/Grok API** with `SearchParameters` for web search (via `xai_sdk`)
- **OpenAI GPT API** with vision support (via `openai` SDK)
- **Anthropic Claude API** with web_search tool (via `anthropic` SDK)
- **twitterapi.io** (optional): Full-text tweet provider via `get_full_text_twitterapiio()` (configured via `TWITTERAPIIO_KEY`)

### Data Flow
1. Monitor mentions (`fetch_and_process_mentions()`) and/or search (`fetch_and_process_search()`)
2. Gather conversation context (`get_tweet_context()`) with caching
3. Run parallel model analysis (`run_model()` x3 lower-tier models)
4. Combine responses (`run_model()` x1 higher-tier model)
5. Post reply with attribution (`post_reply()`)
6. Update caches (`save_bot_tweet()`, `append_reply_to_ancestor_chain()`)
7. Periodically post reflection (`post_reflection_on_recent_bot_threads()`)

## Debugging Tips

### Inspecting Logs
- Check `output.log` for runtime behavior (rotates automatically)
- Key diagnostic strings:
  - `"[Search] Current cap reached"` — search processing skipped due to daily cap
  - `"Error fetching tweet ... from twitterapi.io: 401"` — external provider failure
  - `"[Ancestor Chain] Tweet ... text length in build: X chars"` — confirm ancestor chain building
  - `"[Context Cache] Loaded cached data for conversation"` — cache hit
  - `"[DEBUG] Full mention text length in fact_check: X chars"` — text size in pipeline

### Dry-Run Testing
```bash
python ConSenseAI_v1.3.py --dryrun True --delay 1 --search_term "test"
```
- Exercises full pipeline without posting
- Prints generated responses to console
- Safe for testing context building, model orchestration, reply generation

### Cache Inspection
- `bot_tweets.json`: Check which tweets the bot has posted (by ID)
- `ancestor_chains.json`: Inspect cached conversation hierarchies
- `search_reply_count_{username}.json`: View daily search reply counts by date

### Function Entry Points for Testing
- `load_keys()` — credential loading
- `authenticate()` — API client setup
- `get_tweet_context(tweet, includes, bot_username)` — context building (call with mock tweet objects)
- `fact_check(tweet_text, tweet_id, context, generate_only)` — LLM orchestration (mock clients for deterministic testing)
- `run_model(system_prompt, user_msg, model, verdict, context)` — single-model invocation

### Common Gotchas
- **Tweepy object shapes:** Returned objects may be Tweepy Response objects or dicts; use `getattr()` with fallbacks
- **Global username:** Search flow may not have `username` in scope; always pass explicitly to `get_tweet_context(bot_username=...)`
- **Retweet detection:** Check `referenced_tweets` for type="retweeted" to detect retweets and resolve original
- **Media extraction:** Media may be in `includes['media']` (when expansions used) or embedded in tweet entities; check both

## Small Code Examples

### Add a New Model
In `fact_check()`, modify the `models` list:
```python
models = [
    # lower tier
    {"name": "grok-4-fast-reasoning", "client": xai_client, "api": "xai"},
    {"name": "gpt-5-mini", "client": openai_client, "api": "openai"},
    {"name": "claude-3-5-haiku-latest", "client": anthropic_client, "api": "anthropic"},
    {"name": "my-new-model", "client": my_client, "api": "openai"},  # NEW
    # higher tier
    {"name": "grok-4", "client": xai_client, "api": "xai"},
    ...
]
```
Create the client earlier in `fact_check()`:
```python
my_client = openai.OpenAI(api_key=keys.get('MY_MODEL_API_KEY'), base_url="https://api.mymodel.com/v1")
```

### Change Storage Filenames
Edit constants near top of `ConSenseAI_v1.3.py`:
```python
TWEETS_FILE = 'bot_tweets.json'
ANCESTOR_CHAIN_FILE = 'ancestor_chains.json'
```

### Add Debug Logging to Context Building
Inside `get_tweet_context()`, after bot-replies search:
```python
print(f"[get_tweet_context] bot_username={bot_username or 'None'}, bot_replies_found={len(context['bot_replies_in_thread'])}")
```

## What NOT to Assume
- `README.md` contains historical notes and older filenames; do not treat it as authoritative for current runtime behavior
- The bot enforces character/format constraints via prompts — changing prompt wording can change runtime behavior
- Search processing may be skipped frequently due to dynamic caps; check logs to confirm runs
- twitterapi.io may fail intermittently; the bot will fall back to standard Tweepy text

## Future Improvements (Optional)
- Add instrumentation logs showing which source provided mention_full_text (twitterapi.io vs Tweepy)
- Run dry-run end-to-end tests for retweet handling and ancestor chain building
- Consider inverting full-text resolution fallback order (prefer includes/official API before external provider)
- Add unit tests with mocked LLM clients and Tweepy responses
- Implement CI job for linting and dry-run smoke tests