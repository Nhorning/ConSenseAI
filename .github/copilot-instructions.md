# ConSenseAI Development Guide

ConSenseAI is an X/Twitter bot that provides AI-powered fact-checking by leveraging multiple LLM models (Grok, GPT, Claude) to analyze and verify claims.

## Project Architecture

**Implementation:** Single-file bot `ConSenseAI_v1.6.py` (~6000 lines)  
**Active Branch:** `main`  
**Language:** Python 3.x with Tweepy, OpenAI, Anthropic, and xAI SDKs

### Core Pipeline
1. **Discovery:** 4 parallel paths (mentions, search, followed users, Community Notes)
2. **Context Building:** Cache-first with ancestor chain traversal (`get_tweet_context()`)
3. **Analysis:** 3 lower-tier models → 1 higher-tier combiner (`fact_check()`, `run_model()`)
4. **Response:** Post with attribution + cache updates (`post_reply()`)
5. **Reflection:** Periodic standalone tweets (`post_reflection_on_recent_bot_threads()`)


### Critical State Files
```
keys.txt                    # API credentials (NEVER commit!)
bot_tweets.json             # Bot replies (max 1000, oldest pruned)
ancestor_chains.json        # Conversation hierarchies (max 500)
last_tweet_id_*.txt         # Processing cursors per source
*_reply_count_*.json        # Daily caps tracking
sent_reply_hashes_*.json    # Deduplication hashes
cn_written_*.json           # Community Notes + scores
output.log                  # Rotating logs (10MB, 5 files)
```

## Critical Patterns

### 1. Tweepy Object Type Handling ⚠️
**Most common bug source.** API responses return dicts OR objects unpredictably.

```python
# ALWAYS handle both formats
if isinstance(obj, dict):
    value = obj.get('field', default)
elif hasattr(obj, 'field'):
    value = obj.field
else:
    value = default

# Use helper function
def get_attr(obj, attr, default=None):
    return obj.get(attr, default) if isinstance(obj, dict) else getattr(obj, attr, default)
```

**High-risk locations:** `includes`, `referenced_tweets`, cached data, user/media objects

### 2. Infinite Loop Prevention (KNOWN BUG)
Lines ~3500-3700 in `get_tweet_context()`: ancestor chain building loop lacks depth limit.

```python
# CRITICAL FIX NEEDED:
MAX_ANCESTOR_DEPTH = 20  # Add this constant

while True:
    if len(ancestor_chain) >= MAX_ANCESTOR_DEPTH:  # ADD THIS
        break
    try:
        parent_tweet = read_client.get_tweet(...)
    except tweepy.TweepyException as e:
        print(f"Error: {e}")
        break  # MUST break on errors, not continue
```

**Trigger:** Deep threads (50+ replies) cause rate limit death spiral.

### 3. Context Cache Format
`ancestor_chains.json` has two formats (legacy + current):
```python
{
  "conv_id": [...],  # Legacy: just chain
  "conv_id": {       # Current: full context
    "chain": [...],
    "thread_tweets": [...],
    "bot_replies": [...],
    "media": [...]
  }
}
```
Always check format before accessing fields.


## Key Features

### Quadruple Processing Paths
1. **Mention-based** (`fetch_and_process_mentions()`): Monitors @mentions of the bot
2. **Search-based** (`fetch_and_process_search()`): Proactively searches for keywords (e.g., "fascism")
   - Dynamic daily caps that increase hourly (`get_current_search_cap()`)
   - Deduplication within configurable window (`--dedupe_window_hours`)
   - Optional human approval queue (`--enable_human_approval`)
   - Optional AI-generated search terms when `--search_term "auto"` is used
3. **Followed Users** (`fetch_and_process_followed_users()`): Checks tweets from users the bot follows
   - **Rotation system**: Cycles through N users per cycle (default 3) to avoid overwhelming rate limits
   - **Separate daily cap**: Independent from search cap (default 10 replies/day)
   - **State tracking**: `followed_rotation_{username}.json` tracks position in followers list
   - **Cap tracking**: `followed_reply_count_{username}.json` tracks daily reply counts
   - Introduced in commit 62250d7
4. **Community Notes** (`fetch_and_process_community_notes()`): Fetches posts flagged by Twitter as needing Community Notes
   - **OAuth 1.0a Authentication**: Uses separate CN project keys or falls back to main keys
   - **Test Mode**: Submits notes with `test_mode=True` flag for testing before going live
   - **Integration**: Uses existing fact_check() pipeline for note generation
   - **Deduplication**: Tracks written notes in `cn_written_{username}.json`
   - **Comprehensive Logging**: All processing details logged to `cn_notes_log_{username}.txt`
   - **Adversarial Helpfulness Verification** (`verify_note_helpfulness_adversarial()`): Optional LLM-based verification
     - Enabled via `--cn_verify_helpfulness True` flag
     - Uses last 50 historical notes (helpful vs unhelpful examples from Twitter's ratings)
     - Runs full fact_check module adversarially to predict if note would be rated helpful/unhelpful
     - Includes Twitter's Community Notes helpfulness criteria in prompt
     - If rated unhelpful: generates improvement suggestions and attempts to create improved note
     - If improved note generated: replaces original and re-validates
     - Falls back gracefully on verification errors (allows note to proceed)
   - **Configurable**: `--check_community_notes`, `--cn_max_results`, `--cn_test_mode`, `--cn_verify_helpfulness` flags

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
python ConSenseAI_v1.6.py \
  --username consenseai \
  --delay 3 \
  --dryrun False \
  --search_term "auto" \
  --search_daily_cap 14 \
  --reply_threshold 10 \
  --check_followed_users True \
  --check_community_notes False \
  --post_interval 10
```

### Key Arguments
- `--dryrun True`: Print responses without posting (for testing)
- `--search_term "auto"`: AI-generated search terms; or specific keyword
- `--reply_threshold 5`: Max replies per user per thread (with `--per_user_threshold True`)
- `--check_followed_users True`: Enable followed users rotation system
- `--check_community_notes True`: Enable Community Notes processing
- `--cn_test_mode True`: Submit CN in test mode (only visible to you)
- `--cn_verify_helpfulness True`: Enable adversarial LLM verification of note helpfulness before submission

Run `python ConSenseAI_v1.6.py --help` for full argument list.


## Architecture & Data Flow

### Authentication
- `read_client`: Bearer token (app-only, basic-tier)
- `post_client`: OAuth 1.0a (posts as @ConSenseAI, free tier)
- Auto-handles token refresh via three-legged OAuth
- Caches `BOT_USER_ID` to avoid repeated API calls

### Main Loop Execution
1. Authenticate → Initialize state (caps, deduplication, reflection baseline)
2. **Each cycle:**
   - `fetch_and_process_mentions()` — check @mentions
   - `fetch_and_process_search()` (if enabled) — proactive discovery
   - `fetch_and_process_followed_users()` (if enabled) — check followed users
   - `fetch_and_process_community_notes()` (if enabled) — CN eligible posts
   - Check reflection trigger (`--post_interval`)
   - Sleep for `delay * backoff_multiplier` minutes
3. **On critical error:** Auto-restart after RESTART_DELAY (10s)

### Context Building (`get_tweet_context()`)
1. Check `ancestor_chains.json` cache for conversation_id
2. If cache miss:
   - Detect retweets → resolve original + set `reply_target_id` to retweeter
   - Build ancestor chain: walk up reply hierarchy via `in_reply_to_user_id`
   - Extract media from all tweets (including quoted tweets)
   - Fetch bot's prior replies: search `conversation_id:{conv_id} from:{bot_username}`
3. Return context dict with `ancestor_chain`, `thread_tweets`, `bot_replies_in_thread`, `media`, `reply_target_id`

### Fact-Check Pipeline (`fact_check()`)
1. Construct context string from ancestor_chain, thread_tweets, quoted_tweets, media
2. Initialize LLM clients (xai_sdk, openai, anthropic)
3. **First pass:** 3 randomized lower-tier models → `run_model()`
4. **Second pass:** 1 random higher-tier model combines responses
5. Append model attribution to response
6. If `generate_only=True`: return text (for search/CN pipelines)
7. Else: `post_reply()` → update caches


## Common Issues & Critical Patterns

### 1. Tweepy Dict/Object Ambiguity (MOST FREQUENT BUG)
**Symptoms:** `AttributeError: 'dict' object has no attribute 'field'` or missing data extraction

**Pattern:** APIs return dicts OR objects unpredictably. Always use defensive access:
```python
# Safe pattern for referenced_tweets:
for ref in refs:
    ref_type = ref.get('type') if isinstance(ref, dict) else (ref.type if hasattr(ref, 'type') else None)
    ref_id = ref.get('id') if isinstance(ref, dict) else (ref.id if hasattr(ref, 'id') else None)
```

**High-risk locations:** `includes`, `referenced_tweets`, cached data, Community Notes API responses

### 2. Infinite Loop on Deep Threads (CRITICAL UNFIXED BUG)
**Location:** `get_tweet_context()` lines ~3500-3700

**Problem:** Exception handler in ancestor chain loop catches errors but **doesn't break**, causing infinite retries on deep threads (50+ replies).

**Impact:** Rate limit exhaustion (900 requests/15min), potential cache file corruption

**Fix needed:**
```python
MAX_ANCESTOR_DEPTH = 20  # Add constant
while True:
    if len(ancestor_chain) >= MAX_ANCESTOR_DEPTH: break  # Add depth limit
    try:
        parent_tweet = read_client.get_tweet(...)
    except tweepy.TweepyException as e:
        break  # CRITICAL: Must break, not continue
```

### 3. Retweet Handling
**Pattern:** Detect via `referenced_tweets` type="retweeted", resolve original, set `reply_target_id` to retweeter's tweet (not original).

**Location:** `get_tweet_context()` and `fetch_and_process_followed_users()`

### 4. Duplicate Reply Prevention (KNOWN ISSUE)
**Problem:** `count_bot_replies_by_user_in_conversation()` returns 0 despite prior replies (old cache entries have `author_id: None`)

**Workaround:** Separate tracking file approach recommended (`conversation_replies_{username}.json`)

### 5. Community Notes Score Calibration
**Critical:** Thresholds must match Twitter's actual ClaimOpinion bucket assignments from verification reports, NOT desired distribution percentages.

**Current thresholds** (as of Jan 12, 2026):
- High >= 0.081
- Medium >= -2.109

**Recalibrate periodically** using verification data from `cn_notes_log_*.txt`


## Project Conventions

### Code Style
- **Minimal edits preferred:** User requests focused, safe changes over large refactors
- **Defensive coding:** Use `getattr()`, `hasattr()`, `isinstance()` checks for Tweepy objects (which may be dicts or objects depending on context)
- **Explicit parameters:** Avoid implicit globals; pass required values as function parameters (e.g., `bot_username` in `get_tweet_context()`)

### CRITICAL: Tweepy Object Type Handling
**This is a recurring issue that must be handled consistently:**

Tweepy API responses can return data in **multiple formats**:
1. **Response objects** with attributes (e.g., `response.includes.users`)
2. **Dictionaries** with keys (e.g., `includes['users']`)
3. **Mixed formats** where some fields are objects and others are dicts

**Common symptoms:**
- `hasattr(obj, 'field')` returns False even though data exists
- `AttributeError: 'dict' object has no attribute 'field'`
- Usernames/fields not being extracted despite being in the response

**Required pattern for accessing Tweepy data:**
```python
# ALWAYS handle both dict and object formats
if isinstance(obj, dict):
    value = obj.get('field', default)
elif hasattr(obj, 'field'):
    value = obj.field
else:
    value = default

# For nested structures (like includes.users):
users_list = None
if isinstance(includes, dict):
    users_list = includes.get('users', [])
elif hasattr(includes, 'users'):
    users_list = includes.users

# For iterating over items that may be objects or dicts:
for item in items_list:
    item_id = item.id if hasattr(item, 'id') else item.get('id')
    item_name = item.username if hasattr(item, 'username') else item.get('username')
```

**Use the `get_attr()` helper function when available:**
```python
def get_attr(obj, attr, default=None):
    """Safely get attribute from either dict or object"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)
```

**Places where this commonly occurs:**
- `response.includes` (dict or object)
- Tweet objects in cached vs fresh data
- User objects from expansions
- Media objects
- Any data retrieved from `ancestor_chains.json` or `bot_tweets.json`

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
python ConSenseAI_v1.6.py --dryrun True --delay 1 --search_term "test"
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
Edit constants near top of `ConSenseAI_v1.6.py`:
```python
TWEETS_FILE = 'bot_tweets.json'
ANCESTOR_CHAIN_FILE = 'ancestor_chains.json'
```

### Add Debug Logging to Context Building
Inside `get_tweet_context()`, after bot-replies search:
```python
print(f"[get_tweet_context] bot_username={bot_username or 'None'}, bot_replies_found={len(context['bot_replies_in_thread'])}")
```


## Recent Fixes (v1.6)

### 1. Retweet Handling in Followed Users (Nov 28, 2025)
**Problem**: Bot was extracting and analyzing media from the original tweet when a followed user retweeted something, leading to irrelevant image descriptions.

**Fix**: Added retweet detection in `fetch_and_process_followed_users()` to skip all retweets entirely:
```python
# Skip retweets - we only reply to original content
refs = getattr(t, 'referenced_tweets', None) if hasattr(t, 'referenced_tweets') else (t.get('referenced_tweets') if isinstance(t, dict) else None)
if refs:
    is_retweet = False
    for ref in refs:
        if isinstance(ref, dict):
            rtype = ref.get('type')
        else:
            rtype = getattr(ref, 'type', None)
        if rtype == 'retweeted':
            is_retweet = True
            break
    if is_retweet:
        print(f"[Followed] Skipping retweet {t.id}")
        continue
```

**Status**: Bot now only replies to original content from followed users, not their retweets.

### 2. Quoted Tweet Expansion Breaking (Nov 24, 2025)
**Problem**: Commit c1d31d0 added `author_id` expansion to `collect_quoted()` to fetch usernames, but this caused quoted tweets to stop being fetched properly because the API wasn't returning user data in includes.

**Root Cause**: When fetching a tweet directly by ID, `expansions=["author_id"]` doesn't populate includes with users. The expansion is only needed when expanding references in a parent tweet.

**Fix**: Removed `author_id` expansion and `user_fields` from `collect_quoted()`:
```python
quoted_response = read_client.get_tweet(
    id=ref_tweet.id,
    tweet_fields=["text", "author_id", "created_at", "attachments", "entities"],
    expansions=["attachments.media_keys"],  # Removed author_id
    media_fields=["type", "url", "preview_image_url", "alt_text"]
    # Removed user_fields=["username"]
)
```

**Status**: Quoted tweets now work correctly. Username display for quoted tweets remains a TODO.

### 3. Prompt Improvements
- Fixed typos in combine_msg: "muiltiple" → "multiple", "consise" → "concise"
- Added instruction to reduce repetition: "Do not repeat descriptions of the same images, links, or content multiple times—describe each once and move on."

### 4. Community Notes Dict/Object Handling Bugs (Dec 21, 2025)
**Problem**: Community Notes API returns data as dicts, not Tweepy objects, causing `AttributeError: 'dict' object has no attribute 'type'` crashes.

**Root Cause**: Three functions assumed `referenced_tweets` entries were Tweepy objects with `.type` and `.id` attributes, but Community Notes API returns plain dicts.

**Locations Found**:
1. `collect_quoted()` (~line 2964): `if ref_tweet.type == "quoted"`
2. `get_tweet_context()` (~line 3401): `if ref.type == 'replied_to'`
3. `get_tweet_context()` (~line 3187): Already had proper handling with `get_attr()`

**Fix Applied**: Added dict/object handling pattern:
```python
# Safe pattern for referenced_tweets iteration:
for ref in refs:
    ref_type = ref.get('type') if isinstance(ref, dict) else (ref.type if hasattr(ref, 'type') else None)
    ref_id = ref.get('id') if isinstance(ref, dict) else (ref.id if hasattr(ref, 'id') else None)
    
    if ref_type == 'replied_to':
        parent_id = ref_id
```

**Also Fixed**: Duplicate logging in Community Notes dry run mode (lines 4241-4244 had duplicate `log_to_file()` calls).

**Status**: All Community Notes dict/object handling now robust. Bot can process CN eligible posts without crashes.

**Critical Reminder**: ALWAYS check if objects from APIs are dicts or objects before accessing attributes. Use the safe pattern above or the `get_attr()` helper function.

### 5. Network Resilience and Retry Logic (Dec 21, 2025)
**Problem**: Bot crashed when internet went down during authentication with `socket.gaierror: [Errno -2] Name or service not known - Failed to resolve 'api.twitter.com'`.

**Root Cause**: No retry logic for network failures. Any DNS or connection issue during authentication or API calls caused immediate crash.

**Fix Applied**: Added `retry_with_backoff()` function and comprehensive network error handling:

1. **New retry function** (`retry_with_backoff()`):
   - Retries network operations with exponential backoff (5s → 10s → 20s → 40s → 80s... up to 300s max)
   - Handles: `socket.gaierror`, `NameResolutionError`, `MaxRetryError`, `ConnectionError`, `OSError`
   - Non-network errors raise immediately (no pointless retries)
   - Default: 5 retries, configurable max_retries and delays

2. **Authentication protection**:
   - `post_client.get_me()` wrapped with retry_with_backoff()
   - Main loop `authenticate()` called with `retry_with_backoff(authenticate, max_retries=10, initial_delay=10)`
   - Can survive ~85 minute outage during auth before giving up
   - On total auth failure: 60 second wait before full restart

3. **Main loop error handling improvements**:
   - Network errors (`socket.gaierror`, `ConnectionError`, `OSError`) separated from other exceptions
   - Network errors trigger 60 second restart delay (allow time for network recovery)
   - Other errors keep 10 second restart delay
   - Clear logging shows retry progress and error types

**Retry Timeline Example**:
```
Attempt 1: Immediate
Attempt 2: +10s (total: 10s)
Attempt 3: +20s (total: 30s)
Attempt 4: +40s (total: 70s)
Attempt 5: +80s (total: 150s)
...continues up to 10 attempts for auth
```

**Status**: Bot now survives temporary internet outages and automatically recovers when network comes back online.

### 6. Combining Model Retry Logic (Dec 21, 2025)
**Problem**: If the combining model (final pass) threw an error, the bot would tweet out the error message instead of retrying with a different model.

**Fix Applied**: Added retry logic in `fact_check()`:
- If combining model fails (verdict starts with "Error:"), automatically retry with different higher-tier model
- Clear attribution shows retry: `Combined by: gpt-5.2 (failed), retried with claude-sonnet-4-5`
- If retry also fails, uses original error response (graceful fallback)

**Status**: Bot no longer tweets error messages from model failures; automatically tries alternative models.

### 7. Community Notes Score Threshold Calibration (Jan 11-12, 2026)
**Problem**: Initial score thresholds for predicting Twitter's ClaimOpinion buckets (High/Medium/Low) were set arbitrarily, then incorrectly recalibrated based on desired distribution percentages rather than Twitter's actual API bucket assignments.

**Critical Learning**: Score thresholds must match Twitter's actual bucket assignments from end-of-run verification reports, NOT be chosen to achieve a desired distribution (e.g., 30/40/30).

**Historical Context**: Early High scores (0.7+) were artificially inflated because the bot rejected submissions below 0.7. Once restrictions were removed, true bucket boundaries began revealing themselves.

**Calibration Process**:
1. Extract all post IDs with their Twitter-assigned buckets from verification reports in `cn_notes_log_consenseai.txt` and `cn_notes_log_consenseai.txt.01`
2. Match post IDs to scores in `cn_written_consenseai.json`
3. Find empirical boundaries: max score of each bucket and min score of next bucket
4. Set thresholds at midpoint between boundaries

**Current Thresholds** (as of Jan 12, 2026, based on 38 verified notes):
- **High >= 0.081** (midpoint between max Medium -0.037 and min High 0.200)
- **Medium >= -2.109** (midpoint between max Low -2.468 and min Medium -1.750)
- Distribution: 17 High (44.7%), 13 Medium (34.2%), 8 Low (21.1%)

**Implementation Locations**:
- `get_score_distribution()` (lines ~347-378): Calculates current distribution of recent scores
- `should_reject_score()` (lines ~381-419): Decides whether to reject a note based on predicted bucket

**Data Sources**:
- `cn_notes_log_consenseai.txt`: Current verification reports
- `cn_notes_log_consenseai.txt.01`: Backup log with historical verifications
- `cn_written_consenseai.json`: Stores note text, timestamps, conversation IDs, and **scores**
- `cn_score_history_consenseai.json`: Rolling history of last 50 scores for distribution tracking

**Expected Evolution**: As more notes are submitted without artificial score floors, the min High score threshold may continue to decrease. Thresholds should be recalibrated periodically using latest verification data.

**Commands for Recalibration**:
```bash
# Extract verification data from logs
grep -B 1 "ClaimOpinion:" cn_notes_log_consenseai.txt.01 | grep -E "Post ID:|ClaimOpinion:"

# Match scores to buckets in Python
python3 << 'EOF'
import json
with open('cn_written_consenseai.json') as f: notes = json.load(f)
# ... map post_ids to buckets, extract scores, find boundaries
EOF
```

**Status**: Thresholds calibrated based on empirical Twitter API data. Will self-adjust as more verification data accumulates.

### 8. Community Notes Production Authorization (Jan 12, 2026)
**Status**: Bot authorized to submit Community Notes in production (test_mode=False).

**Implementation Notes**:
- Verification reports show ClaimOpinion, UrlValidity, and HarassmentAbuse scores
- **Critical**: ClaimOpinion bucket assignments are only reliable in test_mode; production buckets may differ
- Score-based rejection (`should_reject_score()`) helps maintain distribution requirements
- Production notes are publicly visible and contribute to Twitter's Community Notes ecosystem

## What NOT to Assume
- `README.md` contains historical notes and older filenames; do not treat it as authoritative for current runtime behavior
- The bot enforces character/format constraints via prompts — changing prompt wording can change runtime behavior
- Search processing may be skipped frequently due to dynamic caps; check logs to confirm runs
- twitterapi.io may fail intermittently; the bot will fall back to standard Tweepy text
- **Community Notes score thresholds are dynamic**: Recalibrate periodically as new verification data arrives
- **ClaimOpinion buckets may differ between test and production**: Use test mode for calibration data

## Future Improvements (Optional)
- Add username extraction for quoted tweets (currently only works for main tweets in thread)
- Add instrumentation logs showing which source provided mention_full_text (twitterapi.io vs Tweepy)
- Run dry-run end-to-end tests for retweet handling and ancestor chain building
- Consider inverting full-text resolution fallback order (prefer includes/official API before external provider)
- Add unit tests with mocked LLM clients and Tweepy responses
- Implement CI job for linting and dry-run smoke tests