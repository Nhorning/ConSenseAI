# ConSenseAI Development Guide

ConSenseAI is an X/Twitter bot that provides AI-powered fact-checking by leveraging multiple LLM models (Grok, GPT, Claude) to analyze and verify claims.

## Project Structure

**Current implementation:** `ConSenseAI_v1.6.py` (single-file bot, ~3813 lines)

**Current branch:** `main`

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
- `followed_reply_count_{username}.json` — Daily reply counts for followed users (separate cap)
- `followed_rotation_{username}.json` — Rotation state for followed users checking
- `sent_reply_hashes_{username}.json` — Reply deduplication hashes
- `approval_queue_{username}.json` — Human approval queue (when enabled)
- `cn_last_check_{username}.txt` — Last Community Notes check timestamp
- `cn_written_{username}.json` — Record of Community Notes written (for deduplication)
- `cn_notes_log_{username}.txt` — Detailed log of all Community Notes processing (flagged posts, generated notes, submission results)
- `output.log` — Rotating log file (rotates at LOG_MAX_SIZE_MB=10MB, keeps LOG_MAX_ROTATIONS=5 files)

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
   - **Configurable**: `--check_community_notes`, `--cn_max_results`, `--cn_test_mode` flags

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
python ConSenseAI_v1.4.py \
  --username consenseai \
  --delay 3 \
  --dryrun False \
  --search_term "auto" \
  --search_daily_cap 14 \
  --search_cap_interval_hours 1 \
  --cap_increase_time 6:00 \
  --reply_threshold 10 \
  --per_user_threshold True \
  --check_followed_users True \
  --followed_users_daily_cap 10 \
  --followed_users_per_cycle 3 \
  --post_interval 10
```

### Command Line Arguments
- `--username`: X/Twitter username to monitor (default: "ConSenseAI")
- `--delay`: Minutes between API checks (default: prompt)
- `--dryrun`: Print responses without posting (default: False)
- `--fetchthread`: Fetch full conversation context (default: True)
- `--reply_threshold`: Max replies per thread or per-user-per-thread (default: 5)
- `--per_user_threshold`: If True, enforce threshold per user; if False, per thread (default: True)
- `--search_term`: Keyword to search for proactive responses, or "auto" for AI-generated terms (default: None)
- `--search_max_results`: Max results per search query (default: 10)
- `--search_daily_cap`: Max automated search replies per day (default: 5)
- `--search_cap_interval_hours`: Hours between cap increases (default: 2)
- `--cap_increase_time`: Earliest time for cap increases, HH:MM format (default: "10:00")
- `--dedupe_window_hours`: Deduplication window in hours (default: 24.0)
- `--enable_human_approval`: Queue replies for approval instead of auto-posting (default: False)
- `--post_interval`: Number of replies between reflection posts (default: 10)
- `--check_followed_users`: Enable checking tweets from followed users (default: False)
- `--followed_users_daily_cap`: Max replies to followed users per day (default: 10)
- `--followed_users_per_cycle`: Number of followed users to check per cycle (rotation) (default: 3)
- `--followed_users_max_tweets`: Max tweets to fetch per followed user (default: 5)
- `--check_community_notes`: Enable Community Notes eligible posts checking (default: False)
- `--cn_max_results`: Max Community Notes posts to fetch per cycle (default: 5)
- `--cn_test_mode`: Submit notes in test mode (visible only to you) (default: True)

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
   - Call `fetch_and_process_followed_users()` (if `--check_followed_users` enabled) — check followed users' tweets
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

### 4. Auto-Follow System Not Working (FIXED)
**Symptoms:** Users with 5+ direct replies to bot are not being auto-followed; `get_user_reply_counts()` returns 0 for all users.

**Root cause:** `get_user_reply_counts()` only checked `ancestor_chains.json` for user replies, but old entries had `author_id: None` or missing fields. The function didn't use the same fallback logic as `count_bot_replies_by_user_in_conversation()`.

**Fix applied:** Enhanced `get_user_reply_counts()` to combine data from both `ancestor_chains.json` AND `bot_tweets.json`:
- First pass: Build set of bot reply tweet IDs using two methods:
  - Method 1: `author_id == bot_id` (explicit field match)
  - Method 2: `tweet_id in bot_tweets.json` (fallback for old entries)
- Second pass: Count user tweets where `in_reply_to_user_id == bot_id` AND `tweet_id NOT in bot_reply_ids`
- This mirrors the logic in `count_bot_replies_by_user_in_conversation()` but works across all conversations

**Status:** Fixed on fix-autofollow branch without modifying ancestor chain caching logic.

### 5. Infinite Loop on Deep Conversation Threads (CRITICAL BUG)
**Symptoms:** Bot hit rate limit 28 times in one day on a single deep conversation thread. Logs show same tweet IDs being fetched repeatedly. ancestor_chains.json was corrupted/wiped to 2 bytes at 18:31.

**Root cause:** In `get_tweet_context()` lines ~2615-2654, the ancestor chain building loop has exception handling that catches `tweepy.TweepyException`, prints error message, but **continues the loop** instead of breaking:
```python
while True:
    try:
        parent_tweet = read_client.get_tweet(...)
        # ... process parent ...
    except tweepy.TweepyException as e:
        print(f"Error building ancestor chain: {e}")
        # BUG: No break statement! Loop continues infinitely
```

**Trigger:** Deep conversation threads (50+ reply levels) on politically charged topics (Turkey/Greece geopolitics in observed case).

**Impact:**
- Rate limit exhausted (Twitter API ~900 requests/15min)
- Bot sleeps 899 seconds, then immediately hits same tweets again
- Potential file corruption (ancestor_chains.json reduced to 2 bytes: `{}`)
- Service downtime

**Current status:** Bug still present in code on fix-autofollow branch. Emergency rollback avoided triggering it, but underlying issue not fixed.

**Recommended fix:**
```python
MAX_ANCESTOR_DEPTH = 20  # Add depth limit constant

while True:
    if len(ancestor_chain) >= MAX_ANCESTOR_DEPTH:
        print(f"[Ancestor Chain] Hit depth limit ({MAX_ANCESTOR_DEPTH}), stopping build")
        break
    
    try:
        parent_tweet = read_client.get_tweet(...)
        # ... process parent ...
    except tweepy.TweepyException as e:
        print(f"Error building ancestor chain: {e}")
        break  # CRITICAL: Break loop on API errors
```

### 6. Duplicate Reply Prevention Not Working (KNOWN ISSUE)
**Symptoms:** Bot replies to same user multiple times in same conversation, hours apart.

**Root cause:** `count_bot_replies_by_user_in_conversation()` shows "FINAL COUNT: 0" despite prior replies existing. Two issues:
- Old entries in ancestor_chains.json have `author_id: None`, so counting logic fails to identify them
- Cached `bot_replies` list is empty (0 entries) even when bot has replied before

**Attempted fix (REVERTED):** Added fallback to check if tweet_id exists in bot_tweets.json. This was part of the commit that caused infinite loop bug, so reverted.

**Current status:** Duplicate prevention is broken on fix-autofollow branch.

**Alternative approaches:**
- Separate tracking file: `conversation_replies_{username}.json` with format `{conversation_id: {user_id: reply_count}}`
- Update after each successful reply, independent of ancestor_chains.json parsing
- Simple counter increment, no complex chain traversal required

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
python ConSenseAI_v1.4.py --dryrun True --delay 1 --search_term "test"
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

## What NOT to Assume
- `README.md` contains historical notes and older filenames; do not treat it as authoritative for current runtime behavior
- The bot enforces character/format constraints via prompts — changing prompt wording can change runtime behavior
- Search processing may be skipped frequently due to dynamic caps; check logs to confirm runs
- twitterapi.io may fail intermittently; the bot will fall back to standard Tweepy text

## Future Improvements (Optional)
- Add username extraction for quoted tweets (currently only works for main tweets in thread)
- Add instrumentation logs showing which source provided mention_full_text (twitterapi.io vs Tweepy)
- Run dry-run end-to-end tests for retweet handling and ancestor chain building
- Consider inverting full-text resolution fallback order (prefer includes/official API before external provider)
- Add unit tests with mocked LLM clients and Tweepy responses
- Implement CI job for linting and dry-run smoke tests