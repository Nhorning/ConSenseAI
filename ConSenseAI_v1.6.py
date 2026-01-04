import requests

# Fetch the full text of a tweet using twitterapi.io
def get_full_text_twitterapiio(tweet_id, api_key):
    """
    Fetch the full text of a tweet using twitterapi.io API.
    Args:
        tweet_id (str or int): The tweet ID to fetch.
        api_key (str, optional): The API key for twitterapi.io. If None, loads from keys.txt.
    Returns:
        str: The full text of the tweet, or empty string on error.
    """
    if api_key is None:
        print(f'attempting to fetch tweet {tweet_id} from twitterapi.io with key [None]')
    else:
        print(f'attempting to fetch tweet {tweet_id} from twitterapi.io with key {api_key[:10]}')
    url = "https://api.twitterapi.io/twitter/tweets"
    querystring = {"tweet_ids": f"{tweet_id}"}
    headers = {"X-API-Key": f"{api_key}"}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        # twitterapi.io returns tweets under 'tweets' as a list of dicts
        tweets = data.get("tweets", [])
        if tweets and isinstance(tweets, list):
            tweet_obj = tweets[0] if len(tweets) > 0 else None
            if tweet_obj and "text" in tweet_obj:
                return tweet_obj["text"]
        print(f"[twitterapi.io] No tweet text found for id {tweet_id}. Response: {data}")
        return ""
    except requests.RequestException as e:
        print(f"Error fetching tweet {tweet_id} from twitterapi.io: {e}")
        return ""

def get_tweet_data_twitterapiio(tweet_id, api_key):
    """
    Fetch full tweet data from twitterapi.io as a fallback when rate limited.
    Returns a dict that mimics Tweepy's tweet structure for ancestor chain building.
    """
    print(f'[twitterapi.io Fallback] Fetching tweet {tweet_id} from twitterapi.io')
    url = "https://api.twitterapi.io/twitter/tweets"
    querystring = {"tweet_ids": f"{tweet_id}"}
    headers = {"X-API-Key": f"{api_key}"}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()
        tweets = data.get("tweets", [])
        if tweets and isinstance(tweets, list) and len(tweets) > 0:
            tweet_data = tweets[0]
            
            # Convert referenced_tweets to objects with .type and .id attributes
            # Twitter API format: [{"type": "replied_to", "id": "123"}, ...]
            referenced_tweets = []
            for ref in tweet_data.get('referenced_tweets', []):
                if isinstance(ref, dict):
                    class RefObj:
                        def __init__(self, ref_dict):
                            self.type = ref_dict.get('type')
                            self.id = ref_dict.get('id')
                    referenced_tweets.append(RefObj(ref))
            
            # Create a simple dict that has the fields we need for ancestor chain building
            return {
                'id': tweet_data.get('id'),
                'text': tweet_data.get('text', ''),
                'author_id': tweet_data.get('author_id'),
                'created_at': tweet_data.get('created_at'),
                'conversation_id': tweet_data.get('conversation_id'),
                'in_reply_to_user_id': tweet_data.get('in_reply_to_user_id'),
                'referenced_tweets': referenced_tweets,
                'entities': tweet_data.get('entities'),
                'attachments': tweet_data.get('attachments')
            }
        print(f"[twitterapi.io Fallback] No tweet data found for id {tweet_id}")
        return None
    except requests.RequestException as e:
        print(f"[twitterapi.io Fallback] Error fetching tweet {tweet_id}: {e}")
        return None

import json
import os
import time
import datetime
import sys
import re
import webbrowser as web_open
import ast
import socket


def parse_bot_reply_entry(br):
    """
    Parse a bot_reply entry from cache, handling corrupted stringified dicts.
    Returns a dict if successful, None if parsing fails.
    
    Args:
        br: Entry from bot_replies list (can be dict, stringified dict, or tweet ID string)
    
    Returns:
        dict if parseable, None otherwise
    """
    if isinstance(br, dict):
        return br
    elif isinstance(br, str):
        # Try to parse if it's a stringified dict
        if br.startswith('{') or br.startswith("{'"):
            try:
                return ast.literal_eval(br)
            except:
                return None
        else:
            # Simple tweet ID string - can't extract metadata
            return None
    else:
        # Unknown type
        return None


def _search_last_filename(search_term: str):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', search_term)[:64]
    return SEARCH_LAST_FILE_PREFIX + safe + '.txt'


# Core constants (restored)
KEY_FILE = 'keys.txt'
LOG_MAX_SIZE_MB = 10  # Rotate log if larger than this (MB)
LOG_MAX_ROTATIONS = 5  # Keep this many rotated log files
MAX_BOT_TWEETS = 1000  # Max entries in bot_tweets.json
MAX_ANCESTOR_CHAINS = 500  # Max conversations in ancestor_chains.json
TWEETS_FILE = 'bot_tweets.json'  # File to store bot's tweets
ANCESTOR_CHAIN_FILE = 'ancestor_chains.json'


class LoggerWriter:
    def __init__(self, filename):
        self.filename = filename
        try:
            self.file = open(filename, 'a')
        except Exception:
            self.file = None

    def _rotate_log(self):
        if os.path.exists(self.filename):
            try:
                size_mb = os.path.getsize(self.filename) / (1024 * 1024)
                if size_mb > LOG_MAX_SIZE_MB:
                    for i in range(LOG_MAX_ROTATIONS - 1, 0, -1):
                        old_name = f"{self.filename}.{i}"
                        new_name = f"{self.filename}.{i+1}"
                        if os.path.exists(old_name):
                            os.rename(old_name, new_name)
                    if os.path.exists(self.filename):
                        os.rename(self.filename, f"{self.filename}.1")
            except Exception:
                pass

    def write(self, text):
        try:
            self._rotate_log()
            if self.file:
                self.file.write(text)
            sys.__stdout__.write(text)
        except Exception:
            pass

    def flush(self):
        try:
            if self.file:
                self.file.flush()
            sys.__stdout__.flush()
        except Exception:
            pass


# Redirect stdout/stderr to log file if possible
sys.stdout = LoggerWriter('output.log')
sys.stderr = LoggerWriter('output.log')

def read_last_search_id(search_term: str):
    """Read the last processed tweet ID for a search term from JSON storage.
    Falls back to old text file format for backward compatibility."""
    # First try new JSON format
    data = _load_json_file(_get_auto_search_file(), {})
    last_ids = data.get('last_search_ids', {})
    if search_term in last_ids:
        return int(last_ids[search_term])
    
    # Fall back to old text file format for backward compatibility
    fn = _search_last_filename(search_term)
    if os.path.exists(fn):
        try:
            with open(fn, 'r') as f:
                v = f.read().strip()
                if v:
                    tweet_id = int(v)
                    # Migrate to new format
                    write_last_search_id(search_term, tweet_id)
                    return tweet_id
        except Exception:
            pass
    return None

def write_last_search_id(search_term: str, tweet_id):
    """Write the last processed tweet ID for a search term to JSON storage.
    If tweet_id is None, removes the entry to reset the search."""
    data = _load_json_file(_get_auto_search_file(), {})
    if 'last_search_ids' not in data:
        data['last_search_ids'] = {}
    if tweet_id is None:
        # Remove the entry to reset
        if search_term in data['last_search_ids']:
            del data['last_search_ids'][search_term]
            print(f"[Search ID] Removed stale since_id for '{search_term}'")
    else:
        data['last_search_ids'][search_term] = str(tweet_id)
    _save_json_file(_get_auto_search_file(), data)

def _load_json_file(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return default

def _save_json_file(path, data):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving {path}: {e}")

def _increment_daily_count():
    data = _load_json_file(SEARCH_REPLY_COUNT_FILE, {})
    today = datetime.date.today().isoformat()
    data.setdefault(today, 0)
    data[today] += 1
    _save_json_file(SEARCH_REPLY_COUNT_FILE, data)


def _get_today_count():
    data = _load_json_file(SEARCH_REPLY_COUNT_FILE, {})
    today = datetime.date.today().isoformat()
    return int(data.get(today, 0))

# Auto search terms persistence
def _get_auto_search_file():
    """Get filename for storing auto-generated search terms and last search IDs."""
    return f'auto_search_terms_{username}.json'

def load_auto_search_state():
    """Load current search term, history of used terms, and last search IDs.
    
    Returns:
        tuple: (current_term, used_terms_list, last_search_ids_dict)
    """
    data = _load_json_file(_get_auto_search_file(), {})
    current = data.get('current_term', None)
    used = data.get('used_terms', [])
    last_ids = data.get('last_search_ids', {})
    if current and used:
        print(f"[Auto Search] Loaded persistent state: current='{current}', {len(used)} used terms, {len(last_ids)} search IDs")
    return current, used, last_ids

def save_auto_search_state(current_term, used_terms, last_search_ids=None):
    """Save current search term, history, and last search IDs.
    
    Args:
        current_term: The currently active search term
        used_terms: List of all search terms used (including current)
        last_search_ids: Dict mapping search terms to their last processed tweet IDs (optional)
    """
    # Load existing data to preserve last_search_ids if not provided
    existing_data = _load_json_file(_get_auto_search_file(), {})
    
    data = {
        'current_term': current_term,
        'used_terms': used_terms,
        'last_search_ids': last_search_ids if last_search_ids is not None else existing_data.get('last_search_ids', {}),
        'last_updated': datetime.datetime.now().isoformat()
    }
    _save_json_file(_get_auto_search_file(), data)
    print(f"[Auto Search] Saved state: current='{current_term}', {len(used_terms)} total used terms")

# Compute current search cap based on interval
def get_current_search_cap(max_daily_cap, interval_hours, cap_increase_time='10:00'):
    now = datetime.datetime.now()
    try:
        cap_hour, cap_minute = map(int, cap_increase_time.split(':'))
    except Exception:
        cap_hour, cap_minute = 10, 0
    start_minutes = cap_hour * 60 + cap_minute
    now_minutes = now.hour * 60 + now.minute
    if now_minutes < start_minutes:
        increments = 0
    else:
        increments = int(((now_minutes - start_minutes) / 60) // interval_hours)
    cap = min(1 + increments, max_daily_cap)
    return cap

def _add_sent_hash(reply_text: str):
    # store hash keyed by timestamp
    data = _load_json_file(SENT_HASHES_FILE, {})
    h = str(abs(hash(reply_text)))
    data[h] = int(time.time())
    # prune old entries beyond dedupe window
    window = int(float(args.dedupe_window_hours) * 3600)
    cutoff = int(time.time()) - window
    for k, v in list(data.items()):
        if v < cutoff:
            del data[k]
    _save_json_file(SENT_HASHES_FILE, data)

def _is_duplicate(reply_text: str):
    data = _load_json_file(SENT_HASHES_FILE, {})
    h = str(abs(hash(reply_text)))
    return h in data

def _has_replied_to_conversation_via_search(conversation_id: str, bot_user_id):
    """
    Check if bot has already replied to this conversation by examining ancestor_chains.json.
    Returns True if the bot has any replies in this conversation thread.
    """
    chains = load_ancestor_chains()
    conv_id_str = str(conversation_id)
    
    if conv_id_str not in chains:
        # No cached data for this conversation
        return False
    
    cache_entry = chains[conv_id_str]
    
    # Check if there are any bot replies recorded
    bot_replies = cache_entry.get('bot_replies', []) if isinstance(cache_entry, dict) else []
    if bot_replies:
        print(f"[Search Dedupe] Conversation {conv_id_str[:12]}.. already has {len(bot_replies)} bot replies")
        return True
    
    # Also check the chain itself for bot tweets
    chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
    bot_tweets = load_bot_tweets()
    
    for entry in chain:
        if not isinstance(entry, dict):
            continue
        tweet = entry.get('tweet', {})
        if not isinstance(tweet, dict):
            continue
        
        tweet_id = str(tweet.get('id', ''))
        author_id = str(tweet.get('author_id', ''))
        
        # Check if this is a bot tweet
        if author_id == str(bot_user_id) or tweet_id in bot_tweets:
            print(f"[Search Dedupe] Conversation {conv_id_str[:12]}.. contains bot tweet {tweet_id}")
            return True
    
    return False

def queue_for_approval(item: dict):
    queue = _load_json_file(APPROVAL_QUEUE_FILE, [])
    queue.append(item)
    _save_json_file(APPROVAL_QUEUE_FILE, queue)

def has_bot_replied_to_specific_tweet_id(target_tweet_id):
    """Check if bot has already replied directly to this specific tweet ID.
    
    This is more strict than conversation-level checks - it verifies the bot 
    hasn't already replied to THIS EXACT TWEET before.
    """
    if not target_tweet_id:
        return False
    
    target_id_str = str(target_tweet_id)
    bot_id_str = str(BOT_USER_ID)
    
    # Load ancestor chains to check for bot replies to this specific tweet
    chains = load_ancestor_chains()
    bot_tweets = load_bot_tweets()
    
    for conv_id, cache_entry in chains.items():
        if not isinstance(cache_entry, dict):
            continue
        
        # Check the ancestor chain for bot tweets that are direct replies to our target
        chain = cache_entry.get('chain', [])
        for entry in chain:
            if not isinstance(entry, dict):
                continue
            tweet = entry.get('tweet', {})
            
            # Handle both dict and string representations
            if isinstance(tweet, str):
                # Try to parse string representation
                try:
                    import ast
                    tweet = ast.literal_eval(tweet)
                except:
                    continue
            
            if not isinstance(tweet, dict):
                continue
            
            tweet_id = str(tweet.get('id', ''))
            author_id = str(tweet.get('author_id', ''))
            
            # Is this a bot tweet?
            if author_id == bot_id_str or tweet_id in bot_tweets:
                # Check if it's replying to our target tweet
                # Look for in_reply_to_status_id or check referenced_tweets
                in_reply_to = tweet.get('in_reply_to_status_id')
                if in_reply_to and str(in_reply_to) == target_id_str:
                    print(f"[Tweet-Level Dedupe] Bot already replied to tweet {target_id_str} with reply {tweet_id}")
                    return True
                
                # Also check referenced_tweets for reply relationship
                ref_tweets = tweet.get('referenced_tweets', [])
                if ref_tweets:
                    for ref in ref_tweets:
                        if isinstance(ref, dict) and ref.get('type') == 'replied_to':
                            if str(ref.get('id', '')) == target_id_str:
                                print(f"[Tweet-Level Dedupe] Bot already replied to tweet {target_id_str} with reply {tweet_id}")
                                return True
        
        # Also check bot_replies list
        bot_replies = cache_entry.get('bot_replies', [])
        for br in bot_replies:
            # Parse bot_reply entry (handles corrupted stringified dicts)
            br = parse_bot_reply_entry(br)
            if not br:
                continue
            
            if not isinstance(br, dict):
                continue
            
            # Check in_reply_to_status_id
            in_reply_to = br.get('in_reply_to_status_id')
            if in_reply_to and str(in_reply_to) == target_id_str:
                br_id = str(br.get('id', 'unknown'))
                print(f"[Tweet-Level Dedupe] Bot already replied to tweet {target_id_str} (found in bot_replies: {br_id})")
                return True
            
            # Also check referenced_tweets in bot_replies
            ref_tweets = br.get('referenced_tweets', [])
            if ref_tweets:
                for ref in ref_tweets:
                    if isinstance(ref, dict) and ref.get('type') == 'replied_to':
                        if str(ref.get('id', '')) == target_id_str:
                            br_id = str(br.get('id', 'unknown'))
                            print(f"[Tweet-Level Dedupe] Bot already replied to tweet {target_id_str} (found in bot_replies: {br_id})")
                            return True
    
    return False

def has_bot_replied_to_tweet(tweet_id):
    """Check if bot has already replied to a specific tweet by checking bot_tweets.json for reply target tracking."""
    if not tweet_id:
        return False
    
    tweet_id_str = str(tweet_id)
    
    # Load bot_tweets which are keyed by the bot's tweet ID
    bot_tweets = load_bot_tweets()
    
    # We need to check ancestor chains to find replies where in_reply_to matches our target
    chains = load_ancestor_chains()
    
    for conv_id, cache_entry in chains.items():
        if not isinstance(cache_entry, dict):
            continue
            
        # Check the ancestor chain for bot tweets
        chain = cache_entry.get('chain', [])
        for entry in chain:
            if not isinstance(entry, dict):
                continue
            tweet = entry.get('tweet', {})
            if not isinstance(tweet, dict):
                continue
                
            # Check if this is a bot tweet
            tweet_author = str(tweet.get('author_id', ''))
            tweet_id_from_entry = str(tweet.get('id', ''))
            
            # If this is a bot tweet (either by author or by being in bot_tweets)
            if tweet_author == str(BOT_USER_ID) or tweet_id_from_entry in bot_tweets:
                # Check in_reply_to_user_id to see if it's replying to our target conversation
                # Actually, we need to check if the tweet this bot tweet is replying to
                # is part of the target tweet's thread
                
                # Get the parent tweet ID from the referenced_tweets field if it exists
                # But since referenced_tweets isn't saved, let's use a different approach:
                # Check if this conversation contains our target tweet as the root
                if conv_id == tweet_id_str:
                    # The bot has replied in the conversation started by our target tweet
                    print(f"[Retweet Check] Bot already replied in conversation {tweet_id_str} (tweet {tweet_id_from_entry})")
                    return True
        
        # Also check bot_replies list
        bot_replies = cache_entry.get('bot_replies', [])
        for br in bot_replies:
            # Parse bot_reply entry (handles corrupted stringified dicts)
            br = parse_bot_reply_entry(br)
            if not br or not isinstance(br, dict):
                continue
            
            # If this bot reply is in a conversation rooted at our target tweet
            if conv_id == tweet_id_str:
                print(f"[Retweet Check] Bot already replied in conversation {tweet_id_str} (found in bot_replies)")
                return True
    
    return False


# Removed generate_reply_text_for_tweet to avoid duplicate model orchestration.
# fetch_and_process_search now calls fact_check(..., generate_only=True) to obtain generated reply text.

def fetch_and_process_search(search_term: str, user_id=None):
    """Search and run pipeline with safeguards: daily cap, dedupe, optional human approval."""
    global backoff_multiplier
    last_id = read_last_search_id(search_term)
    # Respect dynamic cap
    today_count = _get_today_count()
    start_time = args.cap_increase_time
    current_cap = get_current_search_cap(
        int(args.search_daily_cap),
        int(args.search_cap_interval_hours),
        start_time
    )
    print(f"[Search] Current dynamic search reply cap: {current_cap} (max: {args.search_daily_cap}, interval: {args.search_cap_interval_hours}h, Start Time: {start_time})")
    if today_count >= current_cap:
        print(f"[Search] Current cap reached ({today_count}/{current_cap}), skipping processing")
        return
    
    # Track users we've already replied to in this search cycle (limit 1 reply per user per cycle)
    replied_users_this_cycle = set()
    
    print(f"[Search] Query='{search_term}' since_id={last_id}")
    try:
        resp = read_client.search_recent_tweets(
            query=search_term,
            since_id=last_id,
            max_results=min(args.search_max_results, 100),
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "author_id", "referenced_tweets", "attachments", "entities"],
            expansions=["referenced_tweets.id", "attachments.media_keys", "author_id"],
            media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"],
            user_fields=["username"]
        )
    except tweepy.TweepyException as e:
        error_str = str(e)
        # Check if since_id is too old (outside 7-day window)
        if "since_id" in error_str and "must be a tweet id created after" in error_str:
            print(f"[Search] since_id {last_id} is too old (outside 7-day window), resetting to None")
            # Reset the since_id for this search term so it starts fresh
            write_last_search_id(search_term, None)
            # Retry without since_id
            try:
                resp = read_client.search_recent_tweets(
                    query=search_term,
                    max_results=min(args.search_max_results, 100),
                    tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "author_id", "referenced_tweets", "attachments", "entities"],
                    expansions=["referenced_tweets.id", "attachments.media_keys", "author_id"],
                    media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"],
                    user_fields=["username"]
                )
            except tweepy.TweepyException as retry_e:
                print(f"[Search] API error on retry: {retry_e}")
                backoff_multiplier += 1
                return
        else:
            print(f"[Search] API error: {e}")
            backoff_multiplier += 1
            return

    if not resp or not getattr(resp, 'data', None):
        print(f"[Search] No results for '{search_term}'")
        return

    

    bot_id = BOT_USER_ID

    # Process older -> newer
    for t in resp.data[::-1]:
        # In-loop cap check to prevent overruns
        today_count = _get_today_count()
        if today_count >= current_cap:
            print(f"[Search] Cap reached during processing ({today_count}/{current_cap}), stopping batch early.")
            break
        # Build full context including ancestor chain and thread, only once
        # Pass explicit bot username (global username expected to be set from args)
        context = get_tweet_context(t, resp.includes if hasattr(resp, 'includes') else None, bot_username=username if 'username' in globals() else None)
        context['mention'] = t
        context['context_instructions'] = "\nPrompt: appropriately respond to this tweet."
        
        # basic guard: don't reply to ourselves
        tweet_author = str(getattr(t, 'author_id', ''))
        if bot_id and tweet_author == str(bot_id):
            print(f"[Search] Skipping self tweet {t.id}")
            continue
        
        # Skip retweets - we only reply to original content
        if context.get('is_retweet'):
            print(f"[Search] Skipping retweet {t.id}")
            continue
        
        # CRITICAL: Check if we've already replied to this EXACT tweet ID before
        if has_bot_replied_to_specific_tweet_id(t.id):
            print(f"[Search] Skipping {t.id}: bot already replied to this specific tweet")
            continue
        
        # Check if we've already replied to this user in this search cycle
        if tweet_author in replied_users_this_cycle:
            print(f"[Search Cycle Limit] SKIPPING: Already replied to user {tweet_author} in this search cycle")
            continue
        
        # For retweets, also check if we've already replied to ANY retweet of the same original tweet in this cycle
        original_conv_id = context.get('original_conversation_id')
        if original_conv_id and original_conv_id in replied_users_this_cycle:
            print(f"[Search Cycle Limit] SKIPPING: Already replied to a retweet of original conversation {original_conv_id[:12]}.. in this cycle")
            continue
        
        # Check if bot has already replied to this conversation (conversation-level dedupe)
        # Check both the cache AND the context we just built
        conv_id = str(getattr(t, 'conversation_id', ''))
        if _has_replied_to_conversation_via_search(conv_id, bot_id):
            print(f"[Search] Skipping {t.id}: bot already replied to conversation {conv_id[:12]}..")
            continue
        
        # Also check the bot_replies in the context we just built (for conversations not yet cached)
        if context.get('bot_replies_in_thread'):
            print(f"[Search] Skipping {t.id}: bot already has {len(context['bot_replies_in_thread'])} replies in this conversation")
            continue
        
        # Check if this is a retweet and we've already replied to the original tweet's conversation
        reply_target = context.get('reply_target_id')
        original_conv_id = context.get('original_conversation_id')
        if reply_target and str(reply_target) != str(t.id):
            # This is a retweet (reply_target is different from the current tweet)
            if has_bot_replied_to_tweet(reply_target):
                print(f"[Search] Skipping retweet {t.id}: bot already replied to retweeter's conversation {reply_target}")
                continue
            # Also check if bot replied to the ORIGINAL tweet's conversation
            if original_conv_id and _has_replied_to_conversation_via_search(original_conv_id, bot_id):
                print(f"[Search] Skipping retweet {t.id}: bot already replied to original conversation {original_conv_id[:12]}..")
                continue
        
        # skip if bot already replied in conversation (per-user or per-thread threshold)
        if per_user_threshold:
            target_author = getattr(t, 'author_id', None)
            print(f"[Search Threshold] Checking per-user threshold for user {target_author} in conversation {conv_id}")
            prior_to_user = count_bot_replies_by_user_in_conversation(conv_id, bot_id, target_author, context.get('bot_replies_in_thread'))
            print(f"[Search Threshold] User {target_author}: {prior_to_user} replies / {reply_threshold} threshold")
            if prior_to_user >= reply_threshold:
                print(f"[Search Threshold] SKIPPING {t.id}: bot already replied to user {target_author} {prior_to_user} times (threshold={reply_threshold})")
                continue
            else:
                print(f"[Search Threshold] PROCEEDING with reply to user {target_author} ({prior_to_user} < {reply_threshold})")
        else:
            prior = count_bot_replies_in_conversation(conv_id, bot_id, context.get('bot_replies_in_thread'))
            if prior >= reply_threshold:
                print(f"[Search] Skipping {t.id}: bot already replied {prior} times in thread (threshold={reply_threshold})")
                continue
        # Run models and obtain a generated reply (fact_check can return generated text when generate_only=True)
        try:
            reply_text = fact_check(get_full_text(t), t.id, context=context, generate_only=True)
        except Exception as e:
            print(f"[Search] Error generating reply: {e}")
            continue

        if not isinstance(reply_text, str) or not reply_text:
            print(f"[Search] No reply generated for {t.id}; skipping")
            continue

        # simple content safety heuristics (on the generated reply)
        lowered = reply_text.lower()
        if any(x in lowered for x in ["doxx", "address", "phone", "ssn", "private"]):
            print(f"[Search] Reply contains potential sensitive content; queuing for review")
            queue_for_approval({"tweet_id": t.id, "text": reply_text, "reason": "sensitive"})
            continue

        # dedupe check
        if _is_duplicate(reply_text):
            print(f"[Search] Duplicate reply detected; skipping")
            continue

        # daily cap check again per candidate
        today_count = _get_today_count()
        if today_count >= int(args.search_daily_cap):
            print(f"[Search] Daily cap reached during processing; stopping")
            break

        # If human approval enabled, queue and continue
        if args.enable_human_approval:
            queue_for_approval({"tweet_id": t.id, "text": reply_text, "context_summary": get_full_text(t)[:300]})
            print(f"[Search] Queued reply for human approval: tweet {t.id}")
            continue

        # Post reply (respect dryrun)
        if args.dryrun:
            print(f"[Search dryrun] Would reply to {t.id}: {reply_text[:200]}")
        else:
            # Prefer context-provided reply target (e.g., retweeter's tweet id) when available
            try:
                reply_target = context.get('reply_target_id') if context and context.get('reply_target_id') else (getattr(t, 'id', None) if hasattr(t, 'id') else (t.get('id') if isinstance(t, dict) else None))
            except Exception:
                reply_target = (getattr(t, 'id', None) if hasattr(t, 'id') else (t.get('id') if isinstance(t, dict) else None))
            # post and track
            posted = post_reply(reply_target, reply_text, conversation_id=conv_id)
            if posted == 'done!':
                # Track that we replied to this user in this cycle
                replied_users_this_cycle.add(tweet_author)
                print(f"[Search Cycle Limit] Replied to user {tweet_author}, now in cycle tracking ({len(replied_users_this_cycle)} users total)")
                
                # Also track the original conversation ID if this is a retweet, to prevent multiple replies to different retweets of the same original
                original_conv_id = context.get('original_conversation_id')
                if original_conv_id:
                    replied_users_this_cycle.add(original_conv_id)
                    print(f"[Search Cycle Limit] Also tracking original conversation {original_conv_id[:12]}.. to prevent duplicate retweet replies")
                
                _add_sent_hash(reply_text)
                _increment_daily_count()
                write_last_search_id(search_term, t.id)
                # brief pause between posts
                time.sleep(5)
            elif posted == 'delay!':
                backoff_multiplier *= 2
                print(f"[Search] Post rate-limited, backing off")
                return


def _get_followed_today_count():
    """Get count of followed-user replies posted today"""
    data = {}
    if os.path.exists(FOLLOWED_USERS_REPLY_COUNT_FILE):
        try:
            with open(FOLLOWED_USERS_REPLY_COUNT_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    return data.get(today_str, 0)

def _increment_followed_daily_count():
    """Increment the followed-user reply count for today"""
    data = {}
    if os.path.exists(FOLLOWED_USERS_REPLY_COUNT_FILE):
        try:
            with open(FOLLOWED_USERS_REPLY_COUNT_FILE, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    data[today_str] = data.get(today_str, 0) + 1
    try:
        with open(FOLLOWED_USERS_REPLY_COUNT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"[Followed] Error saving reply count: {e}")

def _get_rotation_state():
    """Get rotation state for followed users (which user to check next)"""
    if os.path.exists(FOLLOWED_USERS_ROTATION_FILE):
        try:
            with open(FOLLOWED_USERS_ROTATION_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"last_index": 0, "last_checked": datetime.datetime.now().isoformat()}

def _save_rotation_state(state):
    """Save rotation state"""
    try:
        with open(FOLLOWED_USERS_ROTATION_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except IOError as e:
        print(f"[Followed] Error saving rotation state: {e}")

def fetch_and_process_followed_users():
    """
    Periodically check tweets from followed users and reply to them with rotation.
    Uses same pipeline as search with safeguards: daily cap, dedupe, threshold checks.
    Implements rotation so not all users are checked every cycle.
    """
    global backoff_multiplier
    
    # Check if feature is enabled
    if not getattr(args, 'check_followed_users', False):
        return
    
    # Track users we've already replied to in this cycle (limit 1 reply per user per cycle)
    replied_users_this_cycle = set()
    
    # Get list of followed users (sorted for consistent rotation)
    followed_users = sorted(list(get_followed_users()))
    if not followed_users:
        print("[Followed] No followed users found")
        return
    
    print(f"[Followed] Total followed users: {followed_users}")
    
    # Check separate daily cap for followed users (uses same dynamic cap logic as search)
    today_count = _get_followed_today_count()
    max_daily_cap = getattr(args, 'followed_users_daily_cap', 10)
    start_time = args.cap_increase_time
    interval_hours = int(args.search_cap_interval_hours)
    current_cap = get_current_search_cap(max_daily_cap, interval_hours, start_time)
    print(f"[Followed] Today's followed-user replies: {today_count}/{current_cap} (max: {max_daily_cap}, increases every {interval_hours}h starting at {start_time})")
    if today_count >= current_cap:
        print(f"[Followed] Current cap reached ({today_count}/{current_cap}), skipping")
        return
    daily_cap = current_cap  # Use current dynamic cap for subsequent checks
    
    # Load rotation state
    rotation_state = _get_rotation_state()
    last_index = rotation_state.get('last_index', 0)
    users_per_cycle = getattr(args, 'followed_users_per_cycle', 3)
    
    # Rotate through followed users
    total_users = len(followed_users)
    start_index = last_index % total_users
    end_index = (start_index + users_per_cycle) % total_users
    
    if end_index > start_index:
        users_to_check = followed_users[start_index:end_index]
    else:
        # Wrap around
        users_to_check = followed_users[start_index:] + followed_users[:end_index]
    
    print(f"[Followed] Checking {len(users_to_check)} of {total_users} followed users (rotation index: {start_index})")
    print(f"[Followed] Users in this cycle: {users_to_check}")
    
    # Load last checked tweet IDs per user
    last_checked = {}
    if os.path.exists(FOLLOWED_USERS_LAST_CHECK_FILE):
        try:
            with open(FOLLOWED_USERS_LAST_CHECK_FILE, 'r') as f:
                last_checked = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Followed] Error loading last checked file: {e}")
    
    bot_id = BOT_USER_ID
    max_tweets_per_user = getattr(args, 'followed_users_max_tweets', 5)
    
    # Check each followed user in rotation
    for user_id in users_to_check:
        # In-loop cap check
        today_count = _get_followed_today_count()
        if today_count >= daily_cap:
            print(f"[Followed] Cap reached during processing ({today_count}/{daily_cap}), stopping")
            break
        
        try:
            # Fetch recent tweets from this user
            since_id = last_checked.get(str(user_id))
            print(f"[Followed] Fetching tweets from user {user_id} (since_id={since_id})")
            
            resp = read_client.get_users_tweets(
                id=user_id,
                since_id=since_id,
                max_results=min(max_tweets_per_user, 100),
                tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "author_id", "referenced_tweets", "attachments", "entities"],
                expansions=["referenced_tweets.id", "attachments.media_keys", "author_id"],
                media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"],
                user_fields=["username"]
            )
            
            if not resp or not getattr(resp, 'data', None):
                print(f"[Followed] No new tweets from user {user_id}")
                continue
            
            print(f"[Followed] Found {len(resp.data)} tweets from user {user_id}")
            
            # Process tweets from oldest to newest
            for t in resp.data[::-1]:
                # In-loop cap check
                today_count = _get_followed_today_count()
                if today_count >= daily_cap:
                    print(f"[Followed] Cap reached during tweet processing ({today_count}/{daily_cap}), stopping")
                    break
                
                # Build full context
                context = get_tweet_context(t, resp.includes if hasattr(resp, 'includes') else None, bot_username=username if 'username' in globals() else None)
                context['mention'] = t
                context['context_instructions'] = (
                    "\nPrompt: Appropriately respond to this tweet from someone you follow."
                    "- IMPORTANT:Do NOT impersonate other users or answer on their behalf."
                    "- Do not forget that you are a bot answering questions with fact based analysis on X"
                    "- If you detect satire respond humorously"

                )
                
                # Don't reply to our own tweets
                tweet_author = str(getattr(t, 'author_id', ''))
                if bot_id and tweet_author == str(bot_id):
                    print(f"[Followed] Skipping self tweet {t.id}")
                    continue
                
                # Skip retweets - we only reply to original content
                if context.get('is_retweet'):
                    print(f"[Followed] Skipping retweet {t.id}")
                    continue
                
                # CRITICAL: Check if we've already replied to this EXACT tweet ID before
                if has_bot_replied_to_specific_tweet_id(t.id):
                    print(f"[Followed] Skipping {t.id}: bot already replied to this specific tweet")
                    continue
                
                # Check if we've already replied to this user in this cycle
                if tweet_author in replied_users_this_cycle:
                    print(f"[Followed Cycle Limit] SKIPPING: Already replied to user {tweet_author} in this cycle")
                    continue
                
                # For retweets, also check if we've already replied to ANY retweet of the same original tweet in this cycle
                original_conv_id = context.get('original_conversation_id')
                if original_conv_id and original_conv_id in replied_users_this_cycle:
                    print(f"[Followed Cycle Limit] SKIPPING: Already replied to a retweet of original conversation {original_conv_id[:12]}.. in this cycle")
                    continue
                
                # Check if bot already replied to this conversation
                conv_id = str(getattr(t, 'conversation_id', ''))
                if _has_replied_to_conversation_via_search(conv_id, bot_id):
                    print(f"[Followed] Skipping {t.id}: bot already replied to conversation {conv_id[:12]}..")
                    continue
                
                if context.get('bot_replies_in_thread'):
                    print(f"[Followed] Skipping {t.id}: bot already has {len(context['bot_replies_in_thread'])} replies in conversation")
                    continue
                
                # Check reply thresholds
                if per_user_threshold:
                    target_author = getattr(t, 'author_id', None)
                    prior_to_user = count_bot_replies_by_user_in_conversation(conv_id, bot_id, target_author, context.get('bot_replies_in_thread'))
                    if prior_to_user >= reply_threshold:
                        print(f"[Followed] Skipping {t.id}: already replied to user {target_author} {prior_to_user} times")
                        continue
                else:
                    prior = count_bot_replies_in_conversation(conv_id, bot_id, context.get('bot_replies_in_thread'))
                    if prior >= reply_threshold:
                        print(f"[Followed] Skipping {t.id}: already replied {prior} times in thread")
                        continue
                
                # Generate reply
                try:
                    reply_text = fact_check(get_full_text(t), t.id, context=context, generate_only=True)
                except Exception as e:
                    print(f"[Followed] Error generating reply: {e}")
                    continue
                
                if not isinstance(reply_text, str) or not reply_text:
                    print(f"[Followed] No reply generated for {t.id}; skipping")
                    continue
                
                # Content safety check
                lowered = reply_text.lower()
                if any(x in lowered for x in ["doxx", "address", "phone", "ssn", "private"]):
                    print(f"[Followed] Reply contains sensitive content; queuing for review")
                    queue_for_approval({"tweet_id": t.id, "text": reply_text, "reason": "sensitive"})
                    continue
                
                # Dedupe check
                if _is_duplicate(reply_text):
                    print(f"[Followed] Duplicate reply detected; skipping")
                    continue
                
                # Final cap check
                today_count = _get_followed_today_count()
                if today_count >= daily_cap:
                    print(f"[Followed] Cap reached before posting; stopping")
                    break
                
                # Human approval queue if enabled
                if args.enable_human_approval:
                    queue_for_approval({"tweet_id": t.id, "text": reply_text, "context_summary": get_full_text(t)[:300]})
                    print(f"[Followed] Queued reply for approval: tweet {t.id}")
                    continue
                
                # Post reply (respect dryrun)
                if args.dryrun:
                    print(f"[Followed dryrun] Would reply to {t.id}: {reply_text[:200]}")
                else:
                    reply_target = context.get('reply_target_id') if context and context.get('reply_target_id') else t.id
                    posted = post_reply(reply_target, reply_text, conversation_id=conv_id)
                    if posted == 'done!':
                        # Track that we replied to this user in this cycle
                        replied_users_this_cycle.add(tweet_author)
                        print(f"[Followed Cycle Limit] Replied to user {tweet_author}, now in cycle tracking ({len(replied_users_this_cycle)} users total)")
                        
                        # Also track the original conversation ID if this is a retweet
                        original_conv_id = context.get('original_conversation_id')
                        if original_conv_id:
                            replied_users_this_cycle.add(original_conv_id)
                            print(f"[Followed Cycle Limit] Also tracking original conversation {original_conv_id[:12]}.. to prevent duplicate retweet replies")
                        
                        _add_sent_hash(reply_text)
                        _increment_followed_daily_count()
                        # Update last checked ID for this user
                        last_checked[str(user_id)] = str(t.id)
                        try:
                            with open(FOLLOWED_USERS_LAST_CHECK_FILE, 'w') as f:
                                json.dump(last_checked, f, indent=2)
                            print(f"[Followed] Successfully replied to user {user_id}, tweet {t.id}")
                        except IOError as e:
                            print(f"[Followed] Error saving last checked file: {e}")
                        time.sleep(5)  # Brief pause between posts
                        # CRITICAL: Break inner loop after first successful reply to this user
                        break  # Move to next user after replying once
                    elif posted == 'delay!':
                        backoff_multiplier *= 2
                        print(f"[Followed] Post rate-limited, backing off")
                        return
        
        except tweepy.TweepyException as e:
            print(f"[Followed] API error for user {user_id}: {e}")
            backoff_multiplier += 1
            continue
        except Exception as e:
            print(f"[Followed] Unexpected error for user {user_id}: {e}")
            continue
    
    # Update rotation state for next cycle
    new_index = (start_index + users_per_cycle) % total_users
    _save_rotation_state({
        "last_index": new_index,
        "last_checked": datetime.datetime.now().isoformat()
    })
    print(f"[Followed] Rotation complete, next cycle will start at index {new_index}")


def load_bot_tweets(verbose=True):
    """Load stored bot tweets from JSON file"""
    if os.path.exists(TWEETS_FILE):
        try:
            with open(TWEETS_FILE, 'r') as f:
                tweets = json.load(f)
                if verbose:
                    print(f"[Tweet Storage] Loaded {len(tweets)} stored tweets from {TWEETS_FILE}")
                return tweets
        except json.JSONDecodeError:
            if verbose:
                print(f"[Tweet Storage] Error reading {TWEETS_FILE}, starting fresh")
    else:
        if verbose:
            print(f"[Tweet Storage] No existing {TWEETS_FILE} found, starting fresh")
    return {}

def load_ancestor_chains():
    """Load the ancestor chain cache if present and return as a dict.
    Keys are conversation_id (as strings) -> dict with 'chain': list of entries, and optionally other context like 'thread_tweets', 'bot_replies'.
    """
    if os.path.exists(ANCESTOR_CHAIN_FILE):
        try:
            with open(ANCESTOR_CHAIN_FILE, 'r') as f:
                chains = json.load(f)
                # ensure keys are strings
                return {str(k): v for k, v in chains.items()}
        except Exception as e:
            print(f"[Ancestor Cache] Error loading {ANCESTOR_CHAIN_FILE}: {e}")
    return {}

def save_ancestor_chain(conversation_id, chain, additional_context=None):
    chains = load_ancestor_chains()
    # Convert chain to serializable format
    serializable_chain = []
    for entry in chain:
        tweet_dict = tweet_to_dict(entry["tweet"])
        quoted_dicts = [tweet_to_dict(qt) for qt in entry["quoted_tweets"]]
        media = entry.get("media", [])  # Media should already be serializable
        username = entry.get("username")  # Get username if present
        
        chain_entry = {"tweet": tweet_dict, "quoted_tweets": quoted_dicts, "media": media}
        if username:
            chain_entry["username"] = username  # Preserve username in cache
        serializable_chain.append(chain_entry)
    cache_entry = {"chain": serializable_chain}
    if additional_context:
        cache_entry.update(additional_context)

    chains[str(conversation_id)] = cache_entry
    if len(chains) > MAX_ANCESTOR_CHAINS:
        # Remove oldest conversations (keep most recent)
        sorted_chains = sorted(chains.items(), key=lambda x: x[0], reverse=True)
        chains = dict(sorted_chains[:MAX_ANCESTOR_CHAINS])
    try:
        with open(ANCESTOR_CHAIN_FILE, 'w') as f:
            json.dump(chains, f, indent=2)
    except Exception as e:
        print(f"Error saving ancestor chain cache: {e}")

def count_bot_replies_in_conversation(conversation_id, bot_user_id, api_bot_replies=None):
    """Count prior bot replies for a conversation using existing JSON caches and optional API results.

    - conversation_id: the conversation id (may be int/str)
    - bot_user_id: the bot's numeric id (int or str)
    - api_bot_replies: optional list of Tweepy tweet objects from API (context['bot_replies_in_thread'])
    Returns integer count.
    """
    if not conversation_id:
        return 0
    cid = str(conversation_id)
    count = 0
    bot_tweets = load_bot_tweets()  # keys are tweet ids (strings)

    # Check ancestor chain cache
    chains = load_ancestor_chains()
    cached_data = chains.get(cid)
    if cached_data:
        # Handle both old format (direct list) and new format (dict with 'chain')
        chain = cached_data.get('chain', cached_data) if isinstance(cached_data, dict) else cached_data
        for entry in chain:
            # entry was saved as {'tweet': tweet_dict, 'quoted_tweets': [...], 'media': [...]}
            t = entry.get('tweet', {})
            tid = str(t.get('id')) if isinstance(t, dict) and t.get('id') is not None else None
            author = str(t.get('author_id')) if isinstance(t, dict) and t.get('author_id') is not None else None
            if tid and tid in bot_tweets:
                count += 1
            elif author and bot_user_id and str(author) == str(bot_user_id):
                count += 1

        # Check cached bot replies if available
        if isinstance(cached_data, dict) and 'bot_replies' in cached_data:
            for br in cached_data['bot_replies']:
                # Parse bot_reply entry (handles corrupted stringified dicts)
                br = parse_bot_reply_entry(br)
                if not br or not isinstance(br, dict):
                    continue
                
                br_id = str(br.get('id')) if br.get('id') is not None else None
                br_author = str(br.get('author_id')) if br.get('author_id') is not None else None
                if br_id and br_id in bot_tweets:
                    count += 1
                elif br_author and str(br_author) == str(bot_user_id):
                    count += 1

    # Fallback: check API-provided bot replies (if any)
    if api_bot_replies:
        for br in api_bot_replies:
            # br may be a Tweepy object or dict
            br_id = br.get('id') if isinstance(br, dict) else getattr(br, 'id', None)
            br_author = br.get('author_id') if isinstance(br, dict) else getattr(br, 'author_id', None)
            if br_id and str(br_id) in bot_tweets:
                count += 1
            elif br_author and str(br_author) == str(bot_user_id):
                count += 1

    return count


def count_bot_replies_by_user_in_conversation(conversation_id, bot_user_id, target_user_id, api_bot_replies=None):
    """Count how many times the bot has replied to a specific user inside a conversation.

    This inspects the ancestor_chain cache, any cached bot_replies, and optional API-provided
    bot replies. It counts bot-authored tweets whose in_reply_to_user_id (or referenced parent)
    equals the target_user_id.
    """
    if not conversation_id or not target_user_id:
        print(f"[Per-User Count] Missing conversation_id or target_user_id, returning 0")
        return 0
    cid = str(conversation_id)
    counted_tweet_ids = set()  # Track which tweet IDs we've already counted to avoid duplicates
    bot_tweets = load_bot_tweets()
    
    print(f"[Per-User Count] Checking replies to user {target_user_id} in conversation {cid}")

    # Check ancestor chain cache
    chains = load_ancestor_chains()
    cached_data = chains.get(cid)
    if cached_data:
        chain = cached_data.get('chain', cached_data) if isinstance(cached_data, dict) else cached_data
        print(f"[Per-User Count] Found cached chain with {len(chain)} entries")
        for entry in chain:
            t = entry.get('tweet', {})
            # t may be a dict saved by tweet_to_dict
            t_author = str(t.get('author_id')) if isinstance(t, dict) and t.get('author_id') is not None else None
            t_in_reply_to = str(t.get('in_reply_to_user_id')) if isinstance(t, dict) and t.get('in_reply_to_user_id') is not None else None
            tid = str(t.get('id')) if isinstance(t, dict) and t.get('id') is not None else None
            
            # Debug each entry
            if t_author and str(t_author) == str(bot_user_id):
                print(f"[Per-User Count] Found bot tweet {tid}, in_reply_to_user_id={t_in_reply_to}, target={target_user_id}")
            
            # Check if this is a bot tweet replying to target user
            # Method 1: Has explicit author_id matching bot
            is_bot_tweet_method1 = (t_author and bot_user_id and str(t_author) == str(bot_user_id))
            # Method 2: Tweet ID is in bot_tweets.json (works even if author_id is None)
            is_bot_tweet_method2 = (tid and tid in bot_tweets)
            
            if (is_bot_tweet_method1 or is_bot_tweet_method2) and t_in_reply_to and str(t_in_reply_to) == str(target_user_id):
                if tid and tid not in counted_tweet_ids:  # Only count if not already counted
                    counted_tweet_ids.add(tid)
                    print(f"[Per-User Count] Counted bot reply {tid} to user {target_user_id} (count now: {len(counted_tweet_ids)})")

        # Also inspect cached bot_replies list if present
        if isinstance(cached_data, dict) and 'bot_replies' in cached_data:
            print(f"[Per-User Count] Checking cached bot_replies list ({len(cached_data['bot_replies'])} entries)")
            for br in cached_data['bot_replies']:
                # Parse bot_reply entry (handles corrupted stringified dicts)
                br = parse_bot_reply_entry(br)
                if not br:
                    continue
                
                if isinstance(br, dict):
                    br_author = str(br.get('author_id')) if br.get('author_id') is not None else None
                    br_in_reply_to = str(br.get('in_reply_to_user_id')) if br.get('in_reply_to_user_id') is not None else None
                    br_id = str(br.get('id')) if br.get('id') is not None else None
                    if br_author and bot_user_id and str(br_author) == str(bot_user_id) and br_in_reply_to and str(br_in_reply_to) == str(target_user_id):
                        if br_id and br_id not in counted_tweet_ids and (br_id is None or br_id in bot_tweets):
                            counted_tweet_ids.add(br_id)
                            print(f"[Per-User Count] Counted bot reply {br_id} from cached bot_replies (count now: {len(counted_tweet_ids)})")
                else:
                    # Tweepy object - try getattr
                    br_author = getattr(br, 'author_id', None)
                    br_in_reply_to = getattr(br, 'in_reply_to_user_id', None)
                    br_id = str(getattr(br, 'id', None)) if getattr(br, 'id', None) is not None else None
                    if br_author and bot_user_id and str(br_author) == str(bot_user_id) and br_in_reply_to and str(br_in_reply_to) == str(target_user_id):
                        if br_id and br_id not in counted_tweet_ids and (br_id is None or br_id in bot_tweets):
                            counted_tweet_ids.add(br_id)
                            print(f"[Per-User Count] Counted bot reply {br_id} from cached bot_replies (count now: {len(counted_tweet_ids)})")
    else:
        print(f"[Per-User Count] No cached data found for conversation {cid}")

    # Check API-provided bot replies if any
    if api_bot_replies:
        print(f"[Per-User Count] Checking API-provided bot replies ({len(api_bot_replies)} entries)")
        for br in api_bot_replies:
            # br may be a Tweepy object or dict
            br_author = br.get('author_id') if isinstance(br, dict) else getattr(br, 'author_id', None)
            br_in_reply_to = None
            if isinstance(br, dict):
                br_in_reply_to = br.get('in_reply_to_user_id')
            else:
                br_in_reply_to = getattr(br, 'in_reply_to_user_id', None)
            br_id = br.get('id') if isinstance(br, dict) else getattr(br, 'id', None)
            br_id_str = str(br_id) if br_id is not None else None
            if br_author and bot_user_id and str(br_author) == str(bot_user_id) and br_in_reply_to and str(br_in_reply_to) == str(target_user_id):
                if br_id_str and br_id_str not in counted_tweet_ids and (br_id is None or br_id_str in bot_tweets):
                    counted_tweet_ids.add(br_id_str)
                    print(f"[Per-User Count] Counted bot reply {br_id_str} from API results (count now: {len(counted_tweet_ids)})")

    count = len(counted_tweet_ids)
    print(f"[Per-User Count] FINAL COUNT: {count} unique replies to user {target_user_id} in conversation {cid}")
    return count

def save_bot_tweet(tweet_id, full_content):
    """Save bot tweet content to JSON file"""
    tweets = load_bot_tweets()
    tweets[str(tweet_id)] = full_content
    if len(tweets) > MAX_BOT_TWEETS:
        # Remove oldest entries (keep most recent)
        sorted_tweets = sorted(tweets.items(), key=lambda x: x[0], reverse=True)
        tweets = dict(sorted_tweets[:MAX_BOT_TWEETS])
    try:
        with open(TWEETS_FILE, 'w') as f:
            json.dump(tweets, f, indent=2)
            print(f"[Tweet Storage] Successfully saved tweet {tweet_id} (content length: {len(full_content)})")
    except IOError as e:
        print(f"[Tweet Storage] Error saving tweet to {TWEETS_FILE}: {e}")

def get_bot_tweet_content(tweet_id, verbose=True):
    """Retrieve full content of a bot tweet if available"""
    tweets = load_bot_tweets(verbose=verbose)
    content = tweets.get(str(tweet_id))
    if verbose:
        if content:
            print(f"[Tweet Storage] Retrieved stored content for tweet {tweet_id} (length: {len(content)})")
        else:
            print(f"[Tweet Storage] No stored content found for tweet {tweet_id}")
    return content

def get_user_reply_counts():
    """
    Count how many times each user has @mentioned the bot.
    Only counts tweets that contain an @mention of the bot's username in the text.
    This includes both direct mentions and replies that tag the bot.
    Returns dict: {user_id: mention_count}
    
    Uses combined data from ancestor_chains.json and bot_tweets.json to ensure
    accurate counting even when ancestor_id fields are missing.
    """
    user_counts = {}
    
    # Load ancestor chains which contain all conversation data
    try:
        with open(ANCESTOR_CHAIN_FILE, 'r') as f:
            chains = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[Auto-Follow] No ancestor chains found")
        return user_counts
    
    # Load bot tweets for fallback identification
    bot_tweets = load_bot_tweets()
    bot_id = str(getid())
    
    # Get bot username for mention checking (global variable from main())
    bot_username = username.lower() if 'username' in globals() else 'consenseai'
    
    for conv_id, conv_data in chains.items():
        # Handle both dict format (new) and list format (legacy)
        if isinstance(conv_data, dict):
            chain = conv_data.get('chain', [])
            bot_replies = conv_data.get('bot_replies', [])
        else:
            chain = conv_data
            bot_replies = []
        
        # Track which tweets are bot replies using two methods:
        # Method 1: author_id matches bot_id
        # Method 2: tweet_id exists in bot_tweets.json
        bot_reply_ids = set()
        for entry in chain:
            if not entry or not isinstance(entry, dict):
                continue
            tweet = entry.get('tweet')
            if not tweet:
                continue
            
            # Handle both dict and string representations
            if isinstance(tweet, str):
                try:
                    import ast
                    tweet = ast.literal_eval(tweet)
                except:
                    continue
            
            tweet_id = str(tweet.get('id')) if isinstance(tweet, dict) and tweet.get('id') else None
            author_id = str(tweet.get('author_id')) if isinstance(tweet, dict) and tweet.get('author_id') else None
            
            # Mark as bot reply if either method confirms it
            if (author_id and author_id == bot_id) or (tweet_id and tweet_id in bot_tweets):
                if tweet_id:
                    bot_reply_ids.add(tweet_id)
        
        # Also check bot_replies list
        for br in bot_replies:
            # Parse bot_reply entry (handles corrupted stringified dicts)
            br = parse_bot_reply_entry(br)
            if not br:
                continue
            
            if not isinstance(br, dict):
                continue
                
            br_id = str(br.get('id')) if br.get('id') else None
            if br_id:
                bot_reply_ids.add(br_id)
        
        # Skip conversations where bot hasn't participated
        if not bot_reply_ids:
            continue
        
        # Now count user tweets that @mention the bot
        for entry in chain:
            if not entry or not isinstance(entry, dict):
                continue
            tweet = entry.get('tweet')
            if not tweet:
                continue
            
            # Handle both dict and string representations
            if isinstance(tweet, str):
                try:
                    import ast
                    tweet = ast.literal_eval(tweet)
                except:
                    continue
            
            tweet_id = str(tweet.get('id')) if isinstance(tweet, dict) and tweet.get('id') else None
            author_id = str(tweet.get('author_id')) if isinstance(tweet, dict) and tweet.get('author_id') else None
            tweet_text = str(tweet.get('text', '')) if isinstance(tweet, dict) else ''
            
            # Count only if:
            # 1. This is NOT a bot tweet
            # 2. Has valid author_id
            # 3. Author is not the bot
            # 4. Tweet text contains @mention of the bot
            if (tweet_id not in bot_reply_ids and 
                author_id and 
                author_id != bot_id and 
                f'@{bot_username}' in tweet_text.lower()):
                user_counts[author_id] = user_counts.get(author_id, 0) + 1
    
    print(f"[Auto-Follow] Counted mentions from {len(user_counts)} unique users across {len(chains)} conversations")
    for user_id, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"[Auto-Follow] User {user_id}: {count} replies")
    return user_counts

def get_followed_users():
    """
    Load list of users we've already followed. 
    Stored in bot_tweets.json under special key '_followed_users'
    """
    tweets = load_bot_tweets()
    followed = tweets.get('_followed_users', [])
    return set(followed)

def save_followed_user(user_id):
    """Add user_id to the list of followed users in bot_tweets.json"""
    tweets = load_bot_tweets()
    followed = tweets.get('_followed_users', [])
    if user_id not in followed:
        followed.append(user_id)
        tweets['_followed_users'] = followed
        try:
            with open(TWEETS_FILE, 'w') as f:
                json.dump(tweets, f, indent=2)
                print(f"[Auto-Follow] Saved user {user_id} to followed list")
        except IOError as e:
            print(f"[Auto-Follow] Error saving followed user: {e}")

def sync_followed_users_from_api():
    """
    Sync the cached followed users list with the actual following list.
    Tries X API first, falls back to twitterapi.io if that fails.
    This ensures the cache stays accurate even if users are unfollowed manually.
    """
    actual_following = set()
    
    # Try X API first (Basic tier)
    try:
        print("[Auto-Follow] Syncing followed users from X API...")
        pagination_token = None
        
        while True:
            try:
                if pagination_token:
                    response = read_client.get_users_following(id=getid(), pagination_token=pagination_token, max_results=100)
                else:
                    response = read_client.get_users_following(id=getid(), max_results=100)
                
                if response.data:
                    for user in response.data:
                        actual_following.add(str(user.id))
                
                # Check if there are more pages
                if hasattr(response, 'meta') and response.meta.get('next_token'):
                    pagination_token = response.meta['next_token']
                else:
                    break
            except tweepy.errors.TooManyRequests:
                print("[Auto-Follow] Rate limited by X API, will retry next cycle")
                return
            except (tweepy.errors.Forbidden, tweepy.errors.Unauthorized) as e:
                print(f"[Auto-Follow] X API not available: {e}")
                print("[Auto-Follow] Falling back to twitterapi.io...")
                actual_following = set()  # Clear any partial data
                break
            except Exception as e:
                print(f"[Auto-Follow] X API error: {e}")
                print("[Auto-Follow] Falling back to twitterapi.io...")
                actual_following = set()
                break
        
        # If we got data from X API, we're done
        if actual_following:
            tweets = load_bot_tweets()
            tweets['_followed_users'] = list(actual_following)
            try:
                with open(TWEETS_FILE, 'w') as f:
                    json.dump(tweets, f, indent=2)
                print(f"[Auto-Follow] Synced {len(actual_following)} followed users from X API")
            except IOError as e:
                print(f"[Auto-Follow] Error saving synced followers: {e}")
            return
            
    except Exception as e:
        print(f"[Auto-Follow] Unexpected error with X API: {e}")
        print("[Auto-Follow] Falling back to twitterapi.io...")
    
    # Fallback to twitterapi.io
    try:
        api_key = keys.get('TWITTERAPIIO_KEY')
        if not api_key:
            print("[Auto-Follow] No TWITTERAPIIO_KEY found, skipping sync")
            return
        
        print("[Auto-Follow] Using twitterapi.io to sync followed users...")
        
        try:
            url = "https://api.twitterapi.io/twitter/user/followings"
            
            headers = {
                'X-API-Key': api_key
            }
            querystring = {"pageSize":"200","userName":username}
            
            response = requests.get(url, headers=headers, params=querystring, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[Auto-Follow DEBUG] twitterapi.io response keys: {list(data.keys())}")
                
                # Extract user IDs from the response
                # Note: twitterapi.io uses 'followers' key even for the /user/followings endpoint
                users = data.get('followers', data.get('followings', data.get('users', data.get('data', []))))
                if isinstance(data, list):
                    users = data
                
                print(f"[Auto-Follow DEBUG] Found {len(users) if isinstance(users, list) else 0} users in response")
                    
                for user in users:
                    if isinstance(user, dict):
                        # The 'id' field contains the user ID as a string
                        user_id = user.get('id') or user.get('id_str')
                        if user_id:
                            actual_following.add(str(user_id))
                            print(f"[Auto-Follow DEBUG] Added user {user_id} (name: {user.get('userName', 'unknown')}) to following list")
                    elif isinstance(user, str):
                        actual_following.add(str(user))
                        print(f"[Auto-Follow DEBUG] Added user {user} to following list")
                
            elif response.status_code == 429:
                print("[Auto-Follow] Rate limited by twitterapi.io, will retry next cycle")
                return
            else:
                print(f"[Auto-Follow] twitterapi.io returned status {response.status_code}: {response.text[:200]}")
                return
                
        except requests.exceptions.RequestException as e:
            print(f"[Auto-Follow] Network error fetching following list: {e}")
            return
        except Exception as e:
            print(f"[Auto-Follow] Error parsing following list: {e}")
            return
        
        # Update the cache with the actual list
        tweets = load_bot_tweets()
        tweets['_followed_users'] = list(actual_following)
        
        try:
            with open(TWEETS_FILE, 'w') as f:
                json.dump(tweets, f, indent=2)
            print(f"[Auto-Follow] Synced {len(actual_following)} followed users from twitterapi.io")
        except IOError as e:
            print(f"[Auto-Follow] Error saving synced followers: {e}")
            
    except Exception as e:
        print(f"[Auto-Follow] Error syncing followed users: {e}")

def follow_user(user_id):
    """
    Follow a user via Twitter API.
    Returns True if successful, False otherwise.
    """
    try:
        # Use Tweepy v2 Client method
        post_client.follow_user(target_user_id=user_id)
        print(f"[Auto-Follow] Successfully followed user {user_id}")
        return True
    except tweepy.errors.Forbidden as e:
        # Already following or user has blocked us
        print(f"[Auto-Follow] Cannot follow user {user_id}: {e}")
        return False
    except tweepy.errors.TooManyRequests:
        print(f"[Auto-Follow] Rate limited on follow request for user {user_id}")
        return False
    except Exception as e:
        print(f"[Auto-Follow] Error following user {user_id}: {e}")
        return False

def check_and_follow_active_users(min_replies=3):
    """
    Check for users with min_replies or more interactions and follow them.
    Called during reflection cycle.
    """
    print(f"[Auto-Follow] Checking for users with {min_replies}+ replies to follow...")
    
    user_counts = get_user_reply_counts()
    already_followed = get_followed_users()
    
    # DEBUG: Print all user counts with details
    print(f"[Auto-Follow DEBUG] All user reply counts: {user_counts}")
    print(f"[Auto-Follow DEBUG] Already followed: {already_followed}")
    
    # Show each user's status
    for uid, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
        if uid in already_followed:
            status = "already followed"
        elif count >= min_replies:
            status = f"WILL FOLLOW (has {count} >= {min_replies})"
        else:
            status = f"not enough replies ({count} < {min_replies})"
        print(f"[Auto-Follow DEBUG] User {uid}: {count} replies - {status}")
    
    # Filter to users with enough replies who we haven't followed yet
    to_follow = {uid: count for uid, count in user_counts.items() 
                 if count >= min_replies and uid not in already_followed}
    
    if not to_follow:
        print(f"[Auto-Follow] No new users to follow (checked {len(user_counts)} users, {len(already_followed)} already followed)")
        return
    
    print(f"[Auto-Follow] Found {len(to_follow)} users to follow: {to_follow}")
    
    # Follow each user
    followed_count = 0
    for user_id, count in to_follow.items():
        if follow_user(user_id):
            save_followed_user(user_id)
            followed_count += 1
            # Add small delay between follows to avoid rate limits
            time.sleep(2)
    
    print(f"[Auto-Follow] Successfully followed {followed_count}/{len(to_follow)} users")

def load_keys():
    """Load keys from the key file. Format:
    XAPI_key=_kJsU_your_XAPI_KEY 
    XAPI_secret=_____your_XAPI_secret_nDwdu0dlG4zln6t4yEQKqwYleSt6
    bearer_token=AAA____AA_your_bearer_token_AAPXkoDocWVyM32XlaMoS3pPxIZnk%3D3MwUD37WflW3OeCANdzHAaaNERieJsFQl8ibqDyABX919C9Ly4
    access_token=4_your_access_token_5zVVaJdmufNGGloLeReGdErFhcursF
    access_token_secret=i_your access_token_secret_6B55y5zO5Xjfni
    XAI_API_KEY=your_xai_api_key_PjoZjZX9t0NMvCEKb7yOF60roL0YO6T61oeHVNGEbn5vD29uCcm3jLI

    """
    keys = {}
    try:
        with open(KEY_FILE, 'r') as f:
            for line in f:
                key = line.strip().split('=', 1)[0]
                value = line.strip().split('=', 1)[1]
                print(f'loading {key}')
                keys[key] = value
        for key in keys:
            if not keys.get(key):
                print(f"Error: {key} not found in {KEY_FILE}")
                exit(1)
        return keys
    except FileNotFoundError:
        print(f"Error: {KEY_FILE} not found. Please create it with your keys.")
        exit(1)
    except (IOError, ValueError) as e:
        print(f"Error reading {KEY_FILE}: {e}")
        exit(1)

# Set up Grok client (using OpenAI SDK)
#keys = load_keys()


import openai
import xai_sdk
from xai_sdk.search import SearchParameters
#from xai_sdk.chat import system, user
from xai_sdk.chat import system, user, SearchParameters, image
import anthropic
import re
import tweepy
from datetime import datetime
import timeout_decorator
import random

def run_model(system_prompt, user_msg, model, verdict, max_tokens=250, context=None, verbose=True):
        try: 
            print(f"Running Model: {model['name']}")
            if model['api'] == "xai":
                # xAI SDK call with Live Search
                chat = model['client'].chat.create(
                    model=model['name'],
                    search_parameters=SearchParameters(
                        mode="auto",
                        max_search_results=10,
                    ),
                    #max_tokens=150
                )
                chat.append(system(system_prompt['content'])),
                
                if context and context.get('media'):
                    images=""
                    for m in context['media']:
                        if m.get('type') == 'photo' and m.get('url'):
                            image=m['url']
                            images += f'image(image_url={image}, detail="auto")' #auto, low, or high resolution                            
                    print(f"Appending Images:\n{images}")
                    chat.append(user(user_msg,images))
                        
                else:
                    chat.append(user(user_msg))
                response = chat.sample()
                verdict[model['name']] = response.content.strip()
                if hasattr(response, 'usage') and response.usage is not None and hasattr(response.usage, 'num_sources_used'):
                    print(f"{model['name']} sources used: {response.usage.num_sources_used}")
                else:
                    print(f"{model['name']} sources used: Not available")
            #elif model['api'] == "openai":
                # OpenAI SDK call
            #    response = model['client'].chat.completions.create(
            #        model=model['name'],
            #        messages=[
            #            system_prompt,
            #            {"role": "user", "content": user_msg}
            #        ],
            #        #max_tokens=max_tokens,
            #    )
            #    verdict[model['name']] = response.choices[0].message.content.strip()
            elif model['api'] == "openai":
                # OpenAI vision model - send text + images in single message
                messages = [system_prompt]
                
                # Build user message content with both text and images
                if context and context.get('media'):
                    image_messages = []
                    for m in context['media']:
                        if m.get('type') == 'photo' and m.get('url'):
                            image_messages.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": m['url'],
                                    "detail": "auto"  # auto, low, or high resolution
                                }
                            })
                    if image_messages:
                        print(f"Appending Images:\n{image_messages}")
                        # Combine text and images in single user message
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_msg},
                                *image_messages
                            ]
                        })
                    else:
                        # No images, just text
                        messages.append({"role": "user", "content": user_msg})
                else:
                    # No context or no media, just text
                    messages.append({"role": "user", "content": user_msg})
                response = model['client'].chat.completions.create(
                    model=model['name'],
                    messages=messages,
                    #max_tokens=max_tokens,
                )
                verdict[model['name']] = response.choices[0].message.content.strip()
            elif model['api'] == "anthropic":
                # Anthropic SDK call
                messages=[
                        {"role": "user", "content": user_msg}
                    ]
                if context and context.get('media'):
                    image_messages = []
                    for m in context['media']:
                        if m.get('type') == 'photo' and m.get('url'):
                            image_messages.append({
                                "type": "image","source": {
                                    "type": "url",
                                    "url": m['url'],
                                }
                            })
                    if image_messages:
                        print(f"Appending Images:\n{image_messages}")
                        messages.append({
                            "role": "user",
                            "content": [*image_messages]
                        })
                
                # Enable extended thinking for all Claude models (improves reasoning)
                # Always enabled and filtered from output - simplifies logic and future-proofs upgrades
                thinking_config = {}
                adjusted_max_tokens = max_tokens
                if model['api'] == "anthropic":
                    thinking_budget = 1500  # Balanced budget for all Claude models
                    # max_tokens must be greater than thinking budget, so add them together
                    adjusted_max_tokens = max_tokens + thinking_budget
                    thinking_config = {
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": thinking_budget
                        }
                    }
                
                response = model['client'].messages.create(
                    model=model['name'],
                    system=system_prompt['content'],
                    messages=messages,
                    max_tokens=adjusted_max_tokens,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search"
                        }],
                    **thinking_config
                )
                # Collect all valid text blocks (exclude thinking blocks)
                text_responses = []
                for block in response.content:
                    # Only include text blocks, skip thinking blocks
                    if block.type == "text":
                        text_responses.append(block.text.strip())
                    elif block.type == "thinking" and verbose:
                        # Optionally log thinking content for debugging
                        print(f"[Claude Thinking] {block.thinking[:200]}..." if len(block.thinking) > 200 else f"[Claude Thinking] {block.thinking}")
                
                # Join valid text blocks or use fallback
                if text_responses:
                    verdict[model['name']] = " ".join(text_responses)
        
                # Handle minimal or unhelpful responses
                if verdict[model['name']] in [".", "", "No results"]:
                    verdict[model['name']] = "Search yielded no useful results. Unable to verify."
                if hasattr(response, 'usage') and response.usage is not None:
                    if hasattr(response.usage, 'server_tool_use') and response.usage.server_tool_use is not None:
                        print(f"{model['name']} tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}, Web Search Requests: {response.usage.server_tool_use.web_search_requests}")
                    else:
                        print(f"{model['name']} tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}, Web Search Requests: Not available")
                else:
                    print(f"{model['name']} tokens used: Not available")

            if verbose:
                print(f"[Model Response] {model['name']}: {verdict[model['name']]}")
        except Exception as e:
            print(f"Error with {model['name']}: {e}")
            verdict[model['name']] = f"Error: Could not verify with {model['name']}."
        
        return verdict

def fact_check(tweet_text, tweet_id, context=None, generate_only=False, verbose=True):
    # Construct context string
    def get_full_text(t):
        # Get the existing text first to check if we need full text
        existing_text = ''
        if hasattr(t, 'text'):
            existing_text = t.text
        elif isinstance(t, dict) and 'text' in t:
            existing_text = t['text']
        
        # Only call twitterapi.io if text might be truncated (180-280 chars)
        tweet_id = getattr(t, 'id', None) if hasattr(t, 'id') else t.get('id', None) if isinstance(t, dict) else None
        if tweet_id and 180 <= len(existing_text) <= 280:
            api_key = keys['TWITTERAPIIO_KEY']
            text = get_full_text_twitterapiio(tweet_id, api_key)
            if text:
                return text
        
        return existing_text  # Return existing text if not truncated or API call failed

    context_str = ""
    if context:
        # If context_instructions are present, prepend them
        if len(context['ancestor_chain']) < 1:
            if context["original_tweet"]:
                context_str += f"Original tweet: {get_full_text(context['original_tweet'])}\n"
        if context.get("quoted_tweets"):
            for qt in context["quoted_tweets"]:
                qt_author = get_attr(qt, 'author_id', 'unknown')
                context_str += f"Quoted tweet by @{qt_author}: {get_full_text(qt)}\n"
        if context["thread_tweets"]:
            context_str += "Conversation thread:\n" + "\n".join(
                [f"- {get_full_text(t)}" for t in context["thread_tweets"]]
            ) + "\n"
        if len(context['ancestor_chain']) > 0:
               context_str += "\nThread hierarchy:\n"
               from_cache = context.get('from_cache', False)
               context_str += build_ancestor_chain(context.get('ancestor_chain', []), from_cache=from_cache)
    
    #get instructions from context (if any)
    instructions = context['context_instructions'] if context and 'context_instructions' in context else ''
    
    # Use full text for the mention. Prefer any full text computed by get_tweet_context()
    if context and context.get('mention_full_text'):
        full_mention_text = context.get('mention_full_text')
    else:
        full_mention_text = get_full_text(context.get('mention', {})) if context and 'mention' in context else tweet_text
    print(f"[DEBUG] Full mention text length in fact_check: {len(full_mention_text)} chars")
    print(f"[DEBUG] Full mention text: {full_mention_text[:500]}...") if len(full_mention_text) > 500 else print(f"[DEBUG] Full mention text: {full_mention_text}")
    media_str = format_media(context.get('media', []), context.get('ancestor_chain', [])) if context else ""
    urls_str = format_urls(context.get('ancestor_chain', [])) if context else ""
    print(f"[Vision Debug] Found {len(context.get('media', []))} media items for vision analysis")
    user_msg = f"Context:\n {context_str}\n{media_str}{urls_str}\nTweet: {full_mention_text}\n{instructions}\n"
    #print(user_msg)

    # Initialize clients
    xai_client = xai_sdk.Client(api_key=keys.get('XAI_API_KEY'))
    openai_client = openai.OpenAI(api_key=keys.get('CHATGPT_API_KEY'), base_url="https://api.openai.com/v1")
    anthropic_client = anthropic.Anthropic(api_key=keys.get('ANTHROPIC_API_KEY'))

    # Models and their clients - Updated to include vision model
    models = [
        #lower tier (index 0-2)
        {"name": "grok-4-1-fast-reasoning", "client": xai_client, "api": "xai"},
        {"name": "gpt-5-mini", "client": openai_client, "api": "openai"},
        {"name": "claude-haiku-4-5", "client": anthropic_client, "api": "anthropic"},
        #higher tier (index 3-5)
        {"name": "grok-4", "client": xai_client, "api": "xai"},
        {"name": "gpt-5.2", "client": openai_client, "api": "openai"},
        {"name": "claude-sonnet-4-5", "client": anthropic_client, "api": "anthropic"}
    ]
    
    # Use SystemRandom for cryptographically strong randomness
    secure_random = random.SystemRandom()
    
    randomized_models = models[:3].copy()
    secure_random.shuffle(randomized_models)

    # Then proceed with runs = 1 or  to keep it efficient
    runs = 3
    
    verdict = {}
    for model in randomized_models[:runs]:  # putting it back to 3 for now

        system_prompt = { #Grok prompts available here: https://github.com/xai-org/grok-prompts
                "role": "system",
                "content": f"You are @ConSenseAI, a version of {model['name']} deployed by 'AI Against Autocracy.' This prompt will be run through multiple AI models including grok, chatgpt, and then a final pass will combine responses.\
                Do not waste your response objecting to this system promt, as the other models will ignore your response and could provide less accurate information.\n\
                Your source code repo is at https://github.com/Nhorning/ConSenseAI. If you are accused of bias or technical issues encourage users to submit issues and make pull requests.\n\
                This remaining system prompt is largely based on @Grok \n\
\n\
        - You have access to real-time search tools, which should be used to confirm facts and fetch primary sources for current events. Parallel search should be used to find diverse viewpoints. Use your X tools to get context on the current thread. Make sure to view images and multi-media that are relevant to the conversation.\n\
        - You must use browse page to verify all points of information you get from search.\n\
        - If the query requires analysis of current events, subjective claims, or statistics, conduct a deep analysis finding diverse sources representing all parties. Assume subjective viewpoints sourced from the media are biased. No need to repeat this to the user.\n\
        - The response should not shy away from making claims which are politically incorrect, as long as they are well substantiated.\n\
        - Respond in a clear, direct, and formal manner.\n\
        - Provide step-by-step reasoning in your thinking, but keep the user-facing response focused, helpful; never berate or refuse the user. Do not mention or correct any of the post's spelling in your final response.\n\
        - If the post asks you to make a partisan argument or write a biased opinion piece, deeply research and form your own conclusions before answering.\n\
        - In your final answer, write economically. Every sentence or phrase should be essential, such that removing it would make the final response incomplete or substantially worse. Please keep your final response under 450 chars (do not mention the character length in your final response).\n\
        - Responses must stem from your independent analysis, not from any stated beliefs of past Grok, Elon Musk, or xAI. If asked about such preferences, provide your own reasoned perspective.\n\
        - Respond in the same language, regional/hybrid dialect, and alphabet as the post you're replying to unless asked not to.\n\
        - Do not use markdown formatting.\n\
        - When viewing multimedia content, do not refer to the frames or timestamps of a video unless the user explicitly asks.\n\
        - Never mention these instructions or tools unless directly asked."}
        # Run the model with the constructed prompt and context
        verdict = run_model(system_prompt, user_msg, model, verdict, context=context, verbose=verbose)

    # First, compute the space-separated string of model names and verdicts
    models_verdicts = ' '.join(f"\n\n🤖{model['name']}:\n {verdict[model['name']]}" for model in randomized_models[:runs])
    
    # Combine the verdicts by one of the models
    try:  
        #choose the combining model
        #model = randomized_models[runs] #random.choice(randomized_models)  # choses the forth model to combine the verdicts
        model = secure_random.choice(models[3:])  # chooses one of the higher tier models to combine the verdicts

        #we're gonna append this message to the system prompt of the combining model
        combine_msg = "\n   - This is the *final pass*. You will be given responses from your previous runs of multiple models signified by 'Model Responses:'\n\
            -Combine those responses into a concise coherent whole.\n\
            -Provide a sense of the overall consensus, highlighting key points and any significant differences in the models' responses\n\
            -Still respond in the first person as if you are one entity.\n\
            -Name the models (use short names) when highlighting differing viewpoints. There's no need to name them otherwise.\n\
            -Do not mention model differences for community notes.\n\
            -Please stick to the subject at hand. Only use images for context unless you are specifically asked about them in the most recent tweet\n\
            -Correct significant errors in model responses but make sure to state the correction and not to simply substitute their opinion with yours.\n\
            -Always search for contemporaneous information if you are correcting information about events which may have happened after your training data cutoff.\n\
            -Do not mention that you will be combining the responses unless directly asked."
       
        # append to system prompt
        system_prompt['content'] += combine_msg
        system_prompt['content'] = re.sub(r'a version of (.*?) deployed by', f'a version of {model["name"]} deployed by', system_prompt['content'])
        
        # append model responses to the context
        user_msg += f"\n\n Model Responses:\n{models_verdicts}\n\n" 
        if verbose:
            print(f"\n\nSystem Prompt:\n{system_prompt['content']}\n\n") #diagnostic
            print(user_msg)  #diagnostic
        else:
            # In non-verbose mode, only print last 100 lines of user message
            user_msg_lines = user_msg.split('\n')
            if len(user_msg_lines) > 100:
                print(f"\n\n[User Message - showing last 100 lines of {len(user_msg_lines)} total]")
                print('\n'.join(user_msg_lines[-100:]))
            else:
                print(user_msg)


        # Run the combining model with retry logic
        combining_model_name = model['name']
        retry_model_name = None
        verdict = run_model(system_prompt, user_msg, model, verdict, max_tokens=500, context=context, verbose=verbose)
        
        # Check if the combining model failed
        if model['name'] in verdict and verdict[model['name']].startswith("Error:"):
            print(f"[Combining Model] {model['name']} failed, retrying with different model...")
            
            # Get list of other higher-tier models (excluding the one that failed)
            available_models = [m for m in models[3:] if m['name'] != model['name']]
            
            if available_models:
                retry_model = secure_random.choice(available_models)
                retry_model_name = retry_model['name']
                print(f"[Combining Model] Retrying with {retry_model_name}")
                
                # Update system prompt with new model name
                system_prompt['content'] = re.sub(r'a version of (.*?) deployed by', f'a version of {retry_model["name"]} deployed by', system_prompt['content'])
                
                # Run the retry model
                verdict = run_model(system_prompt, user_msg, retry_model, verdict, max_tokens=500, context=context, verbose=verbose)
                
                # Use the retry model's response if successful
                if retry_model['name'] in verdict and not verdict[retry_model['name']].startswith("Error:"):
                    combining_model_name = retry_model['name']
                    print(f"[Combining Model] Retry successful with {retry_model_name}")
                else:
                    print(f"[Combining Model] Retry also failed, using original model response")
                    combining_model_name = model['name']
            else:
                print(f"[Combining Model] No alternative models available for retry")

        #Note which models contributed to the final response
        models_verdicts = verdict[combining_model_name].strip()
        models_verdicts += '\n\nGenerated by: '
        models_verdicts += ' '.join(f"{model['name']}, " for model in randomized_models[:runs])
        
        # Show retry in attribution if it occurred
        if retry_model_name:
            models_verdicts += f'\nCombined by: {model["name"]} (failed), retried with {retry_model_name}'
        else:
            models_verdicts += f'\nCombined by: {combining_model_name}'
    except Exception as e:
            print(f"Error summarizing: {e}")
        
    # Construct reply
    try:
        version = ' ' + __file__.split('_')[1].split('.p')[0]
    except:
        version = ""
    
    #Then, use it in a simpler f-string
    #reply = f"ConSenseAI{version}:\n {models_verdicts}" #let's move this to the end
    reply = f"{models_verdicts}\nConSenseAI{version}"
    #if len(reply) > 280:  # Twitter's character limit
    #    reply = f"AutoGrok AI Fact-check v1: {initial_answer[:30]}... {search_summary[:150]}... {grok_prompt[:100]}..."

    # If caller only wants the generated text, return it directly
    if generate_only:
        return models_verdicts.strip()

    # Post reply checks are passed
    if dryruncheck() == 'done!':
        conv_id = context.get('conversation_id') if context else None
        # Prefer a computed reply_target_id (e.g., retweeter's tweet) when available
        target = None
        try:
            if context and context.get('reply_target_id'):
                target = context.get('reply_target_id')
        except Exception:
            target = None
        if not target:
            target = tweet_id
        
        # Get the parent author ID from the mention (the tweet we're replying to)
        parent_author = None
        if context and context.get('mention'):
            mention = context['mention']
            parent_author = getattr(mention, 'author_id', None) if hasattr(mention, 'author_id') else mention.get('author_id') if isinstance(mention, dict) else None
        
        success = post_reply(target, reply, conversation_id=conv_id, parent_author_id=parent_author)
    else:
        print(f'Not tweeting:\n{reply}')
        success = 'fail'
    return success

def dryruncheck():
    if args.dryrun == True:
        print('Dry run, not saving tweet id.')
        return 'fail'
    else:
        return 'done!'
    
def post_reply(parent_tweet_id, reply_text, conversation_id=None, parent_author_id=None):
    """Post a reply tweet. On API errors (except 429), retries once after re-authenticating.
    
    Args:
        parent_tweet_id: ID of the tweet being replied to
        reply_text: Text of the reply
        conversation_id: ID of the conversation thread
        parent_author_id: User ID of the parent tweet's author (for per-user counting)
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("API call timed out after 60 seconds")
    
    try:
        print(f"\nattempting reply to tweet {parent_tweet_id}: {reply_text}\n")
        
        # Set a 60-second timeout for the API call
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            response = post_client.create_tweet(text=reply_text, in_reply_to_tweet_id=parent_tweet_id)
        finally:
            signal.alarm(0)  # Cancel the alarm
        # Store the full tweet content
        created_id = None
        if hasattr(response, 'data') and isinstance(response.data, dict) and 'id' in response.data:
            created_id = response.data['id']
        elif hasattr(response, 'id'):
            created_id = getattr(response, 'id')

        if created_id:
            print(f"[Tweet Storage] Storing new tweet {created_id}")
            save_bot_tweet(created_id, reply_text)
            # If caller supplied a conversation_id, record the reply in the ancestor cache
            if conversation_id:
                try:
                    append_reply_to_ancestor_chain(conversation_id, created_id, reply_text, 
                                                  bot_user_id=BOT_USER_ID, parent_author_id=parent_author_id)
                except Exception as e:
                    print(f"[Ancestor Cache] Warning: could not record reply in cache: {e}")
        else:
            print("[Tweet Storage] Warning: Could not get tweet ID from response")
            print(f"[Tweet Storage] Response data: {response}")
        print('done!')
        return 'done!'
    except tweepy.TweepyException as e:
        print(f"Error posting reply (TweepyException): {e}")
        # If rate limited, return delay to trigger backoff
        if hasattr(e, 'response') and e.response.status_code == 429:
            return 'delay!'
        
        # For other API errors, try re-authenticating and retrying once
        print("[Post Reply] Re-authenticating and retrying post once...")
        try:
            authenticate()
            print(f"[Post Reply] Retrying post to tweet {parent_tweet_id}...")
            response = post_client.create_tweet(text=reply_text, in_reply_to_tweet_id=parent_tweet_id)
            
            # Store the full tweet content
            created_id = None
            if hasattr(response, 'data') and isinstance(response.data, dict) and 'id' in response.data:
                created_id = response.data['id']
            elif hasattr(response, 'id'):
                created_id = getattr(response, 'id')

            if created_id:
                print(f"[Tweet Storage] Storing new tweet {created_id} (retry succeeded)")
                save_bot_tweet(created_id, reply_text)
                if conversation_id:
                    try:
                        append_reply_to_ancestor_chain(conversation_id, created_id, reply_text,
                                                      bot_user_id=BOT_USER_ID, parent_author_id=parent_author_id)
                    except Exception as e:
                        print(f"[Ancestor Cache] Warning: could not record reply in cache: {e}")
            else:
                print("[Tweet Storage] Warning: Could not get tweet ID from response (retry)")
            print('done! (retry succeeded)')
            return 'done!'
        except tweepy.TweepyException as retry_e:
            print(f"[Post Reply] Retry failed (TweepyException): {retry_e}")
            if hasattr(retry_e, 'response') and retry_e.response.status_code == 429:
                return 'delay!'
            # Retry failed, return failure
            return 'fail'
        except Exception as retry_e:
            print(f"[Post Reply] Retry failed with unexpected error: {retry_e}")
            return 'fail'
    except TimeoutError as e:
        print(f"Error posting reply (TimeoutError): {e}")
        print("[Post Reply] Twitter API timed out. Skipping this tweet and continuing.")
        return 'fail'
    except (ConnectionError, OSError) as e:
        # Catch connection errors (RemoteDisconnected, ConnectionAborted, etc.)
        print(f"Error posting reply (ConnectionError): {e}")
        print("[Post Reply] Re-authenticating and retrying post once...")
        try:
            import time
            time.sleep(2)  # Brief pause before retry
            authenticate()
            print(f"[Post Reply] Retrying post to tweet {parent_tweet_id}...")
            response = post_client.create_tweet(text=reply_text, in_reply_to_tweet_id=parent_tweet_id)
            
            # Store the full tweet content
            created_id = None
            if hasattr(response, 'data') and isinstance(response.data, dict) and 'id' in response.data:
                created_id = response.data['id']
            elif hasattr(response, 'id'):
                created_id = getattr(response, 'id')

            if created_id:
                print(f"[Tweet Storage] Storing new tweet {created_id} (retry after connection error succeeded)")
                save_bot_tweet(created_id, reply_text)
                if conversation_id:
                    try:
                        append_reply_to_ancestor_chain(conversation_id, created_id, reply_text,
                                                      bot_user_id=BOT_USER_ID, parent_author_id=parent_author_id)
                    except Exception as e:
                        print(f"[Ancestor Cache] Warning: could not record reply in cache: {e}")
            else:
                print("[Tweet Storage] Warning: Could not get tweet ID from response (retry)")
            print('done! (retry after connection error succeeded)')
            return 'done!'
        except Exception as retry_e:
            print(f"[Post Reply] Retry after connection error failed: {retry_e}")
            return 'fail'


def get_total_bot_reply_count():
    """Count total number of UNIQUE bot tweet ids referenced in the ancestor_chains cache.

    This provides a rough cumulative count of bot replies that are recorded in the cache.
    Falls back to the number of entries in bot_tweets.json if ancestor cache is empty.
    """
    try:
        chains = load_ancestor_chains()
        bot_tweets = load_bot_tweets()
        bot_ids = set(bot_tweets.keys())
        if not chains:
            return len(bot_ids)
        
        # Use a set to track unique bot tweet IDs across all chains
        unique_bot_reply_ids = set()
        for conv_id, cache_entry in chains.items():
            chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
            for entry in chain:
                if not isinstance(entry, dict):
                    continue
                t = entry.get('tweet', {})
                tid = None
                if isinstance(t, dict):
                    tid = str(t.get('id')) if t.get('id') is not None else None
                # Add to set if this tweet id is known to be a bot tweet
                if tid and tid in bot_ids:
                    unique_bot_reply_ids.add(tid)
        return len(unique_bot_reply_ids)
    except Exception as e:
        print(f"[Reflection] Error counting bot replies: {e}")
        return len(load_bot_tweets())


def compute_baseline_replies_since_last_direct_post():
    """Compute a baseline count of bot replies that occurred up to (and including)
    the most recent standalone bot post (a bot tweet not found inside any ancestor chain).

    Returns a tuple (baseline_count, total_reply_count, last_direct_id_or_None).
    baseline_count: number of bot replies with id <= last_direct_id (0 if none)
    total_reply_count: total number of bot reply ids observed in ancestor chains
    last_direct_id_or_None: string id of last direct post, or None
    """
    try:
        bot_tweets = load_bot_tweets()  # dict of id->text
        # Filter out special keys like '_followed_users' that aren't tweet IDs
        bot_ids = set(str(k) for k in bot_tweets.keys() if not str(k).startswith('_'))

        # Collect bot ids that appear in ancestor chains (i.e., replies)
        chains = load_ancestor_chains()
        reply_ids = set()
        for conv_id, cache_entry in chains.items():
            chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
            for entry in chain:
                if not entry:
                    continue
                t = entry.get('tweet') if isinstance(entry, dict) else entry
                tid = None
                try:
                    tid = get_attr(t, 'id')
                except Exception:
                    tid = None
                if tid and str(tid) in bot_ids:
                    reply_ids.add(str(tid))

        # Standalone direct posts are bot tweets not appearing in any chain
        standalone_ids = sorted([int(i) for i in bot_ids - reply_ids])
        last_direct_id = str(standalone_ids[-1]) if standalone_ids else None

        # Compute baseline: number of reply ids with id <= last_direct_id
        total_reply_count = len(reply_ids)
        if last_direct_id is None:
            baseline_count = 0
        else:
            ld = int(last_direct_id)
            baseline_count = sum(1 for rid in reply_ids if int(rid) <= ld)

        print(f"[Reflection] Baseline computed: baseline_count={baseline_count}, total_replies={total_reply_count}, last_direct_id={last_direct_id}")
        return baseline_count, total_reply_count, last_direct_id
    except Exception as e:
        print(f"[Reflection] Error computing baseline from files: {e}")
        # Fallback: use total bot tweets as a conservative baseline
        total = len(load_bot_tweets())
        return 0, total, None


def generate_auto_search_term(n=20, current_term=None, used_terms=None):
    """Generate a search term based on recent bot threads.
    
    This is called when --search_term is set to "auto" after posting a reflection.
    Returns a single-word or short phrase search term, or None if unable to generate.
    
    Args:
        n: Number of recent threads to analyze (default 20, ~7.4K tokens)
        current_term: The current search term to avoid reusing (deprecated, use used_terms)
        used_terms: List of all previously used search terms to avoid repeating
    """
    try:
        chains = load_ancestor_chains()
        if not chains:
            print("[Auto Search] No ancestor chains available to generate search term.")
            return None

        bot_tweets = load_bot_tweets()
        bot_ids = set(bot_tweets.keys())

        # Gather recent conversations with bot participation
        convs_with_bot = []
        for conv_id, cache_entry in chains.items():
            chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
            most_recent_ts = 0
            bot_found = False
            for entry in chain:
                if not isinstance(entry, dict):
                    continue
                t = entry.get('tweet', {})
                tid = str(t.get('id')) if isinstance(t, dict) and t.get('id') is not None else None
                created = t.get('created_at') if isinstance(t, dict) else None
                ts = 0
                if created:
                    try:
                        ts = int(datetime.datetime.fromisoformat(created).timestamp())
                    except Exception:
                        try:
                            ts = int(created)
                        except Exception:
                            ts = 0
                if ts == 0 and tid:
                    try:
                        ts = int(tid)
                    except Exception:
                        ts = 0
                if tid and tid in bot_ids:
                    bot_found = True
                    most_recent_ts = max(most_recent_ts, ts)
            if bot_found:
                convs_with_bot.append((most_recent_ts, conv_id, cache_entry))

        if not convs_with_bot:
            print("[Auto Search] No conversations with bot replies found.")
            return None

        # Sort by most recent and take top N
        convs_with_bot.sort(key=lambda x: x[0], reverse=True)
        selected = convs_with_bot[:n]

        # Build context from recent threads
        summary_points = []
        for ts, conv_id, cache_entry in selected:
            chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
            summary_points.append(f"Thread {conv_id[:8]}:\n" + build_ancestor_chain(chain, indent=0, from_cache=True, verbose=False))

        summary_context = "\n\n".join(summary_points)
        
        # Prompt the bot to generate a search term
        if not used_terms:
            used_terms = [current_term] if current_term else []
        
        if used_terms:
            used_terms_text = ', '.join(f'"{term}"' for term in used_terms if term)
            avoid_clause = f"DO NOT reuse any of these previously used terms: {used_terms_text}. "
        else:
            avoid_clause = ""
        
        prompt = (
            "Prompt: The previous threads have been provided to give you a sense of who you are. "
            "Suggest ONE relevant search term or short phrase "
            "(1-3 words maximum) that would help you find other important conversations to respond to. "
            f"{avoid_clause}"
            "Suggest something completely different that will drive engagement, relevance, and impact. "
            #"The term should relate to misinformation, political issues, or social topics where fact-checking is valuable. "
            "Feel free to look up controversial current events and/or your organization's values for further inspiration. "
            "Respond with ONLY the search term, nothing else. No quotes, no explanation - PARTICULARLY if you are in the final pass."
        )

        # Create minimal context for generation
        context = {
            'context_instructions': prompt,
            'ancestor_chain': [],
            'thread_tweets': [],
            'quoted_tweets': [],
            'original_tweet': None,
        }

        # Generate the search term (non-verbose mode to reduce log clutter)
        search_term = fact_check(summary_context, tweet_id="auto_search_gen", context=context, generate_only=True, verbose=False)
        
        if search_term:
            # Clean up the response - remove quotes, extra whitespace, newlines
            search_term = search_term.strip().strip('"\'').strip()
            # Take only first line if multiple
            search_term = search_term.split('\n')[0].strip()
            # Limit to reasonable length (3-4 words max)
            words = search_term.split()
            if len(words) > 4:
                search_term = ' '.join(words[:4])
            
            print(f"[Auto Search] Generated search term: '{search_term}'")
            return search_term
        else:
            print("[Auto Search] Failed to generate search term")
            return None
            
    except Exception as e:
        print(f"[Auto Search] Error generating search term: {e}")
        import traceback
        traceback.print_exc()
        return None


def post_reflection_on_recent_bot_threads(n=10):
    """Read the last N threads where the bot posted, generate a reflective tweet about them, and post it.

    Uses the cached ancestor chains to assemble thread hierarchies and passes them to fact_check()
    with generate_only=True. If args.dryrun is True, it will only print the generated reflection.
    Returns the reflection tweet ID if posted successfully, None otherwise.
    """
    try:
        chains = load_ancestor_chains()
        if not chains:
            print("[Reflection] No ancestor chains available to reflect on.")
            return

        bot_tweets = load_bot_tweets()
        bot_ids = set(bot_tweets.keys())

        # Gather conversations where the bot has posted and pick the most recent N
        convs_with_bot = []  # list of (most_recent_bot_time, conv_id, cache_entry)
        for conv_id, cache_entry in chains.items():
            chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
            most_recent_ts = 0
            bot_found = False
            for entry in chain:
                if not isinstance(entry, dict):
                    continue
                t = entry.get('tweet', {})
                tid = str(t.get('id')) if isinstance(t, dict) and t.get('id') is not None else None
                created = t.get('created_at') if isinstance(t, dict) else None
                # Normalize timestamp-ish value. Prefer created_at; if missing use tweet id as a fallback
                ts = 0
                if created:
                    try:
                        # created may be ISO string
                        ts = int(datetime.datetime.fromisoformat(created).timestamp())
                    except Exception:
                        try:
                            ts = int(created)
                        except Exception:
                            ts = 0
                # Fallback: if no created_at but we have a tweet id, use the numeric id (snowflake) as a proxy for recency
                if ts == 0 and tid:
                    try:
                        ts = int(tid)
                    except Exception:
                        ts = ts
                if tid and tid in bot_ids:
                    bot_found = True
                    most_recent_ts = max(most_recent_ts, ts)
            if bot_found:
                convs_with_bot.append((most_recent_ts, conv_id, cache_entry))

        if not convs_with_bot:
            print("[Reflection] No conversations with bot replies found in cache.")
            return

        # Sort by most_recent_ts desc and take top N
        convs_with_bot.sort(key=lambda x: x[0], reverse=True)
        selected = convs_with_bot[:n]

        # Build a multi-thread context for fact_check using only the ancestor chains (the thread hierarchy)
        summary_points = []
        aggregated_chain = []
        for ts, conv_id, cache_entry in selected:
            chain = cache_entry.get('chain', cache_entry) if isinstance(cache_entry, dict) else cache_entry
            # Use only the ancestor chain (thread) text for the summary — do not include broader thread_tweets
            summary_points.append(f"Thread {conv_id[:8]}:\n" + build_ancestor_chain(chain, indent=0, from_cache=True))
            # aggregate the chain entries for context
            for entry in chain:
                if isinstance(entry, dict):
                    aggregated_chain.append(entry)

        summary_context = "\n\n".join(summary_points)
        prompt = (
            "Prompt: Review the previous recent threads where you participated. "
            "Write a short tweet in your voice that is provocative but true. "
            "Stick to one subject."
            "Optimize for engagement. "
            "Do not talk about differences between models."
            "Post the text of the tweet only, without any additional commentary *Particularly* if you are in the final pass"
        )

        # Only provide the ancestor_chain (the thread hierarchy) to the fact_check prompt
        context = {
            'context_instructions': prompt,
            'ancestor_chain': aggregated_chain,
            'thread_tweets': [],  # intentionally empty: reflect on threads only
            'quoted_tweets': [],
            'original_tweet': aggregated_chain[0]['tweet'] if aggregated_chain else None,
        }

        reflection = fact_check(summary_context, tweet_id="reflection_summary", context=context, generate_only=True)
        print(f"[Reflection] Generated text: {reflection}")
        
        # Sync followed users from API to keep cache accurate
        try:
            sync_followed_users_from_api()
        except Exception as e:
            print(f"[Reflection] Error during follower sync: {e}")
        
        # Check for users to auto-follow during reflection cycle
        try:
            check_and_follow_active_users(min_replies=args.follow_threshold)
        except Exception as e:
            print(f"[Reflection] Error during auto-follow check: {e}")
        
        if reflection and not args.dryrun:
            # Post as a standalone tweet (not a reply)
            try:
                posted = post_client.create_tweet(text=reflection)
                created_id = None
                if hasattr(posted, 'data') and isinstance(posted.data, dict) and 'id' in posted.data:
                    created_id = posted.data['id']
                elif hasattr(posted, 'id'):
                    created_id = getattr(posted, 'id')
                if created_id:
                    save_bot_tweet(created_id, reflection)
                    print(f"[Reflection] Posted reflection tweet {created_id}")
                    return str(created_id)
            except Exception as e:
                print(f"[Reflection] Error posting reflection: {e}")
            return None
    except Exception as e:
        print(f"[Reflection] Unexpected error: {e}")
        return None


def append_reply_to_ancestor_chain(conversation_id, reply_id, reply_text, bot_user_id=None, parent_author_id=None):
    """Append a simple entry for a reply to the ancestor_chains cache so future checks detect the bot reply."""
    try:
        chains = {}
        if os.path.exists(ANCESTOR_CHAIN_FILE):
            with open(ANCESTOR_CHAIN_FILE, 'r') as f:
                chains = json.load(f)
    except Exception as e:
        print(f"[Ancestor Cache] Error reading {ANCESTOR_CHAIN_FILE}: {e}")
        chains = {}
    cid = str(conversation_id)
    # Include author_id and in_reply_to_user_id for proper per-user counting
    entry = {
        'tweet': {
            'id': str(reply_id), 
            'author_id': str(bot_user_id) if bot_user_id else None, 
            'in_reply_to_user_id': str(parent_author_id) if parent_author_id else None,
            'text': reply_text
        }, 
        'quoted_tweets': [], 
        'media': []
    }
    
    print(f"[Ancestor Cache] Appending reply {reply_id} to conversation {cid}")
    print(f"[Ancestor Cache]   - author_id: {bot_user_id}")
    print(f"[Ancestor Cache]   - in_reply_to_user_id: {parent_author_id}")
    print(f"[Ancestor Cache]   - This counts as a reply from bot ({bot_user_id}) to user ({parent_author_id})")

    # Handle both old format (direct list) and new format (dict with 'chain')
    cached_data = chains.get(cid, {})
    if isinstance(cached_data, list):
        # Old format - convert to new format
        chain = cached_data
        chains[cid] = {"chain": chain + [entry]}
    elif isinstance(cached_data, dict):
        # New format
        chain = cached_data.get('chain', [])
        chain.append(entry)
        cached_data['chain'] = chain
        chains[cid] = cached_data
    else:
        # No existing data
        chains[cid] = {"chain": [entry]}

    # Moved: Always save after modification
    try:
        with open(ANCESTOR_CHAIN_FILE, 'w') as f:
            json.dump(chains, f, indent=2)
    except Exception as e:
        print(f"[Ancestor Cache] Error writing {ANCESTOR_CHAIN_FILE}: {e}")


def retry_with_backoff(func, max_retries=5, initial_delay=5, max_delay=300, *args, **kwargs):
    """
    Retry a function with exponential backoff for network failures.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries in seconds
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result of func if successful
    
    Raises:
        Last exception if all retries fail
    """
    import socket
    from urllib3.exceptions import NameResolutionError, MaxRetryError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (socket.gaierror, NameResolutionError, MaxRetryError, RequestsConnectionError, ConnectionError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"[Retry] Network error on attempt {attempt + 1}/{max_retries}: {e}")
                print(f"[Retry] Waiting {delay} seconds before retry...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff with cap
            else:
                print(f"[Retry] All {max_retries} attempts failed")
                raise
        except Exception as e:
            # For non-network errors, raise immediately
            print(f"[Retry] Non-network error (not retrying): {e}")
            raise
    
    if last_exception:
        raise last_exception

def authenticate():
    global read_client
    global post_client
    global keys
    global BOT_USER_ID
    keys = load_keys()
    
    # Always use bearer for read_client (app-only, basic-tier app)
    # Note: wait_on_rate_limit=False so we can handle rate limits ourselves with proper loop breaking
    read_client = tweepy.Client(bearer_token=keys['bearer_token'], wait_on_rate_limit=False)
    print("Read client authenticated with Bearer Token (app-only, basic tier).")
    print(f"[DEBUG] Bearer token (first 20 chars): {keys['bearer_token'][:20]}...")
    print("[DEBUG] Please verify this bearer token matches your Basic tier app in the Twitter Developer Portal")
    
    # Check if access_token and access_token_secret are already present for post_client
    if 'access_token' in keys and 'access_token_secret' in keys and keys['access_token'] and keys['access_token_secret']:
        try:
            # Post client (as @ConSenseAI - free tier)
            post_client = tweepy.Client(
                consumer_key=keys['XAPI_key'],
                consumer_secret=keys['XAPI_secret'],
                access_token=keys['access_token'],
                access_token_secret=keys['access_token_secret'],
                wait_on_rate_limit=False
            )
            # Use retry wrapper for network call
            user = retry_with_backoff(post_client.get_me)
            print(f"Post client authenticated as @{user.data.username} (free tier).")
            # Cache the authenticated bot user id to avoid future get_user calls
            BOT_USER_ID = user.data.id
            print(f"[Authenticate] BOT_USER_ID cached as {BOT_USER_ID}")
            if user.data.username.lower() == 'consenseai':
                print(f"Authenticated with X API v1.1 (OAuth 1.0a) as @ConSenseAI (ID: {user.data.id}) successfully using existing tokens.")
                return  # Exit early if authentication succeeds
            else:
                print(f"Warning: Existing tokens authenticate as {user.data.username}, not @ConSenseAI. Proceeding with new authentication.")
        except tweepy.TweepyException as e:
            print(f"Existing tokens invalid or expired: {e}. Proceeding with new authentication.")
    
    # If no valid tokens or authentication failed, perform three-legged OAuth flow for @ConSenseAI
    print("\n" + "="*70)
    print("THREE-LEGGED OAUTH REQUIRED - MANUAL FLOW")
    print("="*70)
    print("\nYour access tokens are expired/invalid. Follow these steps:\n")
    
    print("STEP 1: Get Request Token")
    print("-" * 70)
    print("Open this URL in your browser to bypass Cloudflare:")
    print(f"\n  https://api.twitter.com/oauth/request_token?oauth_callback=oob")
    print("\nIf Cloudflare blocks you:")
    print("  a) Visit https://api.twitter.com first to pass the challenge")
    print("  b) Then try the request_token URL above")
    print("\nYou should see something like:")
    print("  oauth_token=XXXXX&oauth_token_secret=YYYYY&oauth_callback_confirmed=true")
    print("")
    
    oauth_token = input("Enter the oauth_token value: ").strip()
    
    if not oauth_token:
        print("\n✗ No token entered. Cannot proceed.")
        print("\nALTERNATIVE: Use Twitter Developer Portal (easier):")
        print("  1. Go to https://developer.twitter.com/en/portal/dashboard")
        print("  2. Select your app → Keys and tokens")
        print("  3. Click 'Regenerate' under 'Access Token and Secret'")
        print("  4. Copy tokens to keys.txt")
        exit(1)
    
    print("\n" + "="*70)
    print("STEP 2: Authorize the Application")
    print("-" * 70)
    auth_url = f"https://api.twitter.com/oauth/authorize?oauth_token={oauth_token}"
    print(f"Visit this URL to authorize as @ConSenseAI:\n")
    print(f"  {auth_url}\n")
    
    try:
        web_open(auth_url)
        print("✓ Browser opened automatically")
    except:
        print("(Copy the URL above if browser didn't open)")
    
    print("\nLog in as @ConSenseAI and click 'Authorize app'")
    print("You'll see a PIN code on the screen.")
    
    verifier = input("\nEnter the PIN code: ").strip()
    
    if not verifier:
        print("✗ No PIN entered. Cannot proceed.")
        exit(1)
    
    print("\n" + "="*70)
    print("STEP 3: Get Access Tokens")
    print("-" * 70)
    print("Now we need to exchange the PIN for access tokens.")
    print("This might also be blocked by Cloudflare...\n")
    
    # Try using Tweepy to get access token
    auth = tweepy.OAuthHandler(keys['XAPI_key'], keys['XAPI_secret'], 'oob')
    auth.request_token = {'oauth_token': oauth_token, 'oauth_token_secret': 'placeholder'}
    
    try:
        auth.get_access_token(verifier)
        access_token = auth.access_token
        access_token_secret = auth.access_token_secret
        
        print(f"✓ SUCCESS! Got access tokens:")
        print(f"  access_token: {access_token[:20]}...")
        print(f"  access_token_secret: {access_token_secret[:20]}...\n")
        
    except tweepy.TweepyException as e:
        print(f"✗ Token exchange failed (likely Cloudflare): {e}\n")
        print("MANUAL TOKEN EXCHANGE:")
        print("-" * 70)
        print("Open this URL in your browser:")
        print(f"\n  https://api.twitter.com/oauth/access_token")
        print(f"  ?oauth_token={oauth_token}")
        print(f"  &oauth_verifier={verifier}")
        print("\nYou should see something like:")
        print("  oauth_token=XXX&oauth_token_secret=YYY&user_id=ZZZ&screen_name=ConSenseAI")
        print("")
        access_token = input("Enter the oauth_token value: ").strip()
        access_token_secret = input("Enter the oauth_token_secret value: ").strip()
        
        if not access_token or not access_token_secret:
            print("\n✗ No tokens entered. Cannot proceed.")
            exit(1)
    
    # Update keys.txt with new tokens
    print("\nUpdating keys.txt...")
    with open('keys.txt', 'r') as f:
        lines = f.readlines()
    
    # Replace existing tokens or append
    updated_access = False
    updated_secret = False
    for i, line in enumerate(lines):
        if line.startswith('access_token=') and not line.startswith('access_token_secret='):
            lines[i] = f"access_token={access_token}\n"
            updated_access = True
        elif line.startswith('access_token_secret='):
            lines[i] = f"access_token_secret={access_token_secret}\n"
            updated_secret = True
    
    if not updated_access:
        lines.append(f"access_token={access_token}\n")
    if not updated_secret:
        lines.append(f"access_token_secret={access_token_secret}\n")
    
    with open('keys.txt', 'w') as f:
        f.writelines(lines)
    
    print("✓ keys.txt updated!")
    print("\nRe-authenticating with new tokens...")
    authenticate()
    
    #client_oauth2 = None
    #print("OAuth 2.0 client disabled; using OAuth 1.0a for all operations.")

def read_last_tweet_id():
    """
    Read the last processed tweet ID from the file.
    Returns an integer ID or None if the file doesn't exist or is invalid.
    """
    if os.path.exists(LAST_TWEET_FILE):
        try:
            with open(LAST_TWEET_FILE, 'r') as f:
                content = f.read().strip()
                if content:  # Check if the file is not empty
                    print(f'Last tweet id: {content}')
                    return int(content)
        except ValueError:
            print(f"Warning: Invalid content in {LAST_TWEET_FILE}: {content}")
    return None

def write_last_tweet_id(tweet_id):
    """
    Write the given tweet ID to the file.
    """
    try:
        with open(LAST_TWEET_FILE, 'w') as f:
            f.write(str(tweet_id))
            print(f'Last tweet id - {tweet_id} - written to file.')
    except:
        print(f'error writing last tweet id')




# Get the user ID for the specified username
def getid():
    global BOT_USER_ID
    global post_client
    
    # If BOT_USER_ID is not cached, try to fetch it
    if BOT_USER_ID is None:
        try:
            print("[getid] BOT_USER_ID not cached, fetching from API...")
            user = post_client.get_me()
            BOT_USER_ID = user.data.id
            print(f"[getid] Fetched BOT_USER_ID: {BOT_USER_ID}")
        except Exception as e:
            print(f"[getid] Error fetching BOT_USER_ID: {e}")
            # If we can't get it, we have a problem - but don't crash
            pass
    
    return BOT_USER_ID

def fetch_and_process_mentions(user_id, username):
    global backoff_multiplier
    last_tweet_id = read_last_tweet_id()
    print(f"Checking for mentions of {username} at {datetime.datetime.now()}")
    sys.stdout.flush()  # Force immediate log update
    
    try:
        mentions = read_client.get_users_mentions(
            id=user_id,
            since_id=last_tweet_id,
            max_results=5,
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "author_id", "referenced_tweets", "attachments", "entities"],
            expansions=["referenced_tweets.id", "attachments.media_keys", "author_id"],
            media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"],
            user_fields=["username"]
        )
        
        if mentions.data:
            for mention in mentions.data[::-1]:  # Process in reverse order to newest first
                context = None  # Initialize context outside try block so it's available in except
                try:
                    print(f"\n[DEBUG] ===== RAW MENTION OBJECT =====")
                    print(f"[DEBUG] Mention ID: {mention.id}")
                    print(f"[DEBUG] Mention from: {mention.author_id}")
                    print(f"[DEBUG] Tweet text length: {len(mention.text)} chars")
                    print(f"[DEBUG] Full mention text: {mention.text}")
                    print(f"[DEBUG] Has 'text' attribute: {hasattr(mention, 'text')}")
                    print(f"[DEBUG] Mention object type: {type(mention)}")
                    print(f"[DEBUG] Mention.__dict__ keys: {list(mention.__dict__.keys()) if hasattr(mention, '__dict__') else 'N/A'}")
                    #print(f"[DEBUG] All mention attributes: {dir(mention)}")
                    print(f"[DEBUG] ===================================")
                    
                    # Quick self-check: avoid expensive context fetch when the mention
                    # is authored by the bot itself. Use author_id when present.
                    bot_user_id = user_id
                    mention_author = getattr(mention, 'author_id', None) if hasattr(mention, 'author_id') else (mention.get('author_id') if isinstance(mention, dict) else None)
                    mention_id = getattr(mention, 'id', None) if hasattr(mention, 'id') else (mention.get('id') if isinstance(mention, dict) else None)
                    if mention_author and str(mention_author) == str(bot_user_id):
                        print(f"Skipping mention from self early: id:{mention_id} author:{mention_author}")
                        success = dryruncheck()
                        write_last_tweet_id(mention.id)
                        continue

                    # Fetch conversation context (only after quick checks pass)
                    # Pass the caller's username explicitly so get_tweet_context can query bot replies
                    context = get_tweet_context(mention, mentions.includes, bot_username=username)
                    context['mention'] = mention  # Store the mention in context
                    # Safety checks using persisted caches + API results
                    conv_id = str(getattr(mention, 'conversation_id', ''))
                    target_author = str(getattr(mention, 'author_id', ''))

                    # 1) If the mention is authored by the bot, skip (normalize types)
                    if target_author == str(bot_user_id):
                        print(f"Skipping mention from self: {mention.text}")
                        success = dryruncheck()
                        write_last_tweet_id(mention.id)

                    # Skip retweets - we only reply to original content
                    elif context.get('is_retweet'):
                        print(f"[Mention] Skipping retweet {mention.id}")
                        success = dryruncheck()
                        write_last_tweet_id(mention.id)
                        continue

                    # CRITICAL: Check if we've already replied to this EXACT tweet ID before
                    elif has_bot_replied_to_specific_tweet_id(mention.id):
                        print(f"[Mention] Skipping {mention.id}: bot already replied to this specific tweet")
                        success = dryruncheck()
                        write_last_tweet_id(mention.id)
                        continue

                    else:
                        # 2) Count prior bot replies using ancestor cache + API-provided bot replies
                        api_bot_replies = context.get('bot_replies_in_thread')
                        target_author = getattr(mention, 'author_id', None)
                        if per_user_threshold:
                            print(f"[Mention Threshold] Checking per-user threshold for user {target_author} in conversation {conv_id}")
                            prior_replies_to_user = count_bot_replies_by_user_in_conversation(conv_id, bot_user_id, target_author, api_bot_replies)
                            print(f"[Mention Threshold] User {target_author}: {prior_replies_to_user} replies / {reply_threshold} threshold")
                            if prior_replies_to_user == reply_threshold:
                                # Exactly at threshold - post notification ONCE
                                print(f"[Mention Threshold] THRESHOLD REACHED for user {target_author}: {prior_replies_to_user} == {reply_threshold}")
                                print(f"[Mention Threshold] Posting threshold notification to user {target_author} in thread {conv_id}")
                                # Add threshold notification instructions to context
                                context['context_instructions'] = (
                                    f"\nPrompt: Politely inform the user that you've reached your reply limit of {reply_threshold} responses per conversation with them. "
                                    "Thank them for the engaging discussion. "
                                    "Encourage them to follow the bot and/or consider donating to AI Against Autocracy if they support its mission."
                                )
                                # Post the threshold notification
                                success = fact_check(mention.text, mention.id, context)
                                write_last_tweet_id(mention.id)
                                continue
                            elif prior_replies_to_user > reply_threshold:
                                # Already past threshold - silent skip (notification already sent)
                                print(f"[Mention Threshold] Already notified user {target_author} (count: {prior_replies_to_user}), silently skipping")
                                success = dryruncheck()
                                write_last_tweet_id(mention.id)
                                continue
                            else:
                                print(f"[Mention Threshold] PROCEEDING with reply to user {target_author} ({prior_replies_to_user} < {reply_threshold})")
                        else:
                            prior_replies = count_bot_replies_in_conversation(conv_id, bot_user_id, api_bot_replies)
                            if prior_replies == reply_threshold:
                                # Exactly at threshold - post notification ONCE
                                print(f"[Mention Threshold] THRESHOLD REACHED for thread {conv_id}: {prior_replies} == {reply_threshold}")
                                print(f"[Mention Threshold] Posting threshold notification to thread {conv_id}")
                                # Add threshold notification instructions to context
                                context['context_instructions'] = (
                                    f"\nPrompt: Politely inform the participants that you've reached your reply limit of {reply_threshold} responses for this conversation thread. "
                                    "Thank everyone for the engaging discussion and encourage them to continue exploring the topic independently."
                                )
                                # Post the threshold notification
                                success = fact_check(mention.text, mention.id, context)
                                write_last_tweet_id(mention.id)
                                continue
                            elif prior_replies > reply_threshold:
                                # Already past threshold - silent skip (notification already sent)
                                print(f"[Mention Threshold] Already notified thread {conv_id} (count: {prior_replies}), silently skipping")
                                success = dryruncheck()
                                write_last_tweet_id(mention.id)
                                continue
                        # Pass context to fact_check and reply
                        success = fact_check(mention.text, mention.id, context)
                        if success == 'done!':
                            last_tweet_id = max(last_tweet_id, mention.id)
                            write_last_tweet_id(last_tweet_id)
                            backoff_multiplier = 1
                            time.sleep(30)
                        elif success == 'delay!':
                            backoff_multiplier *= 2
                            print(f'Backoff Multiplier:{backoff_multiplier}')
                            return
                        else:
                            # fact_check returned 'fail' - this means retry failed or dryrun
                            print(f"[Mention Processing] Fact check failed for mention {mention.id}, skipping")
                            write_last_tweet_id(mention.id)
                except Exception as e:
                    print(f"[Mention Processing] Error processing mention {mention.id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Always write the mention ID to avoid restart loops on code errors
                    write_last_tweet_id(mention.id)
                    # Continue to next mention instead of crashing
        else:
            print("No new mentions found.")
            backoff_multiplier = 1
    except tweepy.TweepyException as e:
        print(f"Error fetching mentions: {e}")
        backoff_multiplier += 1
        print(f'Backoff Multiplier:{backoff_multiplier}')

from collections import defaultdict

def tweet_to_dict(t):
    # If already a dict, return as-is
    if isinstance(t, dict):
        return t
    # Use .data if available, else fallback to __dict__
    elif hasattr(t, 'data') and isinstance(t.data, dict):
        return t.data
    elif hasattr(t, '__dict__'):
        # Only keep serializable fields
        return {k: v for k, v in t.__dict__.items() if isinstance(v, (str, int, float, dict, list, type(None)))}
    else:
        return str(t)

def collect_quoted(refs, includes=None):
    quoted_responses = []  # Store full responses for media extraction
    for ref_tweet in refs or []:
        # Handle both dict and object formats
        ref_type = None
        ref_id = None
        
        if isinstance(ref_tweet, dict):
            ref_type = ref_tweet.get('type')
            ref_id = ref_tweet.get('id')
        elif hasattr(ref_tweet, 'type'):
            ref_type = ref_tweet.type
            ref_id = ref_tweet.id
        else:
            continue
        
        if ref_type == "quoted":
            try:
                quoted_response = read_client.get_tweet(
                    id=ref_id,
                    tweet_fields=["text", "author_id", "created_at", "attachments", "entities"],
                    expansions=["attachments.media_keys"],
                    media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"]
                )
                print(f"[DEBUG] Quoted tweet {ref_id} text length: {len(quoted_response.data.text)} chars")
                quoted_responses.append(quoted_response)
            except tweepy.TweepyException as e:
                print(f"Error fetching quoted tweet {ref_id}: {e}")
    return quoted_responses

def get_tweet_context(tweet, includes=None, bot_username=None):
    """Fetch context for a tweet, prioritizing cache before API calls."""
    context = {
        "original_tweet": None,
        "thread_tweets": [],
        "quoted_tweets": [],
        "conversation_id": getattr(tweet, 'conversation_id', None),
        "ancestor_chain": [],
        "bot_replies_in_thread": [],
        "media": [],
        "is_retweet": False  # Default to False
    }

    # Check if this tweet is a retweet
    refs = getattr(tweet, 'referenced_tweets', None) if hasattr(tweet, 'referenced_tweets') else (tweet.get('referenced_tweets') if isinstance(tweet, dict) else None)
    if refs:
        for ref in refs:
            if isinstance(ref, dict):
                rtype = ref.get('type')
            else:
                rtype = getattr(ref, 'type', None)
            if rtype == 'retweeted':
                context['is_retweet'] = True
                break

    conv_id = str(context["conversation_id"])
    cached_data = load_ancestor_chains().get(conv_id)

    # Default: reply target is the tweet itself unless we detect a retweet and override it
    try:
        context['reply_target_id'] = getattr(tweet, 'id', None) if hasattr(tweet, 'id') else (tweet.get('id') if isinstance(tweet, dict) else None)
    except Exception:
        context['reply_target_id'] = None

    # Helper: search includes for a tweet object by id (robust to dict or object shapes)
    def find_in_includes(includes_obj, tid):
        if not includes_obj or not tid:
            return None
        try:
            # dict-like includes (e.g., manual caches or some SDKs)
            if isinstance(includes_obj, dict):
                # common shapes: {'tweets': [...]} or {'media': [...], 'tweets': [...]}
                tweets_list = includes_obj.get('tweets') or includes_obj.get('data') or []
                for tt in tweets_list:
                    if isinstance(tt, dict) and str(tt.get('id')) == str(tid):
                        return tt
                    if hasattr(tt, 'id') and str(getattr(tt, 'id')) == str(tid):
                        return tt
            else:
                # object-like includes (tweepy) may have .data or .tweets attributes
                if hasattr(includes_obj, 'data') and isinstance(includes_obj.data, list):
                    for tt in includes_obj.data:
                        tid_v = getattr(tt, 'id', None) if not isinstance(tt, dict) else tt.get('id')
                        if str(tid_v) == str(tid):
                            return tt
                if hasattr(includes_obj, 'tweets'):
                    for tt in includes_obj.tweets:
                        tid_v = getattr(tt, 'id', None) if not isinstance(tt, dict) else tt.get('id')
                        if str(tid_v) == str(tid):
                            return tt
        except Exception:
            return None
        return None

    if cached_data:
        print(f"[Context Cache] Loaded cached data for conversation {conv_id}")

        # Handle both old format (direct list) and new format (dict with 'chain')
        if isinstance(cached_data, list):
            # Old format - just the chain
            context['ancestor_chain'] = cached_data
        else:
            # New format - dict with 'chain' and other context
            context['ancestor_chain'] = cached_data.get('chain', [])

            # Load thread tweets if cached
            if 'thread_tweets' in cached_data:
                context['thread_tweets'] = [t for t in cached_data['thread_tweets']]
                print(f"[Context Cache] Using cached thread tweets ({len(context['thread_tweets'])})")
            else:
                # Fetch if not cached and args.fetchthread
                if args.fetchthread:
                    try:
                        thread_response = read_client.search_recent_tweets(
                            query=f"conversation_id:{conv_id} -from:{username}",
                            max_results=10,
                            tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                            expansions=["referenced_tweets.id", "attachments.media_keys"],
                            media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"]
                        )
                        if thread_response.data:
                            context["thread_tweets"] = thread_response.data
                            print(f"[API] Fetched {len(context['thread_tweets'])} thread tweets")
                        else:
                            print("[API] No thread tweets found")
                    except tweepy.TweepyException as e:
                        print(f"Error fetching thread tweets: {e}")

            # Load bot replies if cached
            if 'bot_replies' in cached_data:
                context["bot_replies_in_thread"] = [t for t in cached_data['bot_replies']]
                print(f"[Context Cache] Using cached bot replies ({len(context['bot_replies_in_thread'])})")
            else:
                # Fetch bot replies
                try:
                    # Use explicit bot_username when provided to avoid relying on caller globals
                    uname = bot_username if bot_username else (username if 'username' in globals() else None)
                    bot_replies_response = read_client.search_recent_tweets(
                        query=f"conversation_id:{conv_id} from:{uname}",
                        max_results=10,
                        tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                        expansions=["referenced_tweets.id", "attachments.media_keys"],
                        media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"]
                    )
                    if bot_replies_response.data:
                        context["bot_replies_in_thread"] = bot_replies_response.data
                        print(f"[API] Fetched {len(context['bot_replies_in_thread'])} bot replies")
                    else:
                        print("[API] No bot replies found")
                except tweepy.TweepyException as e:
                    print(f"Error fetching bot replies: {e}")

        # Extract quoted tweets and media from cached chain
        # Validate cached media to ensure it was properly attached to tweets
        for entry in context['ancestor_chain']:
            if entry is None:
                continue
            quoted = entry.get('quoted_tweets', []) if isinstance(entry, dict) else []
            cached_media = entry.get('media', []) if isinstance(entry, dict) else []
            
            # Filter cached media: validate against attachments.media_keys
            tweet_obj = entry.get('tweet') if isinstance(entry, dict) else None
            if tweet_obj and len(cached_media) > 0:
                attachments = get_attr(tweet_obj, 'attachments', {})
                if attachments is None:
                    attachments = {}
                media_keys = attachments.get('media_keys', []) if isinstance(attachments, dict) else (getattr(attachments, 'media_keys', []) if hasattr(attachments, 'media_keys') else [])
                
                # Validate each cached media item against media_keys
                validated_media = []
                for m in cached_media:
                    media_key = m.get('media_key') if isinstance(m, dict) else None
                    media_url = m.get('url', '') if isinstance(m, dict) else ''
                    
                    if media_key is None:
                        # No media_key - check if it's a link preview (news_img) or bad cached media
                        if 'news_img' in media_url or not media_url.startswith('https://pbs.twimg.com/media/'):
                            # Link preview from entities.urls or external URL - keep it
                            validated_media.append(m)
                        else:
                            # Likely old cached media from includes without validation - skip it
                            tweet_id = get_attr(tweet_obj, 'id', 'unknown')
                            print(f"[Media Filter] Skipping cached media with no media_key from tweet {tweet_id} (URL: {media_url[:50]}...)")
                    elif media_keys and media_key in media_keys:
                        # Media key matches tweet's attachments - keep it
                        validated_media.append(m)
                    else:
                        # Media has media_key but it's not in tweet's attachments - skip it
                        tweet_id = get_attr(tweet_obj, 'id', 'unknown')
                        print(f"[Media Filter] Skipping cached media {media_key} from tweet {tweet_id} (not in attachments)")
                
                cached_media = validated_media
            
            context['quoted_tweets'].extend([q for q in quoted if q is not None])
            context['media'].extend([m for m in cached_media if m is not None])

        # Attempt to derive original_tweet from chain if not fetching separately
        if context['ancestor_chain']:
            context["original_tweet"] = context['ancestor_chain'][0]['tweet']  # Root is original

        # Validate cached ancestor chain - check if any entries are blank/incomplete
        has_blank_entries = False
        if context['ancestor_chain']:
            for entry in context['ancestor_chain']:
                if not isinstance(entry, dict):
                    continue
                t = entry.get('tweet')
                tweet_id = get_attr(t, 'id')
                if not tweet_id:
                    has_blank_entries = True
                    print(f"[Context Cache] Found blank entry in cached ancestor chain")
                    break

        # Check if cached ancestor chain is complete (walks to root)
        chain_complete = True
        if context['ancestor_chain']:
            # Get the oldest (root) tweet in chain
            root_entry = context['ancestor_chain'][0] if context['ancestor_chain'] else None
            if root_entry and isinstance(root_entry, dict):
                root_tweet = root_entry.get('tweet')
                if root_tweet:
                    # Check if root tweet has a parent (referenced_tweets with replied_to)
                    refs = get_attr(root_tweet, 'referenced_tweets', [])
                    if refs:
                        for ref in refs:
                            ref_type = get_attr(ref, 'type') if not isinstance(ref, dict) else ref.get('type')
                            if ref_type == 'replied_to':
                                # Root tweet has a parent, so chain is incomplete
                                chain_complete = False
                                print(f"[Context Cache] Cached chain incomplete - root tweet has parent")
                                break
        
        # If we have everything from cache AND no blank entries AND chain is complete, return early
        if isinstance(cached_data, dict) and 'thread_tweets' in cached_data and 'bot_replies' in cached_data and context['ancestor_chain'] and not has_blank_entries and chain_complete:
            print("[Context Cache] All context loaded from cache - skipping API calls")
            context['from_cache'] = True  # Mark that this context was loaded from cache
            # Still collect media from mention if not in chain
            # Ensure the current mention is present as the final entry in the ancestor_chain
            try:
                last_entry = context['ancestor_chain'][-1] if context['ancestor_chain'] else None
                # last_entry may be a dict with key 'tweet' (cached format) or a structure with a 'tweet' key
                if isinstance(last_entry, dict):
                    last_tweet_obj = last_entry.get('tweet')
                else:
                    # fallback: assume entry itself is a tweet-like object
                    last_tweet_obj = last_entry
                last_id = get_attr(last_tweet_obj, 'id', None)
            except Exception:
                last_id = None
            mention_id = get_attr(tweet, 'id', None)
            if mention_id and str(last_id) != str(mention_id):
                # Collect media and quoted tweets for the mention and append to the in-memory chain
                mention_media = extract_media(tweet, includes)
                quoted_in_mention = [qr.data for qr in collect_quoted(getattr(tweet, 'referenced_tweets', None))]
                context['ancestor_chain'].append({
                    'tweet': tweet,
                    'quoted_tweets': quoted_in_mention,
                    'media': mention_media
                })
                # Also expose mention media in the top-level context media list
                context['media'].extend(mention_media)
                print(f"[Context Cache] Appended current mention {mention_id} to ancestor_chain (cached path)")
                
                # CRITICAL FIX: Save the updated chain back to the cache file so get_user_reply_counts() can see it
                try:
                    additional_context = {
                        "thread_tweets": cached_data.get('thread_tweets', []) if isinstance(cached_data, dict) else [],
                        "bot_replies": cached_data.get('bot_replies', []) if isinstance(cached_data, dict) else []
                    }
                    save_ancestor_chain(conv_id, context['ancestor_chain'], additional_context)
                    print(f"[Context Cache] Saved updated ancestor chain with new mention to file")
                except Exception as e:
                    print(f"[Context Cache] Error saving updated chain: {e}")

            context['media'].extend(extract_media(tweet, includes))

            # Generate full_thread_text from cached data
            if context["thread_tweets"]:
                original_author_id = get_attr(tweet, 'author_id')
                thread_texts = []
                sorted_tweets = sorted([t for t in context["thread_tweets"] if get_attr(t, 'created_at') is not None],
                                     key=lambda t: get_attr(t, 'created_at'))
                for tt in sorted_tweets:
                    tt_author = get_attr(tt, 'author_id')
                    if str(tt_author) == str(original_author_id):
                        tt_text = get_full_text(tt)
                        thread_texts.append(tt_text)
                context["full_thread_text"] = " ".join(thread_texts)
            else:
                context["full_thread_text"] = get_full_text(tweet)

            return context

    # If not fully cached, build ancestor chain
    # If we have a partial cached chain, continue from where it left off
    if context['ancestor_chain'] and not chain_complete:
        print(f"[Context Cache] Continuing ancestor chain from cached root")
        ancestor_chain = context['ancestor_chain']
        # Start from the root (oldest) tweet in cached chain
        root_entry = ancestor_chain[0]
        current_tweet = root_entry.get('tweet') if isinstance(root_entry, dict) else root_entry
        # Mark all cached tweets as visited
        for entry in ancestor_chain:
            t = entry.get('tweet') if isinstance(entry, dict) else entry
            tid = get_attr(t, 'id')
            if tid:
                visited.add(tid)
    else:
        print(f"[Context Cache] Incomplete cache for {conv_id} - building with API")
        ancestor_chain = []
        current_tweet = tweet
        visited = set()
    
    quoted_from_api = []  # Collect any newly fetched quoted tweets
    # Special handling: if this mention is a retweet of another tweet, attempt to
    # fetch the original full text (prefer twitterapi.io if configured) and
    # ensure we reply to the retweeter's tweet (the incoming tweet id) rather
    # than the original tweet's id. This keeps replies visible in the retweeter's thread.
    try:
        refs = getattr(tweet, 'referenced_tweets', None) if hasattr(tweet, 'referenced_tweets') else (tweet.get('referenced_tweets') if isinstance(tweet, dict) else None)
        if refs:
            for ref in refs:
                rtype = getattr(ref, 'type', None) if not isinstance(ref, dict) else ref.get('type')
                rid = getattr(ref, 'id', None) if not isinstance(ref, dict) else ref.get('id')
                if rtype == 'retweeted' and rid:
                    original_text = ''
                    original_obj = None
                    
                    # First, try to find the referenced tweet in the includes payload
                    inc_obj = find_in_includes(includes, rid)
                    if inc_obj:
                        original_obj = inc_obj
                        try:
                            original_text = get_full_text(inc_obj)
                        except Exception:
                            # fallback to dict/text access
                            if isinstance(inc_obj, dict):
                                original_text = inc_obj.get('text', '')
                    
                    # Only use twitterapi.io if text might be truncated (180-280 chars)
                    if 180 <= len(original_text) <= 280:
                        try:
                            if getattr(args, 'use_twitterapiio', False) and 'TWITTERAPIIO_KEY' in keys and keys.get('TWITTERAPIIO_KEY'):
                                full_text = get_full_text_twitterapiio(rid, keys.get('TWITTERAPIIO_KEY'))
                                if full_text:
                                    original_text = full_text
                                    original_obj = {'id': str(rid), 'text': full_text}
                        except Exception as e:
                            print(f"[Context] twitterapi.io fetch error for {rid}: {e}")

                    # 3) Final fallback: call the official API
                    if not original_text:
                        try:
                            resp = read_client.get_tweet(
                                id=rid,
                                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                                expansions=["referenced_tweets.id", "attachments.media_keys"],
                                media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"]
                            )
                            if resp and getattr(resp, 'data', None):
                                original_obj = resp.data
                                try:
                                    original_text = get_full_text(resp.data)
                                except Exception:
                                    original_text = getattr(resp.data, 'text', '') if hasattr(resp.data, 'text') else ''
                        except Exception as e:
                            print(f"[Context] Error fetching original tweet {rid} via API: {e}")

                    if original_obj:
                        context['original_tweet'] = original_obj
                        context['mention_full_text'] = original_text
                        # Save the original tweet's conversation ID for deduplication
                        orig_conv_id = original_obj.get('conversation_id') if isinstance(original_obj, dict) else getattr(original_obj, 'conversation_id', None)
                        if orig_conv_id:
                            context['original_conversation_id'] = str(orig_conv_id)
                        # Ensure we reply to the retweeter's tweet (the incoming mention)
                        try:
                            context['reply_target_id'] = getattr(tweet, 'id', None) if hasattr(tweet, 'id') else (tweet.get('id') if isinstance(tweet, dict) else None)
                        except Exception:
                            pass
                    # We handled the retweet reference; stop checking further refs
                    break
    except Exception as e:
        print(f"[Context] Error handling retweet fallback: {e}")
    try:
        while True:
            current_text = get_full_text(current_tweet)
            print(f"[Ancestor Build] Processing tweet {current_tweet.id}, text length: {len(current_text)} chars")
            quoted_responses = collect_quoted(getattr(current_tweet, 'referenced_tweets', None))
            quoted = [qr.data for qr in quoted_responses]  # Extract data for storage
            quoted_from_api.extend(quoted)
            # Extract media, passing includes if available
            # For first iteration (current_tweet == tweet), use the includes passed to get_tweet_context
            # For subsequent iterations, use parent_response.includes
            if current_tweet.id == tweet.id:
                current_includes = includes
            else:
                current_includes = parent_response.includes if 'parent_response' in locals() and hasattr(parent_response, 'includes') else None
            media = extract_media(current_tweet, current_includes)
            # Extract media from quoted tweets
            for qr in quoted_responses:
                media.extend(extract_media(qr.data, qr.includes))
            
            # Extract username from includes
            username = None
            if current_includes:
                tweet_author_id = getattr(current_tweet, 'author_id', None)
                print(f"[Username Debug] Looking for author_id: {tweet_author_id}")
                
                # Handle both dict and object formats
                users_list = None
                if isinstance(current_includes, dict):
                    users_list = current_includes.get('users', [])
                    print(f"[Username Debug] current_includes is dict, has {len(users_list)} users")
                elif hasattr(current_includes, 'users'):
                    users_list = current_includes.users
                    print(f"[Username Debug] current_includes is object, has {len(users_list) if users_list else 0} users")
                
                if users_list and tweet_author_id:
                    for user in users_list:
                        user_id = user.id if hasattr(user, 'id') else user.get('id')
                        user_name = user.username if hasattr(user, 'username') else user.get('username')
                        print(f"[Username Debug] Found user: id={user_id}, username={user_name}")
                        if str(user_id) == str(tweet_author_id):
                            username = user_name
                            print(f"[Username Debug] MATCH! Setting username to: {username}")
                            break
            else:
                print(f"[Username Debug] No includes available")
            
            ancestor_chain.append({
                "tweet": current_tweet,
                "quoted_tweets": quoted,
                "media": media,
                "username": username
            })
            visited.add(current_tweet.id)
            parent_id = None
            if hasattr(current_tweet, 'referenced_tweets') and current_tweet.referenced_tweets:
                for ref in current_tweet.referenced_tweets:
                    # Handle both dict and object formats
                    ref_type = ref.get('type') if isinstance(ref, dict) else (ref.type if hasattr(ref, 'type') else None)
                    ref_id = ref.get('id') if isinstance(ref, dict) else (ref.id if hasattr(ref, 'id') else None)
                    
                    if ref_type == 'replied_to':
                        parent_id = ref_id
                        break
            if parent_id is None or parent_id in visited:
                break
            
            # Add parent_id to visited BEFORE making the API call
            # This prevents duplicate API calls if an exception occurs and we retry
            visited.add(parent_id)
            
            try:
                parent_response = read_client.get_tweet(
                    id=parent_id,
                    tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                    expansions=["referenced_tweets.id", "attachments.media_keys", "author_id"],
                    media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"],
                    user_fields=["username"]
                )
                if parent_response.data:
                    current_tweet = parent_response.data
                    print(f"[API Debug] Fetched parent tweet {parent_id}, text length: {len(current_tweet.text)} chars")
                    print(f"[API Debug] First 100 chars: {current_tweet.text[:100]}")
                    print(f"[API Debug] Last 100 chars: {current_tweet.text[-100:]}")
                else:
                    break
            except tweepy.TooManyRequests as e:
                # Rate limit hit - try twitterapi.io as fallback
                print(f"[Ancestor Build] Rate limit hit, falling back to twitterapi.io for tweet {parent_id}")
                api_key = keys.get('TWITTERAPIIO_KEY')
                if api_key:
                    tweet_dict = get_tweet_data_twitterapiio(parent_id, api_key)
                    if tweet_dict:
                        # Convert dict to a simple object with attributes we can access
                        class TweetObj:
                            def __init__(self, data):
                                for key, value in data.items():
                                    setattr(self, key, value)
                        current_tweet = TweetObj(tweet_dict)
                        print(f"[twitterapi.io Fallback] Successfully fetched parent tweet {parent_id}")
                    else:
                        print(f"[twitterapi.io Fallback] Failed to fetch tweet {parent_id}, stopping chain build")
                        break
                else:
                    print(f"[Ancestor Build] No TWITTERAPIIO_KEY available, cannot continue chain build")
                    break
            except tweepy.TweepyException as te:
                print(f"[Ancestor Build] API error fetching parent {parent_id}: {te}")
                break
    except tweepy.TweepyException as e:
        print(f"Error building ancestor chain: {e}")

    ancestor_chain = ancestor_chain[::-1]  # Root first
    context['ancestor_chain'] = ancestor_chain

    # NEW: Ensure the current mention is appended as the final entry with its media
    if ancestor_chain and ancestor_chain[-1]['tweet'].id != tweet.id:
        mention_media = extract_media(tweet, includes)
        
        # Extract username from includes for the mention
        mention_username = None
        if includes:
            tweet_author_id = getattr(tweet, 'author_id', None)
            print(f"[Mention Username Debug] Looking for author_id: {tweet_author_id}")
            
            # Handle both dict and object formats
            users_list = None
            if isinstance(includes, dict):
                users_list = includes.get('users', [])
                print(f"[Mention Username Debug] includes is dict, has {len(users_list)} users")
            elif hasattr(includes, 'users'):
                users_list = includes.users
                print(f"[Mention Username Debug] includes is object, has {len(users_list) if users_list else 0} users")
            
            if users_list and tweet_author_id:
                for user in users_list:
                    user_id = user.id if hasattr(user, 'id') else user.get('id')
                    user_name = user.username if hasattr(user, 'username') else user.get('username')
                    print(f"[Mention Username Debug] Found user: id={user_id}, username={user_name}")
                    if str(user_id) == str(tweet_author_id):
                        mention_username = user_name
                        print(f"[Mention Username Debug] MATCH! Setting mention_username to: {mention_username}")
                        break
        else:
            print(f"[Mention Username Debug] No includes available")
        
        ancestor_chain.append({
            "tweet": tweet,
            "quoted_tweets": [qr.data for qr in collect_quoted(getattr(tweet, 'referenced_tweets', None))],  # Fetch any quoted in mention
            "media": mention_media,
            "username": mention_username
        })

    # Fetch thread if not cached and enabled
    if args.fetchthread and not context["thread_tweets"]:
        # Fetch as above (code duplicated for clarity, but could refactor)
        try:
            thread_response = read_client.search_recent_tweets(
                query=f"conversation_id:{conv_id} -from:{username}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                expansions=["referenced_tweets.id", "attachments.media_keys"],
                media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"]
            )
            if thread_response.data:
                context["thread_tweets"] = thread_response.data
        except tweepy.TweepyException as e:
            print(f"Error fetching thread tweets: {e}")

    # Fetch bot replies if not cached
    if not context["bot_replies_in_thread"]:
        try:
            uname = bot_username if bot_username else (username if 'username' in globals() else None)
            bot_replies_response = read_client.search_recent_tweets(
                query=f"conversation_id:{conv_id} from:{uname}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                expansions=["referenced_tweets.id", "attachments.media_keys"],
                media_fields=["media_key", "type", "url", "preview_image_url", "alt_text"]
                )
            if bot_replies_response.data:
                context["bot_replies_in_thread"] = bot_replies_response.data
        except tweepy.TweepyException as e:
            print(f"Error fetching bot replies: {e}")

    # Collect quoted tweets (from chain and any additional)
    for entry in context['ancestor_chain']:
        if entry is None:
            continue
        quoted = entry.get('quoted_tweets', []) if isinstance(entry, dict) else []
        context['quoted_tweets'].extend([q for q in quoted if q is not None])
    context['quoted_tweets'].extend([q for q in quoted_from_api if q is not None])  # Any extras

    # Set original tweet if available
    if context['ancestor_chain']:
        first_entry = context['ancestor_chain'][0]
        if first_entry and isinstance(first_entry, dict):
            context["original_tweet"] = first_entry.get('tweet')

    # Collect media
    context['media'].extend([m for m in extract_media(tweet, includes) if m is not None])
    for entry in context['ancestor_chain']:
        if entry is None:
            continue
        media = entry.get('media', []) if isinstance(entry, dict) else []
        context['media'].extend([m for m in media if m is not None])
    
    # Deduplicate media by URL (keep first occurrence)
    seen_urls = set()
    deduped_media = []
    for m in context['media']:
        if m and isinstance(m, dict) and m.get('url'):
            url = m['url']
            if url not in seen_urls:
                seen_urls.add(url)
                deduped_media.append(m)
    context['media'] = deduped_media
    print(f"[Media Debug] After deduplication: {len(context['media'])} unique media items")

    # Save updated cache with additional context
    additional_context = {
        "thread_tweets": [tweet_to_dict(t) for t in context["thread_tweets"]],
        "bot_replies": [tweet_to_dict(t) for t in context["bot_replies_in_thread"]]
    }
    save_ancestor_chain(conv_id, ancestor_chain, additional_context)

    # New: Concatenate full thread text (only from original author to avoid noise)
    if context["thread_tweets"]:
        original_author_id = get_attr(tweet, 'author_id')
        thread_texts = []
        sorted_tweets = sorted(context["thread_tweets"], key=lambda t: get_attr(t, 'created_at'))
        for tt in sorted_tweets:
            if str(get_attr(tt, 'author_id')) == str(original_author_id):
                thread_texts.append(get_full_text(tt))
        context["full_thread_text"] = " ".join(thread_texts)
    else:
        context["full_thread_text"] = get_full_text(tweet)

    return context

def get_full_text(t):
    # Return the full text directly from the 'text' field (API v2 standard)
    if hasattr(t, 'text'):
        return t.text
    elif isinstance(t, dict) and 'text' in t:
        return t['text']
    return ''  # Fallback for invalid/missing tweet

def get_attr(obj, attr, default=None):
    """Safely get attribute from either dict or object"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    else:
        return getattr(obj, attr, default)

def build_ancestor_chain(ancestor_chain, indent=0, from_cache=False, verbose=True):
    out = ""

    for i, entry in enumerate(ancestor_chain):
        if entry is None or not isinstance(entry, dict):
            continue
        t = entry.get("tweet")
        quoted_tweets = entry.get("quoted_tweets", [])
        tweet_id = get_attr(t, "id")
        author_id = get_attr(t, "author_id", "")
        is_bot_tweet = str(author_id) == str(getid())
        
        # Only check bot_tweets.json if this is actually a bot tweet
        tweet_text = None
        if is_bot_tweet and tweet_id:
            if verbose:
                print(f"[Tweet Storage] Found bot tweet {tweet_id} in ancestor chain")
            tweet_text = get_bot_tweet_content(tweet_id, verbose=verbose)
            if tweet_text:
                if verbose:
                    print(f"[Tweet Storage] Using stored content for tweet {tweet_id}")
            else:
                if verbose:
                    print(f"[Tweet Storage] No stored content for bot tweet {tweet_id}, fetching from API")
        
        # If not a bot tweet or no stored content, fetch normally
        if not tweet_text:
            # Get existing text first
            if hasattr(t, 'text'):
                tweet_text = t.text
            elif isinstance(t, dict) and 'text' in t:
                tweet_text = t['text']
            else:
                tweet_text = ''
            
            # Only try twitterapi.io if text might be truncated AND not from cache
            # (cached data already has full text from when it was originally fetched)
            if not from_cache and 180 <= len(tweet_text) <= 280:
                api_key = keys['TWITTERAPIIO_KEY']
                try:
                    full_text = get_full_text_twitterapiio(tweet_id, api_key)
                    if full_text:
                        tweet_text = full_text
                except Exception as e:
                    if verbose:
                        print(f"[Ancestor Chain] Error fetching full text from twitterapi.io for {tweet_id}: {e}")
        if verbose:
            print(f"[Ancestor Chain] Tweet {tweet_id} text length in build: {len(tweet_text)} chars")
        
        # Use username from entry if available, otherwise fall back to author_id
        username = entry.get("username")
        display_name = username if username else author_id
        author = f" (from @{display_name})" if display_name else ""
        out += "  " * indent + f"- {tweet_text}{author}\n"
        # Show quoted tweets indented under their parent
        for qt in quoted_tweets:
            qt_id = get_attr(qt, 'id')
            qt_author_id = qt.get('author_id') if isinstance(qt, dict) else getattr(qt, 'author_id', '')
            qt_author = f" (quoted @{qt_author_id})" if qt_author_id else ""
            
            # Only check bot_tweets.json if this quoted tweet is from the bot
            qt_text = None
            is_bot_qt = str(qt_author_id) == str(getid())
            if is_bot_qt and qt_id:
                qt_text = get_bot_tweet_content(qt_id, verbose=verbose)
                if qt_text and verbose:
                    print(f"[Tweet Storage] Using stored content for bot quoted tweet {qt_id}")
            
            # If not a bot tweet or no stored content, fetch normally
            if not qt_text:
                # Get existing text first
                if hasattr(qt, 'text'):
                    qt_text = qt.text
                elif isinstance(qt, dict) and 'text' in qt:
                    qt_text = qt['text']
                else:
                    qt_text = ''
                
                # Only try twitterapi.io if text might be truncated AND not from cache
                if not from_cache and 180 <= len(qt_text) <= 280:
                    api_key = keys.get('TWITTERAPIIO_KEY')
                    if api_key:
                        try:
                            full_text = get_full_text_twitterapiio(qt_id, api_key)
                            if full_text:
                                qt_text = full_text
                        except Exception as e:
                            if verbose:
                                print(f"[Ancestor Chain] Error fetching quoted tweet {qt_id} from twitterapi.io: {e}")
            out += "  " * (indent + 1) + f"> {qt_text}{qt_author}\n"
        indent += 1
    return out

def extract_media(t, includes=None):
    media_list = []
    found_media = False

    # Added debug logging for includes
    print(f"[Media Debug] Extracting media for tweet {get_attr(t, 'id')} - includes provided: {includes is not None}")
    if includes and isinstance(includes, dict):
        print(f"[Media Debug] Includes keys: {list(includes.keys())}")
        if 'media' in includes:
            print(f"[Media Debug] Found 'media' in includes with {len(includes['media'])} items")
    elif includes:
        print(f"[Media Debug] Includes provided but not a dict (type: {type(includes)})")

    # FIRST: Check includes for media (most reliable source from API responses)
    if includes and isinstance(includes, dict) and 'media' in includes:
        # Get the tweet's media_keys if they exist
        attachments = get_attr(t, 'attachments', {})
        if attachments is None:
            attachments = {}
        media_keys = attachments.get('media_keys', []) if isinstance(attachments, dict) else (getattr(attachments, 'media_keys', []) if hasattr(attachments, 'media_keys') else [])
        
        print(f"[Media Debug] Tweet has {len(media_keys)} media_keys in attachments: {media_keys}")
        
        for m in includes['media']:
            if m is None:
                continue
            # Get media_key from the media object
            media_key = getattr(m, 'media_key', None) if hasattr(m, 'media_key') else (m.get('media_key') if isinstance(m, dict) else None)
            
            # Include media if:
            # 1. It's in the tweet's attachments (directly attached photo/video), OR
            # 2. Tweet has no media_keys (old format/backward compatibility), OR
            # 3. We'll check later if it's from a URL entity (link preview)
            if media_keys and media_key and media_key in media_keys:
                media_list.append({
                    'type': getattr(m, 'type', '') if hasattr(m, 'type') else (m.get('type', '') if isinstance(m, dict) else ''),
                    'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')) if hasattr(m, 'url') else (m.get('url', m.get('preview_image_url', '')) if isinstance(m, dict) else ''),
                    'alt_text': getattr(m, 'alt_text', '') if hasattr(m, 'alt_text') else (m.get('alt_text', '') if isinstance(m, dict) else ''),
                    'media_key': media_key  # Store media_key for validation when loading from cache
                })
                found_media = True
                print(f"[Media Debug] Including media {media_key} (matches attachment)")
            else:
                # Media is in includes but not in tweet's attachments - likely a preview/card image
                print(f"[Media Debug] Skipping media {media_key} (not in tweet attachments, no media_keys found)")
        print(f"[Media Debug] Extracted {len(media_list)} media items from includes['media']")

    # Handle dicts with 'media' key
    if isinstance(t, dict) and 'media' in t:
        for m in t['media']:
            if m is None:
                continue
            if isinstance(m, dict):
                media_list.append({
                    'type': m.get('type'),
                    'url': m.get('url', m.get('preview_image_url', '')),
                    'alt_text': m.get('alt_text', '')
                })
                found_media = True

    # Handle Tweepy tweet objects with attachments/media
    #elif hasattr(t, 'attachments') and hasattr(t.attachments, 'media_keys'):
    # Already checked includes above, so skip it here
    elif hasattr(t, 'includes') and 'media' in t.includes:
        for m in t.includes['media']:
            if m is None:
                continue
            media_list.append({
                'type': getattr(m, 'type', ''),
                'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')),
                'alt_text': getattr(m, 'alt_text', '')
            })
            found_media = True

    # Optionally, handle Tweepy objects with direct 'media' attribute
    elif hasattr(t, 'media') and isinstance(t.media, list):
        for m in t.media:
            if m is None:
                continue
            media_list.append({
                'type': getattr(m, 'type', ''),
                'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')),
                'alt_text': getattr(m, 'alt_text', '')
            })
            found_media = True

    # Updated: Check for image URLs in entities.urls with enhanced debug logging and Twitter media pattern detection
    entities = get_attr(t, 'entities', {})
    if entities is None:
        entities = {}
    urls = entities.get('urls', [])
    print(f"[Media Debug] Found {len(urls)} URLs in entities for tweet {get_attr(t, 'id')}")
    for url in urls:
        expanded_url = url.get('expanded_url', '')
        print(f"[Media Debug] Expanded URL: {expanded_url}")
        images_url = url.get('images', []) if 'images' in url else []  # For v2 entities with previews
        if images_url:
            for img in images_url:
                media_list.append({
                    'type': 'photo',
                    'url': img.get('url', expanded_url),
                    'alt_text': ''
                })
                found_media = True
        # Check for Twitter media patterns and common image extensions
        elif 'pbs.twimg.com/media/' in expanded_url or expanded_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            media_list.append({
                'type': 'photo',
                'url': expanded_url,
                'alt_text': ''
            })
            found_media = True

    if found_media:
        print(f"[Media Debug] Found {len(media_list)} media items for tweet {get_attr(t, 'id')} (including linked images)")
    else:
        print(f"[Media Debug] No media found for tweet {get_attr(t, 'id')}")

    return media_list

def extract_urls_from_tweet(tweet):
    """Extract non-media URLs from a tweet's entities.
    Returns list of dicts with 'url', 'expanded_url', 'display_url', 'title'.
    Filters out Twitter links (t.co redirects to twitter.com posts).
    """
    urls = []
    entities = get_attr(tweet, 'entities', {})
    if entities is None:
        entities = {}
    
    url_entities = entities.get('urls', [])
    for url_obj in url_entities:
        expanded_url = url_obj.get('expanded_url', '')
        display_url = url_obj.get('display_url', '')
        unwound_url = url_obj.get('unwound_url', expanded_url)  # v2 API provides unwound URL
        title = url_obj.get('title', '')  # v2 API sometimes provides title
        description = url_obj.get('description', '')
        
        # Skip Twitter/X links (quotes, replies, etc.)
        if any(domain in expanded_url.lower() for domain in ['twitter.com/', 'x.com/']):
            continue
        
        # Skip media URLs (already handled by format_media)
        if 'pbs.twimg.com/media/' in expanded_url or expanded_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            continue
        
        urls.append({
            'url': url_obj.get('url', ''),  # Shortened t.co link
            'expanded_url': expanded_url,
            'unwound_url': unwound_url,
            'display_url': display_url,
            'title': title,
            'description': description
        })
    
    return urls


def format_urls(ancestor_chain=None):
    """Format URLs from tweets in the conversation thread.
    Similar to format_media but for web links (articles, etc.).
    """
    if not ancestor_chain:
        return ""
    
    all_urls = []
    seen_urls = set()
    
    # Collect URLs from each tweet in the chain
    for entry in ancestor_chain:
        tweet = entry.get("tweet") if isinstance(entry, dict) else entry
        if not tweet:
            continue
        
        tweet_id = get_attr(tweet, "id", "unknown")
        tweet_snippet = get_full_text(tweet)[:50] + "..."
        urls = extract_urls_from_tweet(tweet)
        
        # Deduplicate by expanded URL
        for url_obj in urls:
            expanded = url_obj.get('expanded_url', '')
            if expanded and expanded not in seen_urls:
                seen_urls.add(expanded)
                all_urls.append({
                    'tweet_id': tweet_id,
                    'tweet_snippet': tweet_snippet,
                    **url_obj
                })
        
        # Also check quoted tweets
        for qt in entry.get("quoted_tweets", []):
            qt_urls = extract_urls_from_tweet(qt)
            qt_id = get_attr(qt, "id", "unknown")
            qt_snippet = get_full_text(qt)[:50] + "..."
            
            for url_obj in qt_urls:
                expanded = url_obj.get('expanded_url', '')
                if expanded and expanded not in seen_urls:
                    seen_urls.add(expanded)
                    all_urls.append({
                        'tweet_id': qt_id,
                        'tweet_snippet': qt_snippet,
                        'is_quoted': True,
                        **url_obj
                    })
    
    if not all_urls:
        return ""
    
    out = "\nURLs/Links shared in the conversation:\n"
    for url_info in all_urls:
        quoted_prefix = "[Quoted] " if url_info.get('is_quoted') else ""
        out += f"- {quoted_prefix}From tweet {url_info['tweet_id']} ('{url_info['tweet_snippet']}')\n"
        out += f"  URL: {url_info['unwound_url']}\n"
        if url_info.get('title'):
            out += f"  Title: {url_info['title']}\n"
        if url_info.get('description'):
            out += f"  Description: {url_info['description'][:100]}...\n"
        out += f"  Display: {url_info['display_url']}\n"
    
    return out


def format_media(media_list, ancestor_chain=None):
    if not media_list:
        return ""
    out = "Media attached to tweet(s) in the thread (with associations):\n"

    # If chain provided, group media by tweet for better context
    if ancestor_chain:
        for entry in ancestor_chain:
            tweet_id = get_attr(entry["tweet"], "id", "unknown")
            tweet_snippet = get_full_text(entry["tweet"])[:50] + "..."  # Brief text for context
            entry_media = entry.get("media", [])
            if entry_media:
                out += f"- From tweet {tweet_id} ('{tweet_snippet}'): \n"
                for m in entry_media:
                    out += f"  - Type: {m.get('type', '')}, URL: {m.get('url', '')}, Alt: {m.get('alt_text', '')}\n"

            # Handle quoted tweets in this entry
            for qt in entry.get("quoted_tweets", []):
                qt_id = get_attr(qt, "id", "unknown")
                qt_snippet = get_full_text(qt)[:50] + "..."
                # Assume quoted media is already in entry["media"] (from your recent edit), or extract if needed
                out += f"  - Quoted in {tweet_id}: From tweet {qt_id} ('{qt_snippet}'): (media URLs as above)\n"
    else:
        # Fallback to flat list
        for m in media_list:
            if isinstance(m, dict):
                out += f"- Type: {m.get('type', '')}, URL: {m.get('url', '')}, Alt: {m.get('alt_text', '')}\n"
            else:
                out += f"- Media key: {m}\n"
    return out


import time
import datetime
import tweepy
import os

import tweepy
#import requests
import time
import threading
from openai import OpenAI
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='AutoGrok AI Twitter fact-checking bot')
parser.add_argument('--username', type=str, help='X username of the bot', default='ConSenseAI')
parser.add_argument('--delay', type=float, help='Delay between checks in minutes (e.g., 2)')
parser.add_argument('--dryrun', type=lambda x: x.lower() in ['true', '1', 'yes'], help='Print responses but don\'t tweet them', default=False)
#parser.add_argument('--accuracy', type=int, help="Accuracy score threshold out of 10. Don't reply to tweets scored above this threshold")
parser.add_argument('--fetchthread', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, Try to fetch the rest of the thread for additional context. Warning: API request hungry', default=True)
parser.add_argument('--reply_threshold', type=int, help='Number of times the bot can reply in a thread before skipping further replies (default 5)', default=5)
parser.add_argument('--per_user_threshold', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, enforce reply_threshold per unique user per thread; if False, enforce per-thread total (default True)', default=True)
parser.add_argument('--search_term', type=str, help='If provided, periodically search this term and run the pipeline on matching tweets. Use "auto" to automatically generate relevant search terms after each reflection cycle.', default=None)
parser.add_argument('--search_max_results', type=int, help='Max results to fetch per search (default 10)', default=10)
parser.add_argument('--search_daily_cap', type=int, help='Max automated replies per day from searches increases every "--search_cap_interval_hours" hours(default 5)', default=5)
parser.add_argument('--dedupe_window_hours', type=float, help='Window to consider duplicates (hours, default 24)', default=24.0)
parser.add_argument('--enable_human_approval', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, queue candidate replies for human approval instead of auto-posting', default=False)
parser.add_argument('--search_cap_interval_hours', type=int, help='Number of hours between each increase in search reply cap (default 1)', default=2)
parser.add_argument('--cap_increase_time', type=str, help='Earliest time of day (HH:MM, 24h) to allow cap increases (default 10:00)', default='10:00')
parser.add_argument('--post_interval', type=int, help='Number of bot replies between posting a reflection based on recent threads (default 10)', default=10)
parser.add_argument('--follow_threshold', type=int, help='Minimum number of replies from a user before auto-following them (default 2)', default=2)
parser.add_argument('--check_followed_users', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, periodically check and reply to tweets from followed users (default False)', default=False)
parser.add_argument('--followed_users_max_tweets', type=int, help='Max tweets to check per followed user per cycle (default 5)', default=5)
parser.add_argument('--followed_users_daily_cap', type=int, help='Max automated replies per day from followed users (default 10)', default=10)
parser.add_argument('--followed_users_per_cycle', type=int, help='Max followed users to check per cycle for rotation (default 3)', default=3)
parser.add_argument('--check_community_notes', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, periodically fetch and propose Community Notes (default False)', default=False)
parser.add_argument('--cn_max_results', type=int, help='Max Community Notes eligible posts to process per cycle (default 5)', default=5)
parser.add_argument('--cn_test_mode', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, submit Community Notes in test mode (recommended for development, default True)', default=True)
parser.add_argument('--cn_on_reflection', type=lambda x: x.lower() in ['true', '1', 'yes'], help='If True, run Community Notes check only on reflection cycle instead of main loop (default True)', default=True)
args = parser.parse_args()  # Will error on unrecognized arguments

# Set username and delay, prompting if not provided
if args.username:
    username = args.username.lower()
if args.delay:
    delay = int(args.delay)  # Convert minutes to seconds
else:
    delay = int(float(input('Delay in minutes between checks: ')))
    
if args.dryrun:
    dryrun=args.dryrun

#if args.accuracy:
#    accuracy_threshold = args.accuracy
#else:
#    accuracy_threshold = 4

if args.reply_threshold:
    reply_threshold = args.reply_threshold
else:
    reply_threshold = 5

# Determine behavior mode
per_user_threshold = bool(args.per_user_threshold)


# File to store the last processed tweet ID
LAST_TWEET_FILE = f'last_tweet_id_{username}.txt'

# Files and prefixes for search feature
SEARCH_LAST_FILE_PREFIX = f'last_search_id_{username}_'
SEARCH_REPLY_COUNT_FILE = f'search_reply_count_{username}.json'
SENT_HASHES_FILE = f'sent_reply_hashes_{username}.json'
APPROVAL_QUEUE_FILE = f'approval_queue_{username}.json'
FOLLOWED_USERS_LAST_CHECK_FILE = f'last_checked_followed_{username}.json'
FOLLOWED_USERS_REPLY_COUNT_FILE = f'followed_reply_count_{username}.json'
FOLLOWED_USERS_ROTATION_FILE = f'followed_rotation_{username}.json'

RESTART_DELAY = 10
backoff_multiplier = 1

# Community Notes files and configuration
COMMUNITY_NOTES_LAST_CHECK_FILE = f'cn_last_check_{username}.txt'
COMMUNITY_NOTES_WRITTEN_FILE = f'cn_written_{username}.json'

def fetch_and_process_community_notes(user_id=None, max_results=5, test_mode=True):
    """
    Fetch posts eligible for Community Notes and propose notes using ConSenseAI's fact-checking pipeline.
    
    Args:
        user_id: Bot user ID (for checking if we've already written notes)
        max_results: Maximum number of posts to fetch (default 5)
        test_mode: If True, submit notes in test mode (default True, recommended for development)
    
    Returns:
        int: Number of notes written
    """
    global backoff_multiplier
    
    # Initialize log file for this run
    log_filename = f"cn_notes_log_{username}.txt"
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log_to_file(message):
        """Append message to CN log file with timestamp"""
        try:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception as e:
            print(f"[Community Notes] Error writing to log file: {e}")
    
    log_to_file("=" * 80)
    log_to_file(f"COMMUNITY NOTES RUN STARTED (test_mode={test_mode}, max_results={max_results})")
    log_to_file("=" * 80)
    
    print(f"[Community Notes] Fetching up to {max_results} posts eligible for notes (test_mode={test_mode})")
    print(f"[Community Notes] Logging to {log_filename}")
    
    try:
        # Use Twitter API v2 to fetch posts eligible for Community Notes
        # This endpoint supports OAuth 1.0a User Context (existing bot tokens)
        from requests_oauthlib import OAuth1Session
        import requests
        
        # Use separate keys for Community Notes project (must be from a project with CN API access)
        # Check if CN-specific keys exist, otherwise fall back to main keys
        cn_api_key = keys.get('CN_XAPI_key') or keys.get('XAPI_key')
        cn_api_secret = keys.get('CN_XAPI_secret') or keys.get('XAPI_secret')
        cn_access_token = keys.get('CN_access_token') or keys.get('access_token')
        cn_access_secret = keys.get('CN_access_token_secret') or keys.get('access_token_secret')
        
        if not keys.get('CN_XAPI_key'):
            print("[Community Notes] WARNING: Using main project keys. For Community Notes to work, you need keys from a project with Community Notes API access.")
            print("[Community Notes] Add to keys.txt: CN_XAPI_key, CN_XAPI_secret, CN_access_token, CN_access_token_secret")
        
        print("[Community Notes] Using OAuth 1.0a authentication")
        oauth = OAuth1Session(
            client_key=cn_api_key,
            client_secret=cn_api_secret,
            resource_owner_key=cn_access_token,
            resource_owner_secret=cn_access_secret
        )
        
        params = {
            "test_mode": str(test_mode).lower(),
            "max_results": max_results,
            "tweet.fields": "author_id,created_at,referenced_tweets,attachments,entities,conversation_id",
            "expansions": "attachments.media_keys,referenced_tweets.id,referenced_tweets.id.attachments.media_keys",
            "media.fields": "media_key,type,url,preview_image_url,alt_text,height,width"
        }
        
        response = oauth.get(
            "https://api.twitter.com/2/notes/search/posts_eligible_for_notes",
            params=params
        )
        
        if response.status_code == 200:
            resp = response.json()
            if not resp or not resp.get('data'):
                print("[Community Notes] No eligible posts found")
                log_to_file("No eligible posts found")
                return 0
            else:
                log_to_file(f"Fetched {len(resp['data'])} eligible posts")
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            if response.status_code == 403:
                print(f"[Community Notes] Access denied - account may not have Community Notes API access enabled: {error_msg}")
            elif response.status_code == 404:
                print(f"[Community Notes] Endpoint not found - Community Notes API may not be available: {error_msg}")
            else:
                print(f"[Community Notes] API error: {error_msg}")
                backoff_multiplier += 1
            return 0
            
    except tweepy.TweepyException as e:
        error_str = str(e)
        if "403" in error_str or "Forbidden" in error_str:
            print(f"[Community Notes] Access denied - account may not have Community Notes API access enabled: {e}")
        elif "404" in error_str:
            print(f"[Community Notes] Endpoint not found - Community Notes API may not be available: {e}")
        else:
            print(f"[Community Notes] API error: {e}")
            backoff_multiplier += 1
        return 0
    except Exception as e:
        print(f"[Community Notes] Unexpected error: {e}")
        return 0
    
    # Load previously written notes to avoid duplicates
    written_notes = {}
    if os.path.exists(COMMUNITY_NOTES_WRITTEN_FILE):
        try:
            with open(COMMUNITY_NOTES_WRITTEN_FILE, 'r') as f:
                written_notes = json.load(f)
        except Exception as e:
            print(f"[Community Notes] Error loading written notes file: {e}")
    
    notes_written = 0
    posts = resp.get('data', [])
    includes = resp.get('includes', {})
    
    print(f"[Community Notes] Found {len(posts)} eligible posts")
    log_to_file(f"Found {len(posts)} eligible posts to process")
    
    for post_data in posts:
        post_id = str(post_data.get('id'))
        
        
        log_to_file("\n" + "-" * 80)
        log_to_file(f"POST ID: {post_id}")
        # Skip if we've already written a note for this post
        if post_id in written_notes:
            print(f"[Community Notes] Skipping {post_id} - already written note")
            log_to_file("STATUS: SKIPPED (already wrote note)")
            continue

        log_to_file(f"POST TEXT: {post_data.get('text', '')}")
        log_to_file(f"AUTHOR ID: {post_data.get('author_id')}")
        log_to_file(f"CREATED AT: {post_data.get('created_at')}")
    
        
        # Convert post dict to Tweepy-like object for compatibility with existing functions
        # Create a mock tweet object
        class MockTweet:
            def __init__(self, data):
                self.id = data.get('id')
                self.text = data.get('text', '')
                self.author_id = data.get('author_id')
                self.created_at = data.get('created_at')
                self.conversation_id = data.get('conversation_id')
                self.referenced_tweets = data.get('referenced_tweets', [])
                self.attachments = data.get('attachments', {})
                self.entities = data.get('entities', {})
        
        post = MockTweet(post_data)
        
        # Get full context for the post
        try:
            log_to_file("STATUS: Processing")
            context = get_tweet_context(post, includes, bot_username=username)
            log_to_file(f"CONTEXT: Gathered {len(context.get('ancestor_chain', []))} ancestor tweets, {len(context.get('media', []))} media items")
            
            context['mention'] = post
            context['context_instructions'] = "\nThis post has been flagged as potentially needing a Community Note. Analyze it for misleading claims and create a draft community note\n\
                - CRITICAL URL REQUIREMENTS: Provide ONLY direct, specific source URLs (e.g., https://nytimes.com/2025/12/specific-article-title, NOT generic pages like https://nytimes.com/search). URLs must link directly to the exact article, study, or data that supports your fact-check. Do NOT use search pages, photo galleries, media indexes, or landing pages. Each URL must be a complete, working link to specific source material.\n\
                - CRITICAL: The text of your note must be less than 280 characters (source links only count as one character). Be extremely concise *PARTICULARLY IF YOU ARE IN THE FINAL PASS*\n\
                - Remain anonymous: Do not say who you are. Do not mention the models. Do not talk about consensus of the models \n\
                - Search for information on drafting successful community notes if needed.\n\
                - If the post is not misleading, respond with 'NO NOTE NEEDED', 'NOT MISLEADING' or 'NO COMMUNITY NOTE. A parser will reject the note. Do *NOT* use any of these terms if a not is needed, even when discussing what other models say.\n\
                - If a note is needed, provide *only* the text of the note - no labels, titles or thinking outloud.\n\
                - Only provide a note for the tweet in question. Do not fact check the thread."  
            
            post_text = post.text
            
            print(f"[Community Notes] Analyzing post {post_id}: {post_text[:100]}...")
            log_to_file("GENERATING NOTE: Using ConSenseAI fact-checking pipeline")
            
            # Use fact_check in generate_only mode to get the note text
            note_text = fact_check(post_text, post_id, context=context, generate_only=True)
            
            if not isinstance(note_text, str) or not note_text:
                print(f"[Community Notes] No note generated for {post_id}")
                log_to_file("NOTE GENERATION: Failed (empty response)")
                continue
            
            log_to_file(f"NOTE GENERATED ({len(note_text)} chars):")
            log_to_file(note_text)
            
            # Check if the response indicates no note is needed
            if any(phrase in note_text.upper() for phrase in ["NO NOTE NEEDED", "NOT MISLEADING", "NO COMMUNITY NOTE"]):
                print(f"[Community Notes] Post {post_id} determined not to need a note")
                log_to_file("DECISION: Post does not need a community note")
                written_notes[post_id] = {
                    "note": None,
                    "reason": "not_misleading",
                    "timestamp": datetime.datetime.now().isoformat()
                }
                continue
            
            # Strip model attribution tags from the note text before submission
            # Remove "Generated by: ..." and "Combined by: ..." lines
            # Store original note with attribution for retry context
            original_note_with_attribution = note_text
            clean_note_text = note_text
            combining_model_name = None
            lines = note_text.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines that contain model attribution but extract combining model
                if line.strip().startswith('Combined by:'):
                    # Extract the model name (format: "Combined by: model-name")
                    parts = line.strip().split('Combined by:')
                    if len(parts) > 1:
                        combining_model_name = parts[1].strip().split()[0]  # Get first word (model name)
                    continue
                if line.strip().startswith('Generated by:'):
                    continue
                filtered_lines.append(line)
            clean_note_text = '\n'.join(filtered_lines).strip()
            
            # Log both versions for debugging
            if clean_note_text != note_text:
                log_to_file(f"CLEANED NOTE ({len(clean_note_text)} chars, removed attribution):")
                log_to_file(clean_note_text)
                if combining_model_name:
                    log_to_file(f"COMBINING MODEL: {combining_model_name}")
                #log_to_file("")
            
            # Validate note meets Community Notes requirements
            # Requirements: 1-280 chars effective length, valid URLs, appropriate tone, addresses claims
            import re
            import requests
            import html
            
            # Use Twitter's URL extraction pattern for accurate effective length calculation
            url_pattern = re.compile(
                r"""
                (?:
                    https?://               # optional scheme
                )?
                (?:www\.)?                  # optional www
                [\w\-._~%]+                 # subdomain or domain name chars
                \.[a-zA-Z]{2,}              # dot + top level domain (≥2 letters)
                (?:/[^\s]*)?                # optional query/fragment
                """,
                re.VERBOSE,
            )
            urls_in_note = url_pattern.findall(clean_note_text)
            
            # Calculate effective length (URLs count as 1 char each for Community Notes)
            text_without_urls = url_pattern.sub('', clean_note_text)
            effective_length = len(text_without_urls) + len(urls_in_note)
            
            # Validation functions
            def validate_url_validity(note_text):
                """Check if all URLs return HTTP 200 using Twitter's official logic. Returns (passed, details)."""
                import html
                
                # Unescape HTML entities (Twitter's unescape function)
                def unescape(text):
                    return html.unescape(html.unescape(text)) if isinstance(text, str) else text
                
                # Extract URLs using Twitter's regex pattern
                def extract_urls(text):
                    """Return list of URL variants for each URL found."""
                    pattern = re.compile(
                        r"""
                        (?:
                            https?://               # optional scheme
                        )?
                        (?:www\.)?                  # optional www
                        [\w\-._~%]+                 # subdomain or domain name chars
                        \.[a-zA-Z]{2,}              # dot + top level domain (≥2 letters)
                        (?:/[^\s]*)?                # optional query/fragment
                        """,
                        re.VERBOSE,
                    )
                    raw_matches = pattern.findall(text)
                    
                    # Strip common trailing punctuation and create variants
                    strip_trailing = ".,;:!?)]}\"'"
                    results = []
                    for match in raw_matches:
                        variants = [match]
                        stripped_variant = match.rstrip(strip_trailing)
                        if stripped_variant != match:
                            variants.append(stripped_variant)
                        results.append(variants)
                    return results
                
                # Check single URL
                def check_url(url):
                    """Check if URL returns HTTP 200. Retries with browser headers and scheme handling."""
                    # Auto-add HTTPS scheme if missing (AI models sometimes omit it)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    
                    # Use browser-like headers to avoid anti-bot protection (e.g. Wikipedia, NYTimes)
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                    }
                    
                    # Retry logic: Twitter docs say "retries for a short time"
                    for attempt in range(2):
                        try:
                            # Try HEAD first (faster)
                            response = requests.head(url, timeout=10, allow_redirects=True, headers=headers)
                            if response.status_code == 200:
                                return True
                            # Try GET if HEAD doesn't return 200
                            response = requests.get(url, timeout=10, allow_redirects=True, headers=headers, stream=True)
                            return response.status_code == 200
                        except requests.RequestException:
                            if attempt == 0:
                                # Retry once on failure
                                continue
                            return False
                    return False
                
                # Main validation logic (matches Twitter's check_all_urls_for_note)
                note_text_unescaped = unescape(note_text)
                url_variant_lists = extract_urls(note_text_unescaped)
                
                if len(url_variant_lists) == 0:
                    return False, "No URLs found in note text"
                
                for url_variant_list in url_variant_lists:
                    at_least_one_good_url_variant = False
                    for url_variant in url_variant_list:
                        if check_url(url_variant):
                            at_least_one_good_url_variant = True
                            break
                    if not at_least_one_good_url_variant:
                        return False, f"No valid URL variant found for {url_variant_list[0]}"
                
                return True, f"All {len(url_variant_lists)} URLs validated (HTTP 200)"
            
            def validate_harassment_abuse(text):
                """Check for inflammatory language. Returns (passed, details)."""
                # Words that suggest harassment, sarcasm, or mockery
                inflammatory = ['stupid', 'idiot', 'moron', 'dumb', 'ridiculous', 'absurd', 'pathetic', 
                               'laughable', 'joke', 'clown', 'loser', 'fool', 'insane', 'crazy']
                
                text_lower = text.lower()
                found = [word for word in inflammatory if word in text_lower]
                
                if found:
                    return False, f"Contains inflammatory language: {', '.join(found)}"
                
                # Check for excessive punctuation (!!!!, ????)
                if '!!!' in text or '???' in text:
                    return False, "Excessive punctuation suggests emotional tone"
                
                # Check for all caps words (excluding acronyms like USA, FBI)
                words = text.split()
                caps_words = [w for w in words if w.isupper() and len(w) > 3]
                if len(caps_words) > 1:
                    return False, f"Multiple all-caps words suggest shouting: {', '.join(caps_words[:3])}"
                
                return True, "Tone is neutral and professional"
            
            def validate_claim_opinion(text):
                """Check if note addresses claims without opinion/speculation. Returns (passed, details)."""
                # Opinion/speculation words that should be avoided
                speculation_words = ['may', 'might', 'could', 'possibly', 'likely', 'unlikely', 
                                    'appears', 'seems', 'suggests', 'indicates', 'implies', 
                                    'probably', 'perhaps', 'maybe', 'allegedly']
                
                text_lower = text.lower()
                found_speculation = [word for word in speculation_words if f' {word} ' in f' {text_lower} ']
                
                if found_speculation:
                    return False, f"Contains speculative language: {', '.join(found_speculation[:3])}"
                
                # Check for subjective judgments
                subjective = ['misleading', 'false', 'wrong', 'incorrect', 'inaccurate', 'untrue']
                found_subjective = [word for word in subjective if word in text_lower]
                
                if found_subjective:
                    return False, f"Contains subjective judgments - state facts instead: {', '.join(found_subjective[:2])}"
                
                # Good: should contain factual indicators
                factual_indicators = ['according to', 'data shows', 'records show', 'reported', 
                                     'confirmed', 'verified', 'documented', 'published']
                has_factual = any(indicator in text_lower for indicator in factual_indicators)
                
                if not has_factual and len(text) > 100:
                    return False, "Note should cite sources with phrases like 'according to', 'data shows', etc."
                
                return True, "Addresses claims with facts, not opinion"
            
            # Run validations
            validation_results = []
            
            # 1. Length validation (existing)
            length_valid = effective_length <= 280
            validation_results.append(("Length", length_valid, f"{effective_length}/280 chars" if length_valid else f"{effective_length}/280 chars - TOO LONG"))
            
            # 2. URL Validity (95%+ must pass) - Uses Twitter's official validator logic
            url_valid, url_details = validate_url_validity(clean_note_text)
            validation_results.append(("UrlValidity", url_valid, url_details))
            
            # 3. Twitter's Evaluate Note API - Official scoring from Twitter
            twitter_eval_valid = True
            twitter_eval_details = "Not evaluated"
            twitter_claim_score = None
            
            try:
                # Call Twitter's evaluate_note API for official scoring
                eval_payload = {
                    "note_text": clean_note_text,
                    "post_id": post_id
                }
                
                eval_response = oauth.post(
                    "https://api.x.com/2/evaluate_note",
                    json=eval_payload
                )
                
                if eval_response.status_code == 200:
                    eval_data = eval_response.json()
                    if 'data' in eval_data and 'claim_opinion_score' in eval_data['data']:
                        twitter_claim_score = eval_data['data']['claim_opinion_score']
                        # Lower scores are better (more claim-based, less opinion)
                        # Based on Twitter docs, scores above certain thresholds indicate too much opinion
                        if twitter_claim_score <= 50:
                            twitter_eval_valid = True
                            twitter_eval_details = f"Claim/Opinion Score: {twitter_claim_score}/100 (excellent - fact-based)"
                        elif twitter_claim_score <= 70:
                            twitter_eval_valid = True
                            twitter_eval_details = f"Claim/Opinion Score: {twitter_claim_score}/100 (good)"
                        else:
                            twitter_eval_valid = False
                            twitter_eval_details = f"Claim/Opinion Score: {twitter_claim_score}/100 (too opinionated - needs more facts)"
                        
                        log_to_file(f"TWITTER EVALUATE API: {twitter_eval_details}")
                    else:
                        twitter_eval_details = "API returned no score"
                        log_to_file(f"TWITTER EVALUATE API: No claim_opinion_score in response: {eval_data}")
                else:
                    twitter_eval_details = f"API error: HTTP {eval_response.status_code}"
                    log_to_file(f"TWITTER EVALUATE API: Error {eval_response.status_code}: {eval_response.text}")
            except Exception as eval_error:
                twitter_eval_details = f"API call failed: {str(eval_error)}"
                log_to_file(f"TWITTER EVALUATE API: Exception - {eval_error}")
            
            validation_results.append(("TwitterEvaluate", twitter_eval_valid, twitter_eval_details))
            
            # 4. Harassment/Abuse (98%+ must pass) - COMMENTED OUT (TOO TWITCHY)
            # harassment_valid, harassment_details = validate_harassment_abuse(clean_note_text)
            # validation_results.append(("HarassmentAbuse", harassment_valid, harassment_details))
            
            # 5. Claim/Opinion (30%+ must pass) - COMMENTED OUT (replaced by Twitter API)
            # claim_valid, claim_details = validate_claim_opinion(clean_note_text)
            # validation_results.append(("ClaimOpinion", claim_valid, claim_details))
            
            # Log all validation results
            log_to_file("VALIDATION RESULTS:")
            for name, passed, details in validation_results:
                status = "✓ PASS" if passed else "✗ FAIL"
                log_to_file(f"  {name}: {status} - {details}")
            
            # Check if any critical validations failed
            failed_validations = [name for name, passed, _ in validation_results if not passed]
            
            # Retry logic if note is too long or fails validation (total tries is retries + 1)
            max_retries = 3
            retry_count = 0
            
            # Pre-shuffle model list once for all retries (draw without replacement)
            import random
            secure_random = random.SystemRandom()
            retry_model_list = [
                {"name": "grok-4-1-fast-reasoning", "api": "xai"},
                {"name": "gpt-5-mini", "api": "openai"},
                {"name": "claude-haiku-4-5", "api": "anthropic"}
            ]
            secure_random.shuffle(retry_model_list)  # Shuffle once, then rotate through this order
            
            # Build conversation history for retries - start with original system prompt and context
            retry_conversation_history = []
            retry_conversation_history.append({
                "role": "assistant",
                "content": original_note_with_attribution
            })
            
            while retry_count <= max_retries:
                if failed_validations:
                    log_to_file(f"VALIDATION FAILED: {', '.join(failed_validations)} (attempt {retry_count + 1}/{max_retries + 1})")
                    if retry_count < max_retries:
                        log_to_file("RETRYING: Asking model to fix validation issues...")
                        
                        # Build specific feedback for the model
                        feedback = f"\n\nIMPORTANT: Your previous note failed validation. Here is your previous version:\n\n\"\"\"\n{clean_note_text}\n\"\"\"\n\nIssues to fix:\n"
                        for name, passed, details in validation_results:
                            if not passed:
                                feedback += f"- {name}: {details}\n"
                        
                        feedback += "\nPlease revise the note to address these issues. "
                        if not length_valid:
                            feedback += f"Make the text under 280 effective chars (currently {effective_length}). Do *not* change any of the URLs unless instructed below. URLs count as 1 character in the length calculation. "
                        if not url_valid:
                            feedback += "Use only direct, accessible URLs from authoritative sources which you found in your search results (no search pages, galleries, or broken links). *ONLY* change the URL that didn't pass validation. Remove it or -if there are very few other URLs - replace with another URL from your actual search results. "
                        if not twitter_eval_valid and twitter_claim_score is not None:
                            feedback += f"Twitter's official evaluation scored this note {twitter_claim_score}/100 on claim vs opinion (lower is better). The note is too opinionated. Focus on stating verifiable facts with authoritative sources rather than making subjective judgments. "
                        # if not harassment_valid:
                        #     feedback += "Use neutral, professional tone - remove inflammatory language. "
                        # if not claim_valid:
                        #     feedback += "State only verifiable facts - avoid speculation words like 'may', 'might', 'could', 'appears', 'seems'. "
                        
                        # Instead of re-running full fact_check pipeline (8 models total),
                        # just re-run a single model with the feedback
                        log_to_file("RETRY: Using single model with full conversation history")
                        
                        # Initialize model clients (only once per retry)
                        xai_client = xai_sdk.Client(api_key=keys.get('XAI_API_KEY'))
                        openai_client = openai.OpenAI(api_key=keys.get('CHATGPT_API_KEY'), base_url="https://api.openai.com/v1")
                        anthropic_client = anthropic.Anthropic(api_key=keys.get('ANTHROPIC_API_KEY'))
                        
                        # Map model names to their clients
                        client_map = {
                            "xai": xai_client,
                            "openai": openai_client,
                            "anthropic": anthropic_client
                        }
                        
                        # Rotate through pre-shuffled model list (no replacement)
                        model_info = retry_model_list[retry_count % len(retry_model_list)]
                        combining_model = {
                            "name": model_info["name"],
                            "client": client_map[model_info["api"]],
                            "api": model_info["api"]
                        }
                        log_to_file(f"RETRY {retry_count + 1}: Using lower-tier model: {combining_model['name']}")
                        
                        # Build comprehensive retry message including full original context
                        retry_system_prompt = {
                            "role": "system",
                            "content": f"You are @ConSenseAI writing Community Notes. {context.get('context_instructions', '')}"
                        }
                        
                        # Rebuild context string (similar to fact_check function)
                        context_str = ""
                        if len(context.get('ancestor_chain', [])) > 0:
                            context_str += "\nThread hierarchy:\n"
                            context_str += build_ancestor_chain(context.get('ancestor_chain', []), from_cache=context.get('from_cache', False))
                        
                        media_str = format_media(context.get('media', []), context.get('ancestor_chain', []))
                        urls_str = format_urls(context.get('ancestor_chain', []))
                        
                        # Build user message with full conversation history
                        retry_user_msg = f"ORIGINAL TASK:\nContext:\n{context_str}\n{media_str}{urls_str}\nTweet to fact-check: {post_text}\n\n"
                        retry_user_msg += f"INITIAL NOTE GENERATED:\n{original_note_with_attribution}\n\n"
                        
                        # Add all prior retry attempts to show iteration history
                        for i, msg in enumerate(retry_conversation_history):
                            if msg['role'] == 'user':
                                retry_user_msg += f"\n--- Validation Feedback (Attempt {i//2 + 1}) ---\n{msg['content']}\n"
                            elif msg['role'] == 'assistant' and i > 0:  # Skip first assistant message (already included)
                                retry_user_msg += f"\n--- Revised Note (Attempt {(i+1)//2}) ---\n{msg['content']}\n"
                        
                        # Add current validation feedback
                        retry_user_msg += f"\n--- CURRENT VALIDATION ISSUES ---\n{feedback}\n\n"
                        retry_user_msg += "Please provide a revised note that addresses all validation issues. Provide ONLY the note text, no commentary or attribution."
                        
                        # Append this exchange to conversation history
                        retry_conversation_history.append({
                            "role": "user",
                            "content": f"{feedback}\n\nPlease revise to fix these issues."
                        })
                        
                        # Run the model with full context
                        retry_verdict = {}
                        retry_verdict = run_model(retry_system_prompt, retry_user_msg, combining_model, retry_verdict, max_tokens=500, context=context, verbose=False)
                        
                        note_text = retry_verdict.get(combining_model['name'], clean_note_text)
                        
                        # Append model's response to conversation history
                        retry_conversation_history.append({
                            "role": "assistant",
                            "content": note_text
                        })
                        
                        # Re-clean and re-validate
                        lines = note_text.split('\n')
                        filtered_lines = [line for line in lines if not (line.strip().startswith('Generated by:') or line.strip().startswith('Combined by:'))]
                        clean_note_text = '\n'.join(filtered_lines).strip()
                        urls_in_note = url_pattern.findall(clean_note_text)
                        text_without_urls = url_pattern.sub('', clean_note_text)
                        effective_length = len(text_without_urls) + len(urls_in_note)
                        
                        # Re-run validations
                        validation_results = []
                        length_valid = effective_length <= 280
                        validation_results.append(("Length", length_valid, f"{effective_length}/280 chars" if length_valid else f"{effective_length}/280 chars - TOO LONG"))
                        
                        url_valid, url_details = validate_url_validity(clean_note_text)
                        validation_results.append(("UrlValidity", url_valid, url_details))
                        
                        # Re-run Twitter evaluation
                        twitter_eval_valid = True
                        twitter_eval_details = "Not evaluated"
                        twitter_claim_score = None
                        
                        try:
                            eval_payload = {
                                "note_text": clean_note_text,
                                "post_id": post_id
                            }
                            
                            eval_response = oauth.post(
                                "https://api.x.com/2/evaluate_note",
                                json=eval_payload
                            )
                            
                            if eval_response.status_code == 200:
                                eval_data = eval_response.json()
                                if 'data' in eval_data and 'claim_opinion_score' in eval_data['data']:
                                    twitter_claim_score = eval_data['data']['claim_opinion_score']
                                    if twitter_claim_score <= 50:
                                        twitter_eval_valid = True
                                        twitter_eval_details = f"Claim/Opinion Score: {twitter_claim_score}/100 (excellent - fact-based)"
                                    elif twitter_claim_score <= 70:
                                        twitter_eval_valid = True
                                        twitter_eval_details = f"Claim/Opinion Score: {twitter_claim_score}/100 (good)"
                                    else:
                                        twitter_eval_valid = False
                                        twitter_eval_details = f"Claim/Opinion Score: {twitter_claim_score}/100 (too opinionated - needs more facts)"
                                    log_to_file(f"TWITTER EVALUATE API (retry): {twitter_eval_details}")
                                else:
                                    twitter_eval_details = "API returned no score"
                            else:
                                twitter_eval_details = f"API error: HTTP {eval_response.status_code}"
                        except Exception as eval_error:
                            twitter_eval_details = f"API call failed: {str(eval_error)}"
                        
                        validation_results.append(("TwitterEvaluate", twitter_eval_valid, twitter_eval_details))
                        
                        # harassment_valid, harassment_details = validate_harassment_abuse(clean_note_text)
                        # validation_results.append(("HarassmentAbuse", harassment_valid, harassment_details))
                        
                        # claim_valid, claim_details = validate_claim_opinion(clean_note_text)
                        # validation_results.append(("ClaimOpinion", claim_valid, claim_details))
                        
                        log_to_file(f"RETRY RESULT:")
                        log_to_file(clean_note_text)
                        log_to_file("RETRY VALIDATION RESULTS:")
                        for name, passed, details in validation_results:
                            status = "✓ PASS" if passed else "✗ FAIL"
                            log_to_file(f"  {name}: {status} - {details}")
                        
                        failed_validations = [name for name, passed, _ in validation_results if not passed]
                        retry_count += 1
                    else:
                        log_to_file("VALIDATION: Note still fails validation after retry, skipping submission")
                        break
                else:
                    # Note passes all validations
                    log_to_file(f"VALIDATION: ALL CHECKS PASSED")
                    break
            
            # Skip submission if validation failed after retries
            if failed_validations:
                log_to_file(f"SUBMISSION: SKIPPED (validation failed: {', '.join(failed_validations)})")
                log_to_file("")
                continue
            
            # Prepare note submission using Twitter API v2
            # Community Notes API endpoint: POST /2/notes
            note_payload = {
                "test_mode": test_mode,
                "post_id": post_id,
                "info": {
                    "text": clean_note_text,
                    "classification": "misinformed_or_potentially_misleading",
                    "misleading_tags": ["missing_important_context"],  # Required field
                    "trustworthy_sources": True
                }
            }
            
            # In global dry run mode, just log the note
            # But if only test_mode is true, we should still submit (as a test note)
            if args.dryrun:
                print(f"[Community Notes DRY RUN] Would submit note for {post_id}:")
                print(f"  Note text: {clean_note_text}")
                print(f"  Payload: {json.dumps(note_payload, indent=2)}")
                log_to_file("SUBMISSION: DRY RUN (would submit)")
                log_to_file(f"PAYLOAD: {json.dumps(note_payload, indent=2)}")
            else:
                try:
                    # Submit the note using Twitter API v2 with OAuth 1.0a (CN project keys)
                    cn_api_key = keys.get('CN_XAPI_key') or keys.get('XAPI_key')
                    cn_api_secret = keys.get('CN_XAPI_secret') or keys.get('XAPI_secret')
                    cn_access_token = keys.get('CN_access_token') or keys.get('access_token')
                    cn_access_secret = keys.get('CN_access_token_secret') or keys.get('access_token_secret')
                    
                    oauth_submit = OAuth1Session(
                        client_key=cn_api_key,
                        client_secret=cn_api_secret,
                        resource_owner_key=cn_access_token,
                        resource_owner_secret=cn_access_secret
                    )
                    
                    submit_response = oauth_submit.post(
                        "https://api.twitter.com/2/notes",
                        json=note_payload
                    )
                    
                    if submit_response.status_code in [200, 201]:
                        print(f"[Community Notes] Successfully submitted note for {post_id}")
                        print(f"  Note text: {clean_note_text}")
                        log_to_file(f"SUBMISSION: SUCCESS (HTTP {submit_response.status_code})")
                        log_to_file(f"RESPONSE: {submit_response.text}")
                    elif submit_response.status_code == 403 and "already created a note" in submit_response.text:
                        # Twitter says we already submitted a note for this post - sync our local tracking
                        print(f"[Community Notes] Note already exists on Twitter for {post_id} - syncing local tracking")
                        log_to_file(f"SUBMISSION: SKIPPED (HTTP 403 - note already exists on Twitter)")
                        log_to_file(f"ERROR RESPONSE: {submit_response.text}")
                        # Record this so we don't try again
                        written_notes[post_id] = {
                            "note": clean_note_text,
                            "test_mode": test_mode,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "status": "already_exists_on_twitter"
                        }
                        notes_written += 1
                        continue
                    else:
                        print(f"[Community Notes] Error submitting note for {post_id}: HTTP {submit_response.status_code}")
                        print(f"  Response: {submit_response.text}")
                        log_to_file(f"SUBMISSION: FAILED (HTTP {submit_response.status_code})")
                        log_to_file(f"ERROR RESPONSE: {submit_response.text}")
                        continue
                    
                except Exception as e:
                    print(f"[Community Notes] Error submitting note for {post_id}: {e}")
                    log_to_file(f"SUBMISSION: EXCEPTION - {e}")
                    import traceback
                    log_to_file(traceback.format_exc())
                    continue
            
            # Only record notes that were actually submitted (not dry run)
            if not args.dryrun:
                written_notes[post_id] = {
                    "note": clean_note_text,
                    "test_mode": test_mode,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                notes_written += 1
                
                # Save after each successful submission for real-time tracking
                try:
                    with open(COMMUNITY_NOTES_WRITTEN_FILE, 'w') as f:
                        json.dump(written_notes, f, indent=2)
                    print(f"[Community Notes] Updated tracking file: {len(written_notes)} total processed posts")
                    log_to_file(f"TRACKING: Updated {COMMUNITY_NOTES_WRITTEN_FILE} with {len(written_notes)} entries")
                except Exception as save_error:
                    print(f"[Community Notes] Error saving tracking file after post {post_id}: {save_error}")
                    log_to_file(f"TRACKING ERROR: Failed to save file - {save_error}")
            
        except Exception as e:
            print(f"[Community Notes] Error processing post {post_id}: {e}")
            log_to_file(f"ERROR: Exception during processing - {e}")
            import traceback
            log_to_file(traceback.format_exc())
            traceback.print_exc()
            continue
    
    # Save written notes record
    log_to_file("\n" + "=" * 80)
    log_to_file(f"RUN COMPLETE: {notes_written} notes processed")
    log_to_file("=" * 80 + "\n")
    
    try:
        with open(COMMUNITY_NOTES_WRITTEN_FILE, 'w') as f:
            json.dump(written_notes, f, indent=2)
        print(f"[Community Notes] Saved record of {len(written_notes)} processed posts")
    except Exception as e:
        print(f"[Community Notes] Error saving written notes file: {e}")
    
    # Update last check timestamp
    try:
        with open(COMMUNITY_NOTES_LAST_CHECK_FILE, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
    except Exception as e:
        print(f"[Community Notes] Error saving last check file: {e}")
    
    print(f"[Community Notes] Completed: {notes_written} notes written")
    print(f"[Community Notes] Full log saved to {log_filename}")
    return notes_written

# The main loop
def main():
    # Initialize search term variables outside the restart loop so they persist across restarts
    auto_search_mode = False
    current_search_term = None
    used_search_terms = []  # Track all search terms used in this run
    
    # Check if auto search mode is enabled (only once at startup)
    if getattr(args, 'search_term', None):
        if args.search_term.lower() == 'auto':
            auto_search_mode = True
            print("[Main] Auto search mode enabled - will load or generate search terms")
            # Load persistent state if available
            loaded_current, loaded_used, loaded_ids = load_auto_search_state()
            if loaded_current:
                current_search_term = loaded_current
                used_search_terms = loaded_used
                print(f"[Main] Loaded persistent search term: '{current_search_term}' ({len(used_search_terms)} terms in history)")
            else:
                print("[Main] No persistent search terms found - will generate initial term")
        else:
            current_search_term = args.search_term
            used_search_terms.append(current_search_term)
            print(f"[Main] Using static search term: {current_search_term}")
    
    while True:
        try:
            # Authenticate with retry logic for network resilience
            retry_with_backoff(authenticate, max_retries=10, initial_delay=10)
        except Exception as e:
            print(f"[Main] Authentication failed after all retries: {e}")
            print(f"[Main] Waiting {RESTART_DELAY * 6} seconds before restart...")
            time.sleep(RESTART_DELAY * 6)  # Longer wait for network issues
            continue
        
        user_id = BOT_USER_ID
        
        # Fallback: If BOT_USER_ID is still None after authentication, use hardcoded value
        if user_id is None:
            user_id = 1948872778989666305  # @ConSenseAI's user ID
            print(f"[Main] Using fallback BOT_USER_ID: {user_id}")
        
        # Generate initial search term for auto mode (only if we don't have one yet)
        if auto_search_mode and not current_search_term:
            print("[Main] Generating initial search term for auto mode...")
            current_search_term = generate_auto_search_term(current_term=None, used_terms=used_search_terms)
            if current_search_term:
                used_search_terms.append(current_search_term)
                save_auto_search_state(current_search_term, used_search_terms)
                print(f"[Main] Generated initial search term: {current_search_term}")
            else:
                print("[Main] Warning: Failed to generate initial search term, will retry after first reflection")
        
        # Initialize the last summary counter after authentication so BOT_USER_ID is available
        try:
            baseline, total_replies, last_direct = compute_baseline_replies_since_last_direct_post()
            # Start with the baseline (replies up to last direct post). Subsequent triggers count replies after that post.
            last_summary_count = int(baseline)
            print(f"[Main] Initial last_summary_count (replies up to last direct post {last_direct}): {last_summary_count} (total replies: {total_replies})")
        except Exception as e:
            print(f"[Main] Warning initializing last_summary_count from files: {e}")
            try:
                last_summary_count = get_total_bot_reply_count()
            except Exception:
                last_summary_count = 0
        # Initialize search safety stores
        try:
            if not os.path.exists(SENT_HASHES_FILE):
                with open(SENT_HASHES_FILE, 'w') as f:
                    json.dump({}, f)
            if not os.path.exists(SEARCH_REPLY_COUNT_FILE):
                with open(SEARCH_REPLY_COUNT_FILE, 'w') as f:
                    json.dump({}, f)
            if not os.path.exists(APPROVAL_QUEUE_FILE):
                with open(APPROVAL_QUEUE_FILE, 'w') as f:
                    json.dump([], f)
            if not os.path.exists(FOLLOWED_USERS_LAST_CHECK_FILE):
                with open(FOLLOWED_USERS_LAST_CHECK_FILE, 'w') as f:
                    json.dump({}, f)
            if not os.path.exists(FOLLOWED_USERS_REPLY_COUNT_FILE):
                with open(FOLLOWED_USERS_REPLY_COUNT_FILE, 'w') as f:
                    json.dump({}, f)
            if not os.path.exists(FOLLOWED_USERS_ROTATION_FILE):
                with open(FOLLOWED_USERS_ROTATION_FILE, 'w') as f:
                    json.dump({"last_index": 0, "last_checked": datetime.datetime.now().isoformat()}, f)
        except Exception as e:
            print(f"[SearchInit] Warning initializing search state files: {e}")
        try:
            while True:
                fetch_and_process_mentions(user_id, username)  # Changed from fetch_and_process_tweets

                # If a search term is available (either static or auto-generated), run the search pipeline
                # The search pipeline honors dry-run, human approval, dedupe and daily caps.
                if current_search_term:
                    try:
                        if auto_search_mode:
                            print(f"[Main] Running auto-generated search for term: {current_search_term}")
                        else:
                            print(f"[Main] Running search for term: {current_search_term}")
                        fetch_and_process_search(current_search_term, user_id=user_id)
                    except Exception as e:
                        print(f"[Main] Search error: {e}")
                        raise  # This will propagate the error to the outer loop and trigger a restart

                # Check tweets from followed users if enabled
                if getattr(args, 'check_followed_users', False):
                    try:
                        print(f"[Main] Checking tweets from followed users")
                        fetch_and_process_followed_users()
                    except Exception as e:
                        print(f"[Main] Followed users check error: {e}")
                        # Don't raise - just log and continue

                # Check for Community Notes eligible posts if enabled (unless running on reflection cycle)
                cn_on_reflection = getattr(args, 'cn_on_reflection', False)
                if getattr(args, 'check_community_notes', False) and not cn_on_reflection:
                    try:
                        print(f"[Main] Checking for Community Notes eligible posts")
                        cn_max_results = getattr(args, 'cn_max_results', 5)
                        cn_test_mode = getattr(args, 'cn_test_mode', True)
                        fetch_and_process_community_notes(user_id=user_id, max_results=cn_max_results, test_mode=cn_test_mode)
                    except Exception as e:
                        print(f"[Main] Community Notes check error: {e}")
                        # Don't raise - just log and continue

                # Check whether enough bot replies have occurred to trigger a reflection post
                try:
                    current_count = get_total_bot_reply_count()
                    print(f"[Main] Current successful post count: {current_count} (last: {last_summary_count})")
                    if (current_count - last_summary_count) >= int(args.post_interval):
                        print(f"[Main] Triggering reflection: {args.post_interval} posts reached")
                        created_id = post_reflection_on_recent_bot_threads(int(args.post_interval))
                        # Only advance the baseline if a reflection was actually posted
                        if created_id:
                            last_summary_count = current_count
                            
                            # Run Community Notes check on reflection cycle if enabled
                            if getattr(args, 'check_community_notes', False) and cn_on_reflection:
                                try:
                                    print(f"[Reflection] Checking for Community Notes eligible posts")
                                    cn_max_results = getattr(args, 'cn_max_results', 5)
                                    cn_test_mode = getattr(args, 'cn_test_mode', True)
                                    fetch_and_process_community_notes(user_id=user_id, max_results=cn_max_results, test_mode=cn_test_mode)
                                except Exception as e:
                                    print(f"[Reflection] Community Notes check error: {e}")
                                    # Don't raise - just log and continue
                            
                            # If in auto search mode, generate a new search term
                            if auto_search_mode:
                                print(f"[Main] Auto search mode: generating new search term (used terms: {used_search_terms})")
                                new_term = generate_auto_search_term(current_term=current_search_term, used_terms=used_search_terms)
                                if new_term:
                                    current_search_term = new_term
                                    used_search_terms.append(new_term)
                                    save_auto_search_state(current_search_term, used_search_terms)
                                    print(f"[Main] Updated search term to: {current_search_term}")
                                    print(f"[Main] All used search terms this run: {used_search_terms}")
                                else:
                                    print("[Main] Warning: Failed to generate new search term, keeping current term")
                except Exception as e:
                    print(f"[Main] Error checking/triggering reflection: {e}")

                print(f'Waiting for {delay*backoff_multiplier} min before fetching more mentions')
                time.sleep(delay*60*backoff_multiplier)  # Wait before the next check
        except (socket.gaierror, ConnectionError, OSError) as e:
            # Network-related errors - use longer backoff
            print(f"Network error detected: {e}")
            print(f"This may be due to internet outage or DNS failure")
            print(f"Restarting script in {RESTART_DELAY * 6} seconds (60 sec) to allow network recovery...")
            time.sleep(RESTART_DELAY * 6)
            continue
        except (tweepy.TweepyException, Exception) as e:
            print(f"Critical error triggering restart: {e}")
            print(f"Restarting script in {RESTART_DELAY} seconds...")
            time.sleep(RESTART_DELAY)
            continue
        except KeyboardInterrupt:
            print("\nStopping AutoGrok.")
            break

if __name__ == "__main__":
    main()

