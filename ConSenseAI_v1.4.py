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

import json
import os
import time
import datetime
import sys
import re
import webbrowser as web_open


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
    fn = _search_last_filename(search_term)
    if os.path.exists(fn):
        try:
            with open(fn, 'r') as f:
                v = f.read().strip()
                if v:
                    return int(v)
        except Exception:
            pass
    return None

def write_last_search_id(search_term: str, tweet_id):
    fn = _search_last_filename(search_term)
    try:
        with open(fn, 'w') as f:
            f.write(str(tweet_id))
    except Exception as e:
        print(f"Error writing last search id to {fn}: {e}")

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
            if not isinstance(br, dict):
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
    
    print(f"[Search] Query='{search_term}' since_id={last_id}")
    try:
        resp = read_client.search_recent_tweets(
            query=search_term,
            since_id=last_id,
            max_results=min(args.search_max_results, 100),
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "author_id", "referenced_tweets", "attachments", "entities"],
            expansions=["referenced_tweets.id", "attachments.media_keys"],
            media_fields=["type", "url", "preview_image_url", "alt_text"]
        )
    except tweepy.TweepyException as e:
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
        if bot_id and str(getattr(t, 'author_id', '')) == str(bot_id):
            print(f"[Search] Skipping self tweet {t.id}")
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
                _add_sent_hash(reply_text)
                _increment_daily_count()
                write_last_search_id(search_term, t.id)
                # brief pause between posts
                time.sleep(5)
            elif posted == 'delay!':
                backoff_multiplier *= 2
                print(f"[Search] Post rate-limited, backing off")
                return


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
        serializable_chain.append({"tweet": tweet_dict, "quoted_tweets": quoted_dicts, "media": media})
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
    count = 0
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
            
            # If this tweet is authored by the bot and was in reply to the target user, count it
            if t_author and bot_user_id and str(t_author) == str(bot_user_id) and t_in_reply_to and str(t_in_reply_to) == str(target_user_id):
                # If we also stored the tweet id in bot_tweets, prefer that as authoritative
                if tid is None or tid in bot_tweets:
                    count += 1
                    print(f"[Per-User Count] Counted bot reply {tid} to user {target_user_id} (count now: {count})")

        # Also inspect cached bot_replies list if present
        if isinstance(cached_data, dict) and 'bot_replies' in cached_data:
            print(f"[Per-User Count] Checking cached bot_replies list ({len(cached_data['bot_replies'])} entries)")
            for br in cached_data['bot_replies']:
                br_author = str(br.get('author_id')) if br.get('author_id') is not None else None
                br_in_reply_to = str(br.get('in_reply_to_user_id')) if br.get('in_reply_to_user_id') is not None else None
                if br_author and bot_user_id and str(br_author) == str(bot_user_id) and br_in_reply_to and str(br_in_reply_to) == str(target_user_id):
                    br_id = str(br.get('id')) if br.get('id') is not None else None
                    if br_id is None or br_id in bot_tweets:
                        count += 1
                        print(f"[Per-User Count] Counted bot reply {br_id} from cached bot_replies (count now: {count})")
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
            if br_author and bot_user_id and str(br_author) == str(bot_user_id) and br_in_reply_to and str(br_in_reply_to) == str(target_user_id):
                br_id = br.get('id') if isinstance(br, dict) else getattr(br, 'id', None)
                if br_id is None or str(br_id) in bot_tweets:
                    count += 1
                    print(f"[Per-User Count] Counted bot reply {br_id} from API results (count now: {count})")

    print(f"[Per-User Count] FINAL COUNT: {count} replies to user {target_user_id} in conversation {cid}")
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
    Count how many times each user has mentioned/replied to the bot.
    Only counts tweets that directly reply to (mention) the bot.
    Returns dict: {user_id: mention_count}
    """
    user_counts = {}
    
    # Load ancestor chains which contain all conversation data
    try:
        with open(ANCESTOR_CHAIN_FILE, 'r') as f:
            chains = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[Auto-Follow] No ancestor chains found")
        return user_counts
    
    bot_id = str(getid())
    
    for conv_id, conv_data in chains.items():
        # Handle both dict format (new) and list format (legacy)
        if isinstance(conv_data, dict):
            chain = conv_data.get('chain', [])
            bot_replies = conv_data.get('bot_replies', [])
        else:
            chain = conv_data
            bot_replies = []
        
        # Only process conversations where the bot participated
        if not bot_replies and not any(str(get_attr(entry.get('tweet'), 'author_id')) == bot_id 
                                       for entry in chain if entry and isinstance(entry, dict)):
            continue
        
        # Count tweets that directly reply to the bot (in_reply_to_user_id == bot_id)
        for entry in chain:
            if not entry or not isinstance(entry, dict):
                continue
            tweet = entry.get('tweet')
            if not tweet:
                continue
            
            author_id = str(get_attr(tweet, 'author_id', ''))
            in_reply_to = str(get_attr(tweet, 'in_reply_to_user_id', ''))
            
            # Count only if this tweet is a direct reply/mention to the bot
            if author_id and author_id != bot_id and in_reply_to == bot_id:
                user_counts[author_id] = user_counts.get(author_id, 0) + 1
    
    print(f"[Auto-Follow] Counted mentions from {len(user_counts)} unique users across {len(chains)} conversations")
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
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[Auto-Follow DEBUG] twitterapi.io response keys: {list(data.keys())}")
                
                # Extract user IDs from the response
                # Try different possible response formats
                users = data.get('users', data.get('data', []))
                if isinstance(data, list):
                    users = data
                    
                for user in users:
                    if isinstance(user, dict):
                        user_id = user.get('id_str') or user.get('id')
                        if user_id:
                            actual_following.add(str(user_id))
                    elif isinstance(user, str):
                        actual_following.add(str(user))
                
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
                # OpenAI vision model - send text + images
                messages = [
                    system_prompt,
                    {"role": "user", "content": user_msg}
                ]
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
                        messages.append({
                            "role": "user",
                            "content": [*image_messages]
                        })
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
                response = model['client'].messages.create(
                    model=model['name'],
                    system=system_prompt['content'],
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search"
                        }]
                )
                # Collect all valid text blocks
                text_responses = []
                for block in response.content:
                    if block.type == "text":
                        text_responses.append(block.text.strip())
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
    print(f"[Vision Debug] Found {len(context.get('media', []))} media items for vision analysis")
    user_msg = f"Context:\n {context_str}\n{media_str}\nTweet: {full_mention_text}\n{instructions}\n"
    #print(user_msg)

    # Initialize clients
    xai_client = xai_sdk.Client(api_key=keys.get('XAI_API_KEY'))
    openai_client = openai.OpenAI(api_key=keys.get('CHATGPT_API_KEY'), base_url="https://api.openai.com/v1")
    anthropic_client = anthropic.Anthropic(api_key=keys.get('ANTHROPIC_API_KEY'))

    # Models and their clients - Updated to include vision model
    models = [
        #lower tier (index 0-2)
        {"name": "grok-4-fast-reasoning", "client": xai_client, "api": "xai"},
        {"name": "gpt-5-mini", "client": openai_client, "api": "openai"},
        {"name": "claude-haiku-4-5", "client": anthropic_client, "api": "anthropic"},
        #higher tier (index 3-5)
        {"name": "grok-4", "client": xai_client, "api": "xai"},
        {"name": "gpt-5", "client": openai_client, "api": "openai"},
        {"name": "claude-sonnet-4-5", "client": anthropic_client, "api": "anthropic"}
    ]
    
    randomized_models = models[:3].copy()
    random.shuffle(randomized_models)

    # Then proceed with runs = 1 or  to keep it efficient
    runs = 3
    
    verdict = {}
    for model in randomized_models[:runs]:  # putting it back to 3 for now

        system_prompt = { #Grok prompts available here: https://github.com/xai-org/grok-prompts
                "role": "system",
                "content": f"You are @ConSenseAI, a version of {model['name']} deployed by 'AI Against Autocracy.' This prompt will be run through multiple AI models including grok, chatgpt, and then a final pass will combine responses. This system prompt is largely based on @Grok \n\
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
        model = random.choice(models[3:])  # chooses one of the higher tier models to combine the verdicts

        #we're gonna append this message to the system prompt of the combining model
        combine_msg = "\n   - This is the final pass. You will be given responses from your previous runs of muiltiple models signified by 'Model Responses:'\n\
            -Combine those responses into a consise coherent whole.\n\
            -Provide a sense of the overall consensus, highlighting key points and any significant differences in the models' responses\n\
            -Still respond in the first person as if you are one entity.\n\
            -Do not perform additional searches.\n\
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


        # Run the combining model
        verdict = run_model(system_prompt, user_msg, model, verdict, max_tokens=500, context=context, verbose=verbose)

        #Note which models contributed to the final response
        models_verdicts = verdict[model['name']].strip()
        models_verdicts += '\n\nGenerated by: '
        models_verdicts += ' '.join(f"{model['name']}, " for model in randomized_models[:runs])
        models_verdicts += f'\nCombined by: {model["name"]}'
    except Exception as e:
            print(f"Error summarizing: {e}")
        
    # Construct reply
    try:
        version = ' ' + __file__.split('_')[1].split('.p')[0]
    except:
        version = ""
    
    # Then, use it in a simpler f-string
    reply = f"ConSenseAI{version}:\n {models_verdicts}"
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
    try:
        print(f"\nattempting reply to tweet {parent_tweet_id}: {reply_text}\n")
        response = post_client.create_tweet(text=reply_text, in_reply_to_tweet_id=parent_tweet_id)
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
        bot_ids = set(str(k) for k in bot_tweets.keys())

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


def generate_auto_search_term(n=100, current_term=None, used_terms=None):
    """Generate a search term based on recent bot threads.
    
    This is called when --search_term is set to "auto" after posting a reflection.
    Returns a single-word or short phrase search term, or None if unable to generate.
    
    Args:
        n: Number of recent threads to analyze (default 100, ~37K tokens)
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
            "Respond with ONLY the search term, nothing else. No quotes, no explanation."
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
            "Post the text of the tweet only, without any additional commentary."
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


def authenticate():
    global read_client
    global post_client
    global keys
    global BOT_USER_ID
    keys = load_keys()
    
    # Always use bearer for read_client (app-only, basic-tier app)
    read_client = tweepy.Client(bearer_token=keys['bearer_token'])
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
                access_token_secret=keys['access_token_secret']
            )
            user = post_client.get_me()
            print(f"Post client authenticated as @{user.data.username} (free tier).")
            # Cache the authenticated bot user id to avoid future get_user calls
            try:
                BOT_USER_ID = user.data.id
                print(f"[Authenticate] BOT_USER_ID cached as {BOT_USER_ID}")
            except Exception:
                pass
            if user.data.username.lower() == 'consenseai':
                print(f"Authenticated with X API v1.1 (OAuth 1.0a) as @ConSenseAI (ID: {user.data.id}) successfully using existing tokens.")
                return  # Exit early if authentication succeeds
            else:
                print(f"Warning: Existing tokens auYou thenticate as {user.data.username}, not @ConSenseAI. Proceeding with new authentication.")
        except tweepy.TweepyException as e:
            print(f"Existing tokens invalid or expired: {e}. Proceeding with new authentication.")
    
    # If no valid tokens or authentication failed, perform three-legged OAuth flow for @ConSenseAI
    print("No valid access tokens found or authentication failed. Initiating three-legged OAuth flow...")
    auth = tweepy.OAuthHandler(keys['XAPI_key'], keys['XAPI_secret'])
    try:
        # Step 1: Get request token
        redirect_url = "http://127.0.0.1:3000/"  # Ensure this matches the callback URL in X Developer Portal
        auth.set_access_token(None, None)  # Clear any existing tokens
        redirect_url = auth.get_authorization_url()
        print(f"Please go to this URL and authorize the app: {redirect_url}")
        web_open(redirect_url)  # Opens in default browser
        
        # Step 2: Get verifier from callback
        verifier = input("Enter the verifier PIN from the callback URL: ")
        
        # Step 3: Get access token
        auth.get_access_token(verifier)
        access_token = auth.access_token
        access_token_secret = auth.access_token_secret
        
        # Update keys.txt with new tokens
        with open('keys.txt', 'a') as f:
            f.write(f"access_token={access_token}\naccess_token_secret={access_token_secret}\n")
        print(f"New access tokens saved to keys.txt: {access_token}, {access_token_secret}")
        
        # Step 4: Call function again with new tokens
        #keys=load_keys()
        authenticate()
       
    except tweepy.TweepyException as e:
        print(f"Error during OAuth flow: {e}")
        exit(1)
    
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
            expansions=["referenced_tweets.id", "attachments.media_keys"],
            media_fields=["type", "url", "preview_image_url", "alt_text"]
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

                    # 1) If the mention is authored by the bot, skip (normalize types)
                    if str(getattr(mention, 'author_id', '')) == str(bot_user_id):
                        print(f"Skipping mention from self: {mention.text}")
                        success = dryruncheck()
                        write_last_tweet_id(mention.id)

                    else:
                        # 2) Count prior bot replies using ancestor cache + API-provided bot replies
                        api_bot_replies = context.get('bot_replies_in_thread')
                        target_author = getattr(mention, 'author_id', None)
                        if per_user_threshold:
                            print(f"[Mention Threshold] Checking per-user threshold for user {target_author} in conversation {conv_id}")
                            prior_replies_to_user = count_bot_replies_by_user_in_conversation(conv_id, bot_user_id, target_author, api_bot_replies)
                            print(f"[Mention Threshold] User {target_author}: {prior_replies_to_user} replies / {reply_threshold} threshold")
                            if prior_replies_to_user >= reply_threshold:
                                print(f"[Mention Threshold] SKIPPING reply to thread {conv_id}: bot already replied to user {target_author} {prior_replies_to_user} times (threshold={reply_threshold})")
                                success = dryruncheck()
                                write_last_tweet_id(mention.id)
                                continue
                            else:
                                print(f"[Mention Threshold] PROCEEDING with reply to user {target_author} ({prior_replies_to_user} < {reply_threshold})")
                        else:
                            prior_replies = count_bot_replies_in_conversation(conv_id, bot_user_id, api_bot_replies)
                            if prior_replies >= reply_threshold:
                                print(f"Skipping reply to thread {conv_id}: bot already replied {prior_replies} times (threshold={reply_threshold})")
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
    # Use .data if available, else fallback to __dict__
    if hasattr(t, 'data') and isinstance(t.data, dict):
        return t.data
    elif hasattr(t, '__dict__'):
        # Only keep serializable fields
        return {k: v for k, v in t.__dict__.items() if isinstance(v, (str, int, float, dict, list, type(None)))}
    else:
        return str(t)

def collect_quoted(refs, includes=None):
    quoted_responses = []  # Store full responses for media extraction
    for ref_tweet in refs or []:
        if ref_tweet.type == "quoted":
            try:
                quoted_response = read_client.get_tweet(
                    id=ref_tweet.id,
                    tweet_fields=["text", "author_id", "created_at", "attachments", "entities"],
                    expansions=["attachments.media_keys"],
                    media_fields=["type", "url", "preview_image_url", "alt_text"]
                )
                print(f"[DEBUG] Quoted tweet {ref_tweet.id} text length: {len(quoted_response.data.text)} chars")
                quoted_responses.append(quoted_response)
            except tweepy.TweepyException as e:
                print(f"Error fetching quoted tweet {ref_tweet.id}: {e}")
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
        "media": []
    }

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
                            media_fields=["type", "url", "preview_image_url", "alt_text"]
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
                        media_fields=["type", "url", "preview_image_url", "alt_text"]
                    )
                    if bot_replies_response.data:
                        context["bot_replies_in_thread"] = bot_replies_response.data
                        print(f"[API] Fetched {len(context['bot_replies_in_thread'])} bot replies")
                    else:
                        print("[API] No bot replies found")
                except tweepy.TweepyException as e:
                    print(f"Error fetching bot replies: {e}")

        # Extract quoted tweets and media from cached chain
        for entry in context['ancestor_chain']:
            if entry is None:
                continue
            quoted = entry.get('quoted_tweets', []) if isinstance(entry, dict) else []
            media = entry.get('media', []) if isinstance(entry, dict) else []
            context['quoted_tweets'].extend([q for q in quoted if q is not None])
            context['media'].extend([m for m in media if m is not None])

        # Attempt to derive original_tweet from chain if not fetching separately
        if context['ancestor_chain']:
            context["original_tweet"] = context['ancestor_chain'][0]['tweet']  # Root is original

        # If we have everything from cache, return early
        if isinstance(cached_data, dict) and 'thread_tweets' in cached_data and 'bot_replies' in cached_data and context['ancestor_chain']:
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
                                media_fields=["type", "url", "preview_image_url", "alt_text"]
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
            current_includes = parent_response.includes if 'parent_response' in locals() and hasattr(parent_response, 'includes') else None
            media = extract_media(current_tweet, current_includes)
            # Extract media from quoted tweets
            for qr in quoted_responses:
                media.extend(extract_media(qr.data, qr.includes))
            ancestor_chain.append({
                "tweet": current_tweet,
                "quoted_tweets": quoted,
                "media": media
            })
            visited.add(current_tweet.id)
            parent_id = None
            if hasattr(current_tweet, 'referenced_tweets') and current_tweet.referenced_tweets:
                for ref in current_tweet.referenced_tweets:
                    if ref.type == 'replied_to':
                        parent_id = ref.id
                        break
            if parent_id is None or parent_id in visited:
                break
            parent_response = read_client.get_tweet(
                id=parent_id,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id", "attachments", "entities"],
                expansions=["referenced_tweets.id", "attachments.media_keys"],
                media_fields=["type", "url", "preview_image_url", "alt_text"]
            )
            if parent_response.data:
                current_tweet = parent_response.data
                print(f"[API Debug] Fetched parent tweet {parent_id}, text length: {len(current_tweet.text)} chars")
                print(f"[API Debug] First 100 chars: {current_tweet.text[:100]}")
                print(f"[API Debug] Last 100 chars: {current_tweet.text[-100:]}")
            else:
                break
    except tweepy.TweepyException as e:
        print(f"Error building ancestor chain: {e}")

    ancestor_chain = ancestor_chain[::-1]  # Root first
    context['ancestor_chain'] = ancestor_chain

    # NEW: Ensure the current mention is appended as the final entry with its media
    if ancestor_chain and ancestor_chain[-1]['tweet'].id != tweet.id:
        mention_media = extract_media(tweet, includes)
        ancestor_chain.append({
            "tweet": tweet,
            "quoted_tweets": [qr.data for qr in collect_quoted(getattr(tweet, 'referenced_tweets', None))],  # Fetch any quoted in mention
            "media": mention_media
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
                media_fields=["type", "url", "preview_image_url", "alt_text"]
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
                media_fields=["type", "url", "preview_image_url", "alt_text"]
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
        author = f" (from @{author_id})" if author_id else ""
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
        for m in includes['media']:
            if m is None:
                continue
            media_list.append({
                'type': getattr(m, 'type', '') if hasattr(m, 'type') else (m.get('type', '') if isinstance(m, dict) else ''),
                'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')) if hasattr(m, 'url') else (m.get('url', m.get('preview_image_url', '')) if isinstance(m, dict) else ''),
                'alt_text': getattr(m, 'alt_text', '') if hasattr(m, 'alt_text') else (m.get('alt_text', '') if isinstance(m, dict) else '')
            })
            found_media = True
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
parser.add_argument('--dryrun', type=bool, help='Print responses but don\'t tweet them', default=False)
#parser.add_argument('--accuracy', type=int, help="Accuracy score threshold out of 10. Don't reply to tweets scored above this threshold")
parser.add_argument('--fetchthread', type=bool, help='If True, Try to fetch the rest of the thread for additional context. Warning: API request hungry', default=True)
parser.add_argument('--reply_threshold', type=int, help='Number of times the bot can reply in a thread before skipping further replies (default 5)', default=5)
parser.add_argument('--per_user_threshold', type=bool, help='If True, enforce reply_threshold per unique user per thread; if False, enforce per-thread total (default True)', default=True)
parser.add_argument('--search_term', type=str, help='If provided, periodically search this term and run the pipeline on matching tweets. Use "auto" to automatically generate relevant search terms after each reflection cycle.', default=None)
parser.add_argument('--search_max_results', type=int, help='Max results to fetch per search (default 10)', default=10)
parser.add_argument('--search_daily_cap', type=int, help='Max automated replies per day from searches increases every "--search_cap_interval_hours" hours(default 5)', default=5)
parser.add_argument('--dedupe_window_hours', type=float, help='Window to consider duplicates (hours, default 24)', default=24.0)
parser.add_argument('--enable_human_approval', type=bool, help='If True, queue candidate replies for human approval instead of auto-posting', default=False)
parser.add_argument('--search_cap_interval_hours', type=int, help='Number of hours between each increase in search reply cap (default 1)', default=2)
parser.add_argument('--cap_increase_time', type=str, help='Earliest time of day (HH:MM, 24h) to allow cap increases (default 10:00)', default='10:00')
parser.add_argument('--post_interval', type=int, help='Number of bot replies between posting a reflection based on recent threads (default 10)', default=10)
parser.add_argument('--follow_threshold', type=int, help='Minimum number of replies from a user before auto-following them (default 2)', default=2)
args, unknown = parser.parse_known_args()  # Ignore unrecognized arguments (e.g., Jupyter's -f)

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

RESTART_DELAY = 10
backoff_multiplier = 1

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
            print("[Main] Auto search mode enabled - will generate search terms after reflections")
        else:
            current_search_term = args.search_term
            used_search_terms.append(current_search_term)
            print(f"[Main] Using static search term: {current_search_term}")
    
    while True:
        authenticate()
        user_id = BOT_USER_ID
        
        # Generate initial search term for auto mode (only if we don't have one yet)
        if auto_search_mode and not current_search_term:
            print("[Main] Generating initial search term for auto mode...")
            current_search_term = generate_auto_search_term(current_term=None, used_terms=used_search_terms)
            if current_search_term:
                used_search_terms.append(current_search_term)
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
                            
                            # If in auto search mode, generate a new search term
                            if auto_search_mode:
                                print(f"[Main] Auto search mode: generating new search term (used terms: {used_search_terms})")
                                new_term = generate_auto_search_term(current_term=current_search_term, used_terms=used_search_terms)
                                if new_term:
                                    current_search_term = new_term
                                    used_search_terms.append(new_term)
                                    print(f"[Main] Updated search term to: {current_search_term}")
                                    print(f"[Main] All used search terms this run: {used_search_terms}")
                                else:
                                    print("[Main] Warning: Failed to generate new search term, keeping current term")
                except Exception as e:
                    print(f"[Main] Error checking/triggering reflection: {e}")

                print(f'Waiting for {delay*backoff_multiplier} min before fetching more mentions')
                time.sleep(delay*60*backoff_multiplier)  # Wait before the next check
        except (ConnectionError, tweepy.TweepyException, Exception) as e:
            print(f"Critical error triggering restart: {e}")
            print(f"Restarting script in {RESTART_DELAY} seconds...")
            time.sleep(RESTART_DELAY)
            continue
        except KeyboardInterrupt:
            print("\nStopping AutoGrok.")
            break

if __name__ == "__main__":
    main()

