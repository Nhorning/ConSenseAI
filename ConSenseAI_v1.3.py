
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

def queue_for_approval(item: dict):
    queue = _load_json_file(APPROVAL_QUEUE_FILE, [])
    queue.append(item)
    _save_json_file(APPROVAL_QUEUE_FILE, queue)


# Removed generate_reply_text_for_tweet to avoid duplicate model orchestration.
# fetch_and_process_search now calls fact_check(..., generate_only=True) to obtain generated reply text.

def fetch_and_process_search(search_term: str, user_id=None):
    """Search and run pipeline with safeguards: daily cap, dedupe, optional human approval."""
    global backoff_multiplier
    last_id = read_last_search_id(search_term)
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

    # Respect daily cap
    today_count = _get_today_count()
    if today_count >= int(args.search_daily_cap):
        print(f"[Search] Daily cap reached ({today_count}/{args.search_daily_cap}), skipping processing")
        return

    me = None
    try:
        me = read_client.get_user(username=username)
        bot_id = me.data.id
    except Exception:
        bot_id = None

    # Process older -> newer
    for t in resp.data[::-1]:
        # Debug: check whether the tweet text is already truncated when returned from the search API
        try:
            fetched_text = get_full_text(t)
        except Exception:
            fetched_text = getattr(t, 'text', '') if hasattr(t, 'text') else ''
        print(f"[Search Debug] Fetched tweet {getattr(t, 'id', 'unknown')} - type={type(t)} - text_len={len(fetched_text)}")
        print(f"[Search Debug] Preview: {fetched_text[:500]}")
        # If the returned text looks short, try re-fetching the tweet via get_tweet to compare
        if len(fetched_text) < 200:
            try:
                full = read_client.get_tweet(
                    id=getattr(t, 'id', None),
                    tweet_fields=["text", "entities", "referenced_tweets"],
                    expansions=["referenced_tweets.id"]
                )
                if getattr(full, 'data', None):
                    ref_text = full.data.get('text') if isinstance(full.data, dict) else getattr(full.data, 'text', '')
                    print(f"[Search Debug] Re-fetched tweet {getattr(t, 'id', 'unknown')} - refetch_len={len(ref_text)}")
                    print(f"[Search Debug] Re-fetch preview: {ref_text[:500]}")
            except Exception as e:
                print(f"[Search Debug] Re-fetch failed for {getattr(t, 'id', 'unknown')}: {e}")
        # basic guard: don't reply to ourselves
        if bot_id and str(getattr(t, 'author_id', '')) == str(bot_id):
            print(f"[Search] Skipping self tweet {t.id}")
            continue

        # build context
        context = get_tweet_context(t, resp.includes if hasattr(resp, 'includes') else None)
        context['mention'] = t

        # skip if bot already replied in conversation
        conv_id = str(getattr(t, 'conversation_id', ''))
        prior = count_bot_replies_in_conversation(conv_id, bot_id, context.get('bot_replies_in_thread'))
        if prior >= reply_threshold:
            print(f"[Search] Skipping {t.id}: bot already replied {prior} times")
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
            # post and track
            posted = post_reply(t.id, reply_text, conversation_id=conv_id)
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


def load_bot_tweets():
    """Load stored bot tweets from JSON file"""
    if os.path.exists(TWEETS_FILE):
        try:
            with open(TWEETS_FILE, 'r') as f:
                tweets = json.load(f)
                print(f"[Tweet Storage] Loaded {len(tweets)} stored tweets from {TWEETS_FILE}")
                return tweets
        except json.JSONDecodeError:
            print(f"[Tweet Storage] Error reading {TWEETS_FILE}, starting fresh")
    else:
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

def get_bot_tweet_content(tweet_id):
    """Retrieve full content of a bot tweet if available"""
    tweets = load_bot_tweets()
    content = tweets.get(str(tweet_id))
    if content:
        print(f"[Tweet Storage] Retrieved stored content for tweet {tweet_id} (length: {len(content)})")
    else:
        print(f"[Tweet Storage] No stored content found for tweet {tweet_id}")
    return content

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

def run_model(system_prompt, user_msg, model, verdict, max_tokens=250, context=None):
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

            #print(verdict[model['name']])
        except Exception as e:
            print(f"Error with {model['name']}: {e}")
            verdict[model['name']] = f"Error: Could not verify with {model['name']}."
        
        return verdict

def fact_check(tweet_text, tweet_id, context=None, generate_only=False):
    # Construct context string
    def get_full_text(t):
        # Return the full text directly from the 'text' field (API v2 standard)
        if hasattr(t, 'text'):
            return t.text
        elif isinstance(t, dict) and 'text' in t:
            return t['text']
        return ''  # Fallback for invalid/missing tweet

    context_str = ""
    if context:
        if len(context['ancestor_chain']) <= 1:
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
        if len(context['ancestor_chain']) > 1:
               context_str += "\nThread hierarchy:\n"
               context_str += build_ancestor_chain(context.get('ancestor_chain', []))

    # Use full text for the mention
    full_mention_text = get_full_text(context.get('mention', {})) if context and 'mention' in context else tweet_text
    print(f"[DEBUG] Full mention text length in fact_check: {len(full_mention_text)} chars")
    print(f"[DEBUG] Full mention text: {full_mention_text[:500]}...") if len(full_mention_text) > 500 else print(f"[DEBUG] Full mention text: {full_mention_text}")
    media_str = format_media(context.get('media', []), context.get('ancestor_chain', [])) if context else ""
    print(f"[Vision Debug] Found {len(context.get('media', []))} media items for vision analysis")
    user_msg = f"Context:\n {context_str}\n{media_str}\nTweet: {full_mention_text}"
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
        {"name": "claude-3-5-haiku-latest", "client": anthropic_client, "api": "anthropic"},
        #higher tier (index 3-5)
        {"name": "grok-4", "client": xai_client, "api": "xai"},
        {"name": "gpt-5", "client": openai_client, "api": "openai"},
        {"name": "claude-sonnet-4-5", "client": anthropic_client, "api": "anthropic"}
        #vision models (index 6)
        #{"name": "gpt-5", "client": openai_client, "api": "openai-vision"}
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
        verdict = run_model(system_prompt, user_msg, model, verdict, context=context)

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
        print(f"\n\nSystem Prompt:\n{system_prompt['content']}\n\n") #diagnostic
        print(user_msg)  #diagnostic


        # Run the combining model
        verdict = run_model(system_prompt, user_msg, model, verdict, max_tokens=500, context=context)

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
        success = post_reply(tweet_id, reply, conversation_id=conv_id)
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
    
def post_reply(parent_tweet_id, reply_text, conversation_id=None):
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
                    append_reply_to_ancestor_chain(conversation_id, created_id, reply_text)
                except Exception as e:
                    print(f"[Ancestor Cache] Warning: could not record reply in cache: {e}")
        else:
            print("[Tweet Storage] Warning: Could not get tweet ID from response")
            print(f"[Tweet Storage] Response data: {response}")
        print('done!')
        return 'done!'
    except tweepy.TweepyException as e:
        print(f"Error posting reply: {e}")
        #ifthe there have been too many tweets sent out, return to the main function to wait for the delay.
        if e.response.status_code == 429:
            return 'delay!'


def append_reply_to_ancestor_chain(conversation_id, reply_id, reply_text):
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
    entry = {'tweet': {'id': str(reply_id), 'author_id': None, 'text': reply_text}, 'quoted_tweets': [], 'media': []}

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
    try:
        user_info = read_client.get_user(username=username)
        user_id = user_info.data.id
        print(f"User ID for @{username} (app-only auth): {user_id}")
        return user_id
    except tweepy.TweepyException as e:
        print(f"Error fetching @{username}'s user ID: {e}")
        exit(1)

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
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "referenced_tweets", "attachments", "entities"],
            expansions=["referenced_tweets.id", "attachments.media_keys"],
            media_fields=["type", "url", "preview_image_url", "alt_text"]
        )
        
        if mentions.data:
            for mention in mentions.data[::-1]:  # Process in reverse order to newest first
                # print(f"\n[DEBUG] ===== RAW MENTION OBJECT =====")
                # print(f"[DEBUG] Mention ID: {mention.id}")
                # print(f"[DEBUG] Mention from: {mention.author_id}")
                # print(f"[DEBUG] Tweet text length: {len(mention.text)} chars")
                # print(f"[DEBUG] Full mention text: {mention.text}")
                # print(f"[DEBUG] Has 'text' attribute: {hasattr(mention, 'text')}")
                # print(f"[DEBUG] Mention object type: {type(mention)}")
                # print(f"[DEBUG] Mention.__dict__ keys: {list(mention.__dict__.keys()) if hasattr(mention, '__dict__') else 'N/A'}")
                # print(f"[DEBUG] All mention attributes: {dir(mention)}")
                # print(f"[DEBUG] ===================================")
                
                # Fetch conversation context
                context = get_tweet_context(mention, mentions.includes)
                context['mention'] = mention  # Store the mention in context


                # Safety checks using persisted caches + API results
                conv_id = str(getattr(mention, 'conversation_id', ''))
                bot_user_id = user_id

                # 1) If the mention is authored by the bot, skip (normalize types)
                if str(getattr(mention, 'author_id', '')) == str(bot_user_id):
                    print(f"Skipping mention from self: {mention.text}")
                    success = dryruncheck()

                else:
                    # 2) Count prior bot replies using ancestor cache + API-provided bot replies
                    api_bot_replies = context.get('bot_replies_in_thread')
                    prior_replies = count_bot_replies_in_conversation(conv_id, bot_user_id, api_bot_replies)
                    if prior_replies >= reply_threshold:
                        print(f"Skipping reply to thread {conv_id}: bot already replied {prior_replies} times (threshold={reply_threshold})")
                        success = dryruncheck()
                    else:
                        # Pass context to fact_check and reply
                        success = fact_check(mention.text, mention.id, context)
                if success == 'done!':
                    last_tweet_id = max(last_tweet_id, mention.id)
                    write_last_tweet_id(last_tweet_id)
                    backoff_multiplier = 1
                    time.sleep(30)
                if success == 'delay!':
                    backoff_multiplier *= 2
                    print(f'Backoff Multiplier:{backoff_multiplier}')
                    return
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

def get_tweet_context(tweet, includes=None):
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
                    bot_replies_response = read_client.search_recent_tweets(
                        query=f"conversation_id:{conv_id} from:{username}",
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
            context['quoted_tweets'].extend(entry.get('quoted_tweets', []))
            context['media'].extend(entry.get('media', []))

        # Attempt to derive original_tweet from chain if not fetching separately
        if context['ancestor_chain']:
            context["original_tweet"] = context['ancestor_chain'][0]['tweet']  # Root is original

        # If we have everything from cache, return early
        if isinstance(cached_data, dict) and 'thread_tweets' in cached_data and 'bot_replies' in cached_data and context['ancestor_chain']:
            print("[Context Cache] All context loaded from cache - skipping API calls")
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
            bot_replies_response = read_client.search_recent_tweets(
                query=f"conversation_id:{conv_id} from:{username}",
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
        context['quoted_tweets'].extend(entry.get('quoted_tweets', []))
    context['quoted_tweets'].extend(quoted_from_api)  # Any extras

    # Set original tweet if available
    if context['ancestor_chain']:
        context["original_tweet"] = context['ancestor_chain'][0]['tweet']

    # Collect media
    context['media'].extend(extract_media(tweet, includes))
    for entry in context['ancestor_chain']:
        context['media'].extend(entry.get('media', []))

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

def build_ancestor_chain(ancestor_chain, indent=0):
    out = ""

    for i, entry in enumerate(ancestor_chain):
        t = entry["tweet"]
        quoted_tweets = entry["quoted_tweets"]
        # Support Tweepy objects and cached dicts
        tweet_id = get_attr(t, "id")
        author_id = get_attr(t, "author_id", "")
        tweet_text = get_full_text(t)
        print(f"[Ancestor Chain] Tweet {tweet_id} text length in build: {len(tweet_text)} chars")
        is_bot_tweet = str(author_id) == str(getid())
        if is_bot_tweet and tweet_id:
            print(f"[Tweet Storage] Found bot tweet {tweet_id} in ancestor chain")
            stored_content = get_bot_tweet_content(tweet_id)
            if stored_content:
                tweet_text = stored_content
                print(f"[Tweet Storage] Using stored content for tweet {tweet_id}")
            else:
                print(f"[Tweet Storage] Using API content for bot tweet {tweet_id} (no stored version found)")
        author = f" (from @{author_id})" if author_id else ""
        out += "  " * indent + f"- {tweet_text}{author}\n"
        # Show quoted tweets indented under their parent
        for qt in quoted_tweets:
            qt_author_id = qt.get('author_id') if isinstance(qt, dict) else getattr(qt, 'author_id', '')
            qt_author = f" (quoted @{qt_author_id})" if qt_author_id else ""
            qt_text = get_full_text(qt)
            out += "  " * (indent + 1) + f"> {qt_text}{qt_author}\n"
        indent += 1
    return out

def extract_media(t, includes=None):
    media_list = []
    found_media = False

    # Added debug logging for includes
    print(f"[Media Debug] Extracting media for tweet {get_attr(t, 'id')} - includes provided: {includes is not None}")
    if includes:
        print(f"[Media Debug] Includes keys: {list(includes.keys()) if isinstance(includes, dict) else 'Not dict'}")
        if 'media' in includes:
            print(f"[Media Debug] Found 'media' in includes with {len(includes['media'])} items")

    # FIRST: Check includes for media (most reliable source from API responses)
    if includes and 'media' in includes:
        for m in includes['media']:
            media_list.append({
                'type': getattr(m, 'type', '') if hasattr(m, 'type') else m.get('type', ''),
                'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')) if hasattr(m, 'url') else m.get('url', m.get('preview_image_url', '')),
                'alt_text': getattr(m, 'alt_text', '') if hasattr(m, 'alt_text') else m.get('alt_text', '')
            })
            found_media = True
        print(f"[Media Debug] Extracted {len(media_list)} media items from includes['media']")

    # Handle dicts with 'media' key
    if isinstance(t, dict) and 'media' in t:
        for m in t['media']:
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
            media_list.append({
                'type': getattr(m, 'type', ''),
                'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')),
                'alt_text': getattr(m, 'alt_text', '')
            })
            found_media = True

    # Optionally, handle Tweepy objects with direct 'media' attribute
    elif hasattr(t, 'media') and isinstance(t.media, list):
        for m in t.media:
            media_list.append({
                'type': getattr(m, 'type', ''),
                'url': getattr(m, 'url', getattr(m, 'preview_image_url', '')),
                'alt_text': getattr(m, 'alt_text', '')
            })
            found_media = True

    # Updated: Check for image URLs in entities.urls with enhanced debug logging and Twitter media pattern detection
    entities = get_attr(t, 'entities', {})
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
parser.add_argument('--search_term', type=str, help='If provided, periodically search this term and run the pipeline on matching tweets', default=None)
parser.add_argument('--search_max_results', type=int, help='Max results to fetch per search (default 10)', default=10)
parser.add_argument('--search_daily_cap', type=int, help='Max automated replies per day from searches (default 5)', default=5)
parser.add_argument('--dedupe_window_hours', type=float, help='Window to consider duplicates (hours, default 24)', default=24.0)
parser.add_argument('--enable_human_approval', type=bool, help='If True, queue candidate replies for human approval instead of auto-posting', default=False)
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
    while True:
        authenticate()
        user_id = getid()
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

                # If a search term was provided on the CLI, run the search pipeline each cycle.
                # The search pipeline honors dry-run, human approval, dedupe and daily caps.
                if getattr(args, 'search_term', None):
                    try:
                        print(f"[Main] Running search for term: {args.search_term}")
                        fetch_and_process_search(args.search_term, user_id=user_id)
                    except Exception as e:
                        print(f"[Main] Search error: {e}")

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

