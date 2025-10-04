import json
# Cell 1: Fact-check and reply functions
from openai import OpenAI
import json
import os
import webbrowser as web_open

import sys

class LoggerWriter:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'a')  # Append mode for continuous logging

    def _rotate_log(self):
        '''Rotate log file if it exceeds size limit.'''
        if os.path.exists(self.filename):
            size_mb = os.path.getsize(self.filename) / (1024 * 1024)
            if size_mb > LOG_MAX_SIZE_MB:
                for i in range(LOG_MAX_ROTATIONS - 1, 0, -1):
                    old_name = f"{self.filename}.{i}"
                    new_name = f"{self.filename}.{i+1}"
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)
                if os.path.exists(self.filename):
                    os.rename(self.filename, f"{self.filename}.1")
                print(f"[Log Rotation] Rotated {self.filename} due to size > {LOG_MAX_SIZE_MB}MB")

    def write(self, text):
        self._rotate_log()  # Check rotation before writing
        self.file.write(text)
        sys.__stdout__.write(text)  # Mirror to console

    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()

sys.stdout = LoggerWriter('output.log')

# Also redirect stderr for error logging
sys.stderr = LoggerWriter('output.log')


KEY_FILE = 'keys.txt'
LOG_MAX_SIZE_MB = 10  # Rotate log if larger than this (MB)
LOG_MAX_ROTATIONS = 5  # Keep this many rotated log files
MAX_BOT_TWEETS = 1000  # Max entries in bot_tweets.json
MAX_ANCESTOR_CHAINS = 500  # Max conversations in ancestor_chains.json"
TWEETS_FILE = 'bot_tweets.json'  # File to store bot's tweets
ANCESTOR_CHAIN_FILE = 'ancestor_chains.json'

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
    Keys are conversation_id (as strings) -> list of entries saved by save_ancestor_chain().
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

def save_ancestor_chain(conversation_id, chain):
    chains = load_ancestor_chains()
    # Convert chain to serializable format
    serializable_chain = []
    for entry in chain:
        tweet_dict = tweet_to_dict(entry["tweet"])
        quoted_dicts = [tweet_to_dict(qt) for qt in entry["quoted_tweets"]]
        # Extract and cache media for this tweet
        media = extract_media(entry["tweet"])
        serializable_chain.append({"tweet": tweet_dict, "quoted_tweets": quoted_dicts, "media": media})
    chains[str(conversation_id)] = serializable_chain
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
    chain = chains.get(cid)
    if chain:
        for entry in chain:
            # entry was saved as {'tweet': tweet_dict, 'quoted_tweets': [...], 'media': [...]}
            t = entry.get('tweet', {})
            tid = str(t.get('id')) if isinstance(t, dict) and t.get('id') is not None else None
            author = str(t.get('author_id')) if isinstance(t, dict) and t.get('author_id') is not None else None
            if tid and tid in bot_tweets:
                count += 1
            elif author and bot_user_id and str(author) == str(bot_user_id):
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
from xai_sdk.chat import system, user
from xai_sdk.chat import system, user, SearchParameters
import anthropic
import re
import tweepy
from datetime import datetime
import timeout_decorator
import random

def run_model(system_prompt, user_msg, model, verdict, max_tokens=250):
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
                chat.append(user(user_msg))
                response = chat.sample()
                verdict[model['name']] = response.content.strip()
                if hasattr(response, 'usage') and response.usage is not None and hasattr(response.usage, 'num_sources_used'):
                    print(f"{model['name']} sources used: {response.usage.num_sources_used}")
                else:
                    print(f"{model['name']} sources used: Not available")
            elif model['api'] == "openai":
                # OpenAI SDK call
                response = model['client'].chat.completions.create(
                    model=model['name'],
                    messages=[
                        system_prompt,
                        {"role": "user", "content": user_msg}
                    ],
                    #max_tokens=max_tokens,
                )
                verdict[model['name']] = response.choices[0].message.content.strip()
            elif model['api'] == "anthropic":
                # Anthropic SDK call
                response = model['client'].messages.create(
                    model=model['name'],
                    system=system_prompt['content'],
                    messages=[
                        {"role": "user", "content": user_msg}
                    ],
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

def fact_check(tweet_text, tweet_id, context=None):
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
                context_str += f"Quoted tweet by @{qt.author_id}: {get_full_text(qt)}\n"
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
    media_str = format_media(context.get('media', [])) if context else ""
    user_msg = f"Context:\n {context_str}\n{media_str}Tweet: {full_mention_text}"
    #print(user_msg)

    # Initialize clients
    xai_client = xai_sdk.Client(api_key=keys.get('XAI_API_KEY'))
    openai_client = openai.OpenAI(api_key=keys.get('CHATGPT_API_KEY'), base_url="https://api.openai.com/v1")
    anthropic_client = anthropic.Anthropic(api_key=keys.get('ANTHROPIC_API_KEY'))
    
    # Models and their clients
    models = [
        #lower tier (index 0-2)
        {"name": "grok-4-fast-reasoning", "client": xai_client, "api": "xai"},
        {"name": "gpt-5-mini", "client": openai_client, "api": "openai"},
        {"name": "claude-3-5-haiku-latest", "client": anthropic_client, "api": "anthropic"},
        #higher tier (index 3-5)
        {"name": "grok-4", "client": xai_client, "api": "xai"},
        {"name": "gpt-5", "client": openai_client, "api": "openai"},
        {"name": "claude-sonnet-4-5", "client": anthropic_client, "api": "anthropic"}
    ]
    randomized_models = models[:3].copy() # slice is to only use lower tier for first pass
    random.shuffle(randomized_models)  # Shuffle models to randomize order of execution

    verdict = {}

    runs = 3
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
        if model['api'] == 'openai-vision' and context.get('media'):
            user_msg_with_media = user_msg + "\nMedia URLs:\n" + "\n".join([m['url'] for m in context['media'] if isinstance(m, dict) and m.get('url')])
            verdict = run_model(system_prompt, user_msg_with_media, model, verdict)
        else:
            verdict = run_model(system_prompt, user_msg, model, verdict)
    
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
        verdict = run_model(system_prompt, user_msg, model, verdict, max_tokens=500)
        
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
    #if len(reply) > 280:  # Twitter’s character limit
    #    reply = f"AutoGrok AI Fact-check v1: {initial_answer[:30]}... {search_summary[:150]}... {grok_prompt[:100]}..."

    # Post reply checks are passed
    '''if 'not a factual claim' in reply.lower() or accuracy_score == 'N/A':
        print(f'No claim detected. Not tweeting:\n{reply}')
        success = dryruncheck()
    elif accuracy_score > accuracy_threshold:
        print(f'Accuracy above threshold ({accuracy_threshold}), Not Tweeting:\n {reply} ')
        success = dryruncheck()
    elif 'satire' in verdict.lower():
        print(f'Satire detected, not tweeting:\n{reply}')
        success = dryruncheck()'''
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
    chain = chains.get(cid, [])
    chain.append(entry)
    chains[cid] = chain
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
                print(f"Warning: Existing tokens authenticate as {user.data.username}, not @ConSenseAI. Proceeding with new authentication.")
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
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "referenced_tweets"]
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
                context = get_tweet_context(mention)
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

def get_tweet_context(tweet):
    # Ancestor chain cache file
    ANCESTOR_CHAIN_FILE = 'ancestor_chains.json'

    def load_ancestor_chain(conversation_id):
        if os.path.exists(ANCESTOR_CHAIN_FILE):
            try:
                with open(ANCESTOR_CHAIN_FILE, 'r') as f:
                    chains = json.load(f)
                    return chains.get(str(conversation_id))
            except Exception as e:
                print(f"Error loading ancestor chain cache: {e}")
        return None

    def save_ancestor_chain(conversation_id, chain):
        chains = {}
        if os.path.exists(ANCESTOR_CHAIN_FILE):
            try:
                with open(ANCESTOR_CHAIN_FILE, 'r') as f:
                    chains = json.load(f)
            except Exception as e:
                print(f"Error reading ancestor chain cache: {e}")
        # Convert chain to serializable format
        serializable_chain = []
        for entry in chain:
            tweet_dict = tweet_to_dict(entry["tweet"])
            quoted_dicts = [tweet_to_dict(qt) for qt in entry["quoted_tweets"]]
            # Extract and cache media for this tweet
            media = extract_media(entry["tweet"])
            serializable_chain.append({"tweet": tweet_dict, "quoted_tweets": quoted_dicts, "media": media})
        chains[str(conversation_id)] = serializable_chain
        if len(chains) > MAX_ANCESTOR_CHAINS:
            # Remove oldest conversations (keep most recent)
            sorted_chains = sorted(chains.items(), key=lambda x: x[0], reverse=True)
            chains = dict(sorted_chains[:MAX_ANCESTOR_CHAINS])
        try:
            with open(ANCESTOR_CHAIN_FILE, 'w') as f:
                json.dump(chains, f, indent=2)
        except Exception as e:
            print(f"Error saving ancestor chain cache: {e}")

def get_tweet_context(tweet):
    """Fetch context for a tweet, including conversation thread or original tweet."""
    context = {"original_tweet": None, "thread_tweets": [], "quoted_tweets": [], "conversation_id": getattr(tweet, 'conversation_id', None)}
    
    # Check if tweet is a reply
    def collect_quoted(refs):
        quoted = []
        for ref_tweet in refs or []:
            if ref_tweet.type == "quoted":
                try:
                    quoted_tweet = read_client.get_tweet(
                        id=ref_tweet.id,
                        tweet_fields=["text", "author_id", "created_at"]
                    )
                    print(f"[DEBUG] Quoted tweet {ref_tweet.id} text length: {len(quoted_tweet.data.text)} chars")
                    quoted.append(quoted_tweet.data)
                except tweepy.TweepyException as e:
                    print(f"Error fetching quoted tweet {ref_tweet.id}: {e}")
        return quoted

    # Always collect quoted tweets in the mention
    context["quoted_tweets"].extend(collect_quoted(getattr(tweet, 'referenced_tweets', None)))

    # If this is a reply, fetch the original tweet and collect its quoted tweets
    original_tweet_obj = None
    if tweet.in_reply_to_user_id or tweet.referenced_tweets:
        for ref_tweet in tweet.referenced_tweets or []:
            if ref_tweet.type == "replied_to":
                try:
                    original_tweet = read_client.get_tweet(
                        id=ref_tweet.id,
                        tweet_fields=["text", "author_id", "created_at", "referenced_tweets"]
                    )
                    original_tweet_obj = original_tweet.data
                    print(f"[DEBUG] Original tweet {ref_tweet.id} text length: {len(original_tweet_obj.text)} chars")
                    context["original_tweet"] = original_tweet_obj
                    # Collect quoted tweets from the original tweet
                    context["quoted_tweets"].extend(collect_quoted(getattr(original_tweet_obj, 'referenced_tweets', None)))
                except tweepy.TweepyException as e:
                    print(f"Error fetching original tweet {ref_tweet.id}: {e}")

    # Try to build ancestor chain and collect quoted tweets, but don't block above
    ancestor_chain = load_ancestor_chains().get(str(getattr(tweet, 'conversation_id', None)), [])  # Use plural function and get chain for this ID
    if ancestor_chain:
        print(f"Loaded ancestor chain from cache for conversation {getattr(tweet, 'conversation_id', None)}")
        context['ancestor_chain'] = ancestor_chain
    else:
        ancestor_chain = []
        current_tweet = tweet
        visited = set()
        try:
            while True:
                quoted = collect_quoted(getattr(current_tweet, 'referenced_tweets', None))
                ancestor_chain.append({"tweet": current_tweet, "quoted_tweets": quoted})
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
                    tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id"],
                    expansions=["referenced_tweets.id"]
                )
                if parent_response.data:
                    current_tweet = parent_response.data
                else:
                    break
            ancestor_chain = ancestor_chain[::-1]
        except tweepy.TweepyException as e:
            print(f"Error building ancestor chain: {e}")
        context['ancestor_chain'] = ancestor_chain
        # Save to cache
        save_ancestor_chain(getattr(tweet, 'conversation_id', None), ancestor_chain)
    
    # Fetch conversation thread
    if args.fetchthread == True:
        try:
            thread_tweets = read_client.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id} -from:{username}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id"],
                expansions=["referenced_tweets.id"]
            )
            if thread_tweets.data:
                context["thread_tweets"] = thread_tweets.data

                # New: Concatenate full thread text (only from original author to avoid noise)
                original_author_id = getattr(tweet, 'author_id', None)  # Or from context["original_tweet"]
                thread_texts = []
                # Sort by created_at to maintain order (oldest first)
                sorted_tweets = sorted(thread_tweets.data, key=lambda t: t.created_at)
                for tt in sorted_tweets:
                    if str(tt.author_id) == str(original_author_id):  # Only include original author's tweets
                        thread_texts.append(tt.text)
                context["full_thread_text"] = " ".join(thread_texts)  # Concatenated full text
            else:
                context["full_thread_text"] = get_full_text(tweet)  # Fallback to single tweet
        except tweepy.TweepyException as e:
            print(f"Error fetching conversation thread {tweet.conversation_id}: {e}")

        # Fetch bot's own replies in this thread to count prior responses
        try:
            bot_replies = read_client.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id} from:{username}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id"],
                expansions=["referenced_tweets.id"]
            )
            if bot_replies.data:
                context["bot_replies_in_thread"] = bot_replies.data
        except tweepy.TweepyException as e:
            print(f"Error fetching bot's replies in thread {tweet.conversation_id}: {e}")
        
        # New: Build ancestor chain from mention to root
        ancestor_chain = []
        current_tweet = tweet
        visited = set()  # Avoid cycles
        while True:
            quoted = collect_quoted(getattr(current_tweet, 'referenced_tweets', None))
            ancestor_chain.append({"tweet": current_tweet, "quoted_tweets": quoted})
            visited.add(current_tweet.id)
            parent_id = None
            if hasattr(current_tweet, 'referenced_tweets') and current_tweet.referenced_tweets:
                for ref in current_tweet.referenced_tweets:
                    if ref.type == 'replied_to':
                        parent_id = ref.id
                        break
            if parent_id is None or parent_id in visited:
                break  # No parent or cycle detected
            try:
                parent_response = read_client.get_tweet(
                    id=parent_id,
                    tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id"],
                    expansions=["referenced_tweets.id"]
                )
                if parent_response.data:
                    current_tweet = parent_response.data
                else:
                    break
            except tweepy.TweepyException as e:
                print(f"Error fetching parent {parent_id}: {e}")
                break
        
        # Reverse to root-first order
        ancestor_chain = ancestor_chain[::-1]
        context['ancestor_chain'] = ancestor_chain

    # For loop detection, keep context["bot_replies_in_thread"] as before
    context["bot_replies_in_thread"] = bot_replies.data or []


    # Extract and add media information from every tweet in the ancestor chain (prefer cached if present)
    all_media = []
    # Media from the mention itself
    all_media.extend(extract_media(tweet))
    # Media from ancestor chain tweets
    for entry in context.get('ancestor_chain', []):
        # Use cached media if present, else extract
        if 'media' in entry:
            all_media.extend(entry['media'])
        else:
            t = entry.get('tweet')
            all_media.extend(extract_media(t))
    context['media'] = all_media

    return context

def build_ancestor_chain(ancestor_chain, indent=0):
    out = ""
    def get_full_text(t):
        return t.text if hasattr(t, 'text') else t.get('text', '')

    for i, entry in enumerate(ancestor_chain):
        t = entry["tweet"]
        quoted_tweets = entry["quoted_tweets"]
        # Support Tweepy objects and cached dicts
        tweet_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        author_id = t.get("author_id", "") if isinstance(t, dict) else getattr(t, "author_id", "")
        tweet_text = get_full_text(t)
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

def extract_media(t):
    media_list = []
    found_media = False
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
    elif hasattr(t, 'attachments') and hasattr(t.attachments, 'media_keys'):
        # Try to get media from 'includes' if present
        includes = getattr(t, 'includes', None)
        if includes and hasattr(includes, 'media'):
            for m in includes.media:
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
    if found_media:
        print('Found media, appending to context.')
    return media_list

def format_media(media_list):
    if not media_list:
        return ""
    out = "Media attached to tweet(s):\n"
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

RESTART_DELAY = 10
backoff_multiplier = 1

# The main loop
def main():
    while True:
        authenticate()
        user_id = getid()
        try:
            while True:
                fetch_and_process_mentions(user_id, username)  # Changed from fetch_and_process_tweets
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




