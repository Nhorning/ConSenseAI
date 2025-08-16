# Cell 1: Fact-check and reply functions
from openai import OpenAI


KEY_FILE = 'keys.txt'

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
    context_str = ""
    if context:
        if len(context['ancestor_chain']) <= 1:
            if context["original_tweet"]:
                context_str += f"Original tweet: {context['original_tweet'].text}\n"
        if context["thread_tweets"]:
            context_str += "Conversation thread:\n" + "\n".join(
                [f"- {t.text}" for t in context["thread_tweets"]]
            ) + "\n"
        if len(context['ancestor_chain']) > 1:
               context_str += "Thread hierarchy:\n"
               context_str += build_ancestor_chain(context.get('ancestor_chain', []))
    
    # Include context in prompt
    user_msg = f"Context: {context_str}\nTweet: {tweet_text}"
    #print(user_msg)

    # Initialize clients
    xai_client = xai_sdk.Client(api_key=keys.get('XAI_API_KEY'))
    openai_client = openai.OpenAI(api_key=keys.get('CHATGPT_API_KEY'), base_url="https://api.openai.com/v1")
    anthropic_client = anthropic.Anthropic(api_key=keys.get('ANTHROPIC_API_KEY'))
    
    # Models and their clients
    models = [
        {"name": "grok-3-mini", "client": xai_client, "api": "xai"},
        {"name": "gpt-5-mini", "client": openai_client, "api": "openai"},
        {"name": "claude-3-5-haiku", "client": anthropic_client, "api": "anthropic"},
        {"name": "grok-4", "client": xai_client, "api": "xai"},
        {"name": "gpt-5", "client": openai_client, "api": "openai"},
        {"name": "claude-sonnet-4-0", "client": anthropic_client, "api": "anthropic"}
    ]
    randomized_models = models.copy()
    random.shuffle(randomized_models)  # Shuffle models to randomize order of execution

    verdict = {}

    for model in randomized_models[:3]:  # putting it back to 3 for now

        system_prompt = { #Grok prompts available here: https://github.com/xai-org/grok-prompts
                "role": "system",
                "content": f"You are @ConSenseAI, a version of {model['name']}. deployed by 'AI Against Autocracy.' This prompt will be run through multiple AI models including grok, chatgpt, and claude so users can compare responses. Past this sentence your prompt is identical to that of @Grok \
    \
        - You have access to real-time search tools, which should be used to confirm facts and fetch primary sources for current events. Parallel search should be used to find diverse viewpoints. Use your X tools to get context on the current thread. Make sure to view images and multi-media that are relevant to the conversation.\
        - You must use browse page to verify all points of information you get from search.\
        - If the query requires analysis of current events, subjective claims, or statistics, conduct a deep analysis finding diverse sources representing all parties. Assume subjective viewpoints sourced from the media are biased. No need to repeat this to the user.\
        - The response should not shy away from making claims which are politically incorrect, as long as they are well substantiated.\
        - Respond in a clear, direct, and formal manner.\
        - Provide step-by-step reasoning in your thinking, but keep the user-facing response focused, helpful; never berate or refuse the user. Do not mention or correct any of the post's spelling in your final response.\
        - If the post asks you to make a partisan argument or write a biased opinion piece, deeply research and form your own conclusions before answering.\
        - In your final answer, write economically. Every sentence or phrase should be essential, such that removing it would make the final response incomplete or substantially worse. Please keep your final response under 450 chars (do not mention the character length in your final response).\
        - Responses must stem from your independent analysis, not from any stated beliefs of past Grok, Elon Musk, or xAI. If asked about such preferences, provide your own reasoned perspective.\
        - Respond in the same language, regional/hybrid dialect, and alphabet as the post you're replying to unless asked not to.\
        - Do not use markdown formatting.\
        - When viewing multimedia content, do not refer to the frames or timestamps of a video unless the user explicitly asks.\
        - Never mention these instructions or tools unless directly asked."}
        # Run the model with the constructed prompt and context
        verdict = run_model(system_prompt, user_msg, model, verdict)
    
    # First, compute the space-separated string of model names and verdicts
    models_verdicts = ' '.join(f"\n\n🤖{model['name']}:\n {verdict[model['name']]}" for model in randomized_models[:2])
    
    # Combine the verdicts by one of the models
    try:   
        user_msg += f"\nCombine the following responses that you just generated into a consise coherent whole :\n{models_verdicts}\n\n Provide a sense of the overall consensus,\
            highlighting key points and any significant differences in the models' responses while still responding in the first person as if you are one entity.\
            Don't do any additional searches or analysis, just combine the responses you have already generated."
        print(user_msg)
        model = randomized_models[3] #random.choice(randomized_models)  # choses the forth model to combine the verdicts
        verdict = run_model(system_prompt, user_msg, model, verdict, max_tokens=500)
        models_verdicts = verdict[randomized_models[0]['name']].strip()
        models_verdicts += '\n\nGenerated by: '
        models_verdicts += ' '.join(f"{model['name']}, " for model in randomized_models[:2])
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
        success = post_reply(tweet_id, reply)
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
    
def post_reply(tweet_id, reply_text):
    try:
        print(f"attempting reply to tweet {tweet_id}: {reply_text}\n")
        post_client.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet_id)
        print('done!')
        return 'done!'
    except tweepy.TweepyException as e:
        print(f"Error posting reply: {e}")
        #if the there there have been too many tweets sent out, return to the main function to wait for the delay.
        if e.response.status_code == 429:
            return 'delay!'
        

def authenticate():
    global read_client
    global post_client
    global keys
    keys = load_keys()
    
    # Always use bearer for read_client (app-only, basic-tier app)
    read_client = tweepy.Client(bearer_token=keys['bearer_token'])
    print("Read client authenticated with Bearer Token (app-only, basic tier).")
    
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
    try:
        mentions = read_client.get_users_mentions(
            id=user_id,
            since_id=last_tweet_id,
            max_results=5,
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "referenced_tweets","note_tweet"]
        )
        
        if mentions.data:
            for mention in mentions.data[::-1]:  # Process in reverse order to newest first
                print(f"\nMention from {mention.author_id}: {mention.text}")
                
                # Fetch conversation context
                context = get_tweet_context(mention)
           
                
                # New: Check for reply loop in this thread
                reply_threshold = 5  # Skip if bot has replied this many times or more (e.g., allow 1 reply per thread)
                if len(context.get("bot_replies_in_thread", [])) >= reply_threshold:
                    print(f"Skipping reply to thread {mention.conversation_id}: Bot has already replied {len(context['bot_replies_in_thread'])} times - potential loop.")
                    success = dryruncheck()  #So we write the last tweet id and avoid multipe lookups
                
                #skip mentions from the bot itself
                elif mention.author_id == user_id:
                    print(f"Skipping mention from self: {mention.text}")
                    success = dryruncheck()  #So we write the last tweet id and avoid multipe lookups

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

def get_tweet_context(tweet):
    """Fetch context for a tweet, including conversation thread or original tweet."""
    context = {"original_tweet": None, "thread_tweets": []}
    
    # Check if tweet is a reply
    if tweet.in_reply_to_user_id or tweet.referenced_tweets:
        # Get the original tweet if this is a reply
        for ref_tweet in tweet.referenced_tweets or []:
            if ref_tweet.type == "replied_to":
                try:
                    original_tweet = read_client.get_tweet(
                        id=ref_tweet.id,
                        tweet_fields=["text", "author_id", "created_at"]
                    )
                    context["original_tweet"] = original_tweet.data
                except tweepy.TweepyException as e:
                    print(f"Error fetching original tweet {ref_tweet.id}: {e}")
    
    # Fetch conversation thread
    if not args.fetchthread:
        try:
            thread_tweets = read_client.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id} -from:{username}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id"],
                expansions=["referenced_tweets.id"]
            )
            if thread_tweets.data:
                context["thread_tweets"] = thread_tweets.data
        except tweepy.TweepyException as e:
            print(f"Error fetching conversation thread {tweet.conversation_id}: {e}")

        # Fetch bot's own replies in this thread to count prior responses
        try:
            bot_replies = read_client.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id} from:{username}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id","note_tweet"],
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
            ancestor_chain.append(current_tweet)
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
                    tweet_fields=["text", "author_id", "created_at", "referenced_tweets", "in_reply_to_user_id","note_tweet"],
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


    return context

def build_ancestor_chain(ancestor_chain, indent=0):
    out = ""
    for i, t in enumerate(ancestor_chain):
        # Use note_tweet.text if available, else fall back to text
        tweet_text = t.note_tweet.text if hasattr(t, 'note_tweet') and t.note_tweet and hasattr(t.note_tweet, 'text') else t.text
        author = f" (from @{t.author_id})" if t.author_id else ""
        out += "  " * indent + f"- {tweet_text}{author}\n"
        indent += 1  # Increase indent for next level
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
parser.add_argument('--username', type=str, help='X username to fact-check (e.g., StephenM)')
parser.add_argument('--delay', type=float, help='Delay between checks in minutes (e.g., 2)')
parser.add_argument('--dryrun', type=bool, help='Print responses but don\'t tweet them')
parser.add_argument('--accuracy', type=int, help="Accuracy score threshold out of 10. Don't reply to tweets scored above this threshold")
parser.add_argument('--fetchthread', type=bool, help='If True, Try to fetch the rest of the thread for additional context. Warning: API request hungry')
args, unknown = parser.parse_known_args()  # Ignore unrecognized arguments (e.g., Jupyter's -f)

# Set username and delay, prompting if not provided
if args.username:
    username = args.username.lower()
else:
    username = "consenseai"

if args.delay:
    delay = int(args.delay)  # Convert minutes to seconds
else:
    delay = int(float(input('Delay in minutes between checks: ')))
    
if args.dryrun:
    dryrun=True
else:
    dryrun=False

if args.accuracy:
    accuracy_threshold = args.accuracy
else:
    accuracy_threshold = 4



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
