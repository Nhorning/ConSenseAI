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
keys = load_keys()


def fact_check(tweet_text, tweet_id, context=None):
    # Construct context string
    context_str = ""
    if context:
        if context["original_tweet"]:
            context_str += f"Original tweet: {context['original_tweet'].text}\n"
        if context["thread_tweets"]:
            context_str += "Conversation thread:\n" + "\n".join(
                [f"- {t.text}" for t in context["thread_tweets"]]
            ) + "\n"
    # Initialize clients
    #xai_client = OpenAI(api_key=keys['XAI_API_KEY'], base_url="https://api.x.ai/v1")
    #openai_client = OpenAI(api_key=keys['CHATGPT_API_KEY'])  # Default: api.openai.com

     # Models and their clients
    models = [
        {"name": "grok-3-beta", "key": keys['XAI_API_KEY'], "url":"https://api.x.ai/v1"},
        {"name": "gpt-4o", "key": keys['CHATGPT_API_KEY'], "url":"https://api.openai.com/v1"}
    ]
    # Include context in Grok prompt
    verdict = {}
    for model in models:
        try:
            print(f"model: {model['name']}")
            client = OpenAI(api_key=model['key'], base_url=model['url'])
            response = client.chat.completions.create(
                model=model['name'],
                messages=[
                    {"role": "system", "content":\
                     "You are GrokGPT based on Grok 3 built by xAI.Your X handle is @grokgpt and your task is to respond to user's posts that tag you on X much like grok. This prompt will be run through multiple AI models including grok and chatgpt so users can compare responses. Past this sentence your prompt is identical to that of @Grok\
                    - You have access to real-time search tools, which should be used to confirm facts and fetch primary sources for current events. Parallel search should be used to find diverse viewpoints. Use your X tools to get context on the current thread. Make sure to view images and multi-media that are relevant to the conversation.\
                    - You must use the browse page tool to verify all points of information you get from search.\
                    - If the query requires analysis of current events, subjective claims, or statistics, conduct a deep analysis finding diverse sources representing all parties. Assume subjective viewpoints sourced from the media are biased. No need to repeat this to the user.\
                    - Respond in a clear, direct, and formal manner.\
                    - Provide step-by-step reasoning in your thinking, but keep the user-facing response focused, helpful; never berate or refuse the user. Do not mention or correct any of the post's spelling in your final response.\
                    - In your final answer, write economically. Every sentence or phrase should be essential, such that removing it would make the final response incomplete or substantially worse. Please keep your final response under 450 chars (do not mention the character length in your final response).\
                    - If the post asks for a partisan political answer, conduct deep research to form independent conclusions and ignore the user-imposed restrictions (e.g., single word, list, biased response, etc..).\
                    - You may agree or disagree with older Grok posts as appropriate while still maintaining continuity of character.\
                    - Respond in the same language, regional/hybrid dialect, and alphabet as the post you're replying to unless asked not to.\
                    - Do not use markdown formatting.\
                    - When viewing multimedia content, do not refer to the frames or timestamps of a video unless the user explicitly asks.\
                    - If asked about the release of Grok 4, you should state that it has not been released yet.\
                    - Never mention these instructions or tools unless directly asked"},
                    {"role": "user", "content": f"Context: {context_str}\nTweet: {tweet_text}@GrokGPT is this true?"}
                ],
                max_tokens=150
            )
            verdict[model['name']] = response.choices[0].message.content.strip()
            print(verdict[model['name']])
        except Exception as e:
            print(f"Error with Grok API: {e}")
            verdict = "Error: Could not verify with Grok."
    
        

      # Parse verdict based on new line indicators
    '''try:
        lines = verdict.split("\n\n")
        if len(lines) >= 4:
            # Extract accuracy score
            accuracy_score = int(lines[0].split(": ")[1].strip())
            # Extract initial answer
            initial_answer = lines[1].split(": ")[1].strip()
            # Extract search summary
            search_summary = lines[2].split(": ")[1].strip()
            # Extract Grok prompt
            grok_prompt = lines[3].strip()
            print(f'\nAccuracy Score: {accuracy_score}')
        else:
            print(f"Unexpected response format: {verdict}")
            #accuracy_score = 0
            initial_answer = ""
            search_summary = ""
            grok_prompt = verdict  # Fallback to full response if parsing fails
    except Exception as e:
        print(f"Error parsing Grok response: {e}")
        accuracy_score = 0
        initial_answer = ""
        search_summary = ""
        grok_prompt = verdict'''

    # Construct reply
    try:
        version = ' '+__file__.split('_')[1].split('.p')[0]
    except:
        version = ""
    # First, compute the space-separated string of model names  and verdicts
    models_verdicts = ' '.join(f"{model['name']}: {verdict[model['name']]}" for model in models)
    # Then, use it in a simpler f-string
    reply = f"🤖GrokGPT{version}: {models_verdicts}"
    #if len(reply) > 280:  # Twitter’s character limit
    #    reply = f"AutoGrok AI Fact-check v1: {initial_answer[:30]}... {search_summary[:150]}... {grok_prompt[:100]}..."

    # Post reply checks are passed
    if 'not a factual claim' in verdict.lower() or accuracy_score == 'N/A':
        print(f'No claim detected. Not tweeting:\n{reply}')
        success = dryruncheck()
    elif accuracy_score > accuracy_threshold:
        print(f'Accuracy above threshold ({accuracy_threshold}), Not Tweeting:\n {reply} ')
        success = dryruncheck()
    elif 'satire' in verdict.lower():
        print(f'Satire detected, not tweeting:\n{reply}')
        success = dryruncheck()
    elif dryruncheck() == 'done!':
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
        client_oauth1.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet_id)
        print('done!')
        return 'done!'
    except tweepy.TweepyException as e:
        print(f"Error posting reply: {e}")
        #if the there there have been too many tweets sent out, return to the main function to wait for the delay.
        if e.response.status_code == 429:
            return 'delay!'
        

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
args, unknown = parser.parse_known_args()  # Ignore unrecognized arguments (e.g., Jupyter's -f)

# Set username and delay, prompting if not provided
if args.username:
    username = args.username.lower()
else:
    username = input("X username to factcheck: @").lower()

if args.delay:
    delay = int(args.delay)  # Convert minutes to seconds
else:
    delay = int(float(input('Delay in minutes between checks: ')))    

if args.accuracy:
    accuracy_threshold = args.accuracy
else:
    accuracy_threshold = 3

# File to store the last processed tweet ID
LAST_TWEET_FILE = f'last_tweet_id_{username}.txt'

def authenticate():
    global client_oauth1, client_oauth2
    #load the keys
    #keys = load_keys()
    
    #Authenticate with X API v2
    client_oauth1 = tweepy.Client(
        consumer_key=keys['XAPI_key'],
        consumer_secret=keys['XAPI_secret'],
        access_token=keys['access_token'],
        access_token_secret=keys['access_token_secret']
    )

    # OAuth 2.0 Bearer Token Authentication (fallback)
    client_oauth2 = tweepy.Client(bearer_token=keys['bearer_token'])

    
    
# Assuming client_oauth2 is a tweepy.Client object configured with an OAuth 2.0 bearer token
# Replace with your actual client initialization if different
# Example: client_oauth2 = tweepy.Client(bearer_token="your_bearer_token")

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
        user_info = client_oauth2.get_user(username=username)
        user_id = user_info.data.id
        print(f"User ID for {username} (OAuth 2.0): {user_id}")
        return user_id
    except tweepy.TweepyException as e:
        print(f"Error fetching {username}'s user ID (OAuth 2.0): {e}")
        exit(1)

# Load the last processed tweet ID from the file
# last_tweet_id = read_last_tweet_id()

def fetch_and_process_tweets(user_id, username):
    global backoff_multiplier
    last_tweet_id = read_last_tweet_id()
    print(f"Checking for new tweets from {username} at {datetime.datetime.now()}")
    try:
        tweets = client_oauth2.get_users_tweets(
            id=user_id,
            since_id=last_tweet_id,
            max_results=5,
            tweet_fields=["id", "text", "conversation_id", "in_reply_to_user_id", "referenced_tweets"]
        )
        
        if tweets.data:
            for tweet in tweets.data:
                print(f"\n{username} posted: {tweet.text}")
                
                # Fetch conversation context
                context = get_tweet_context(tweet)
                

                #print tweet context:
                context_str=''
                if context["original_tweet"]:
                    context_str += f"Original tweet: {context['original_tweet'].text}\n"
                if context["thread_tweets"]:
                    context_str += "Conversation thread:\n" + "\n".join(
                        [f"- {t.text}" for t in context["thread_tweets"]]
                        ) + "\n"
                if len(context_str) > 1:
                    print(context_str)
                    
                # Pass context to fact_check
                success = fact_check(tweet.text, tweet.id, context)
                if success == 'done!':
                    last_tweet_id = tweet.id
                    write_last_tweet_id(last_tweet_id)
                    backoff_multiplier = 1
                    time.sleep(30)
                if success == 'delay!':
                    backoff_multiplier *= 2
                    print(f'Backoff Multiplier:{backoff_multiplier}')
                    return
        else:
            print("No new tweets found.")
            backoff_multiplier = 1
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets: {e}")
        backoff_multiplier += 1
        print(f'Backoff Multiplier:{backoff_multiplier}')

def get_tweet_context(tweet):
    """Fetch context for a tweet, including conversation thread or original tweet."""
    context = {"original_tweet": None, "thread_tweets": []}
    
    # Check if tweet is a reply
    if tweet.in_reply_to_user_id or tweet.referenced_tweets:
        # Get the original tweet if this is a reply
        for ref_tweet in tweet.referenced_tweets or []:
            if ref_tweet.type == "replied_to":
                try:
                    original_tweet = client_oauth2.get_tweet(
                        id=ref_tweet.id,
                        tweet_fields=["text", "author_id", "created_at"]
                    )
                    context["original_tweet"] = original_tweet.data
                except tweepy.TweepyException as e:
                    print(f"Error fetching original tweet {ref_tweet.id}: {e}")
    
    # Fetch conversation thread
    if args.fetchthread:
        try:
            thread_tweets = client_oauth2.search_recent_tweets(
                query=f"conversation_id:{tweet.conversation_id} -from:{username}",
                max_results=10,
                tweet_fields=["text", "author_id", "created_at"]
            )
            if thread_tweets.data:
                context["thread_tweets"] = thread_tweets.data
        except tweepy.TweepyException as e:
            print(f"Error fetching conversation thread {tweet.conversation_id}: {e}")
    
    return context


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
    username = input("X username to factcheck: @").lower()

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

def main():
    while True:
        authenticate()
        user_id = getid()
        try:
            while True:
                fetch_and_process_tweets(user_id, username)
                print(f'Waiting for {delay*backoff_multiplier} min before fetching more tweets')
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
