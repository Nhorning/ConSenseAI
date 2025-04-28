# Cell 1: Fact-check and reply functions
from openai import OpenAI

KEY_FILE = 'keys.txt'

def load_keys():
    """Load keys from the key file. Format:
    consumer_key=V0q_your_consumer_key_Fvn
    consumer_secret=m76B6_your_consumer_secret_fHuAXstAmxHwuEx84G9nHAO
    access_token=48365055_your_access_token_Py4EEaZ9K5AXd0LfAWVbNP4
    access_token_secret=x__your_access_toke_secret_T2xmh5TPTHaIhxJpb9
    bearer_token=AAAAAAAAA_your_bearer_token_AAPXkoDocWVyM32XlaMoS3pPxIZnk%3D3MwUD37WflW3OeCANdzHAaaNERieJsFQl8ibqDyABX919C9Ly4
    XAI_API_KEY=your_xai_api_key
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
client = OpenAI(api_key=keys['XAI_API_KEY'], base_url="https://api.x.ai/v1")

def fact_check(tweet_text, tweet_id):
    # Ask Grok for initial answer, web search, and fact-check prompt
    try:
        response = client.chat.completions.create(
            model="grok-3-beta",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant for X posts.\
                You are skeptical of authoritarianism. Your job is to verify claims, provide a brief initial answer,\
                include web search results via DeepSearch, and prompt @Grok for further verification.\
                Verify the content of the claim, not the fact that it was claimed.\
                When in doubt, prompt @Grok to rate the post for accuracy on a scale from 1 to 10.\
                Use DeepSearch to fetch and summarize web search results supporting your answer in 300 characters or less.\
                If you detect satire attempt a witty response. If there is absolutely no information to verify respond only\
                \"not a factual claim.\""},
                {"role": "user", "content": "Friendly reminder that the courts created a Constitutional crisis, not Trump When they stop him from exercising his legal right to deport illegals, they infringe on the executive branch’s constitutional power"},
                {"role": "assistant", "content": "Initial answer: Courts have authority to review executive actions. Search: Courts can check executive power (Constitution). @grok is he correct?"},
                {"role": "user", "content": "If you were an American falsely accused of wrongdoing on January 6th it wasn’t merely difficult to get “due process,” it was impossible. The entire system was rigged against you. All of it. Those persecuted Americans could only dream of the “due process” afforded illegal aliens."},
                {"role": "assistant", "content": "Initial answer: Due process was available but challenging. DeepSearch Summary: Jan 6 defendants had trials (DOJ). @Grok is @StephenM lying here?"},
                {"role": "user", "content": "America is a CHRISTIAN nation!"},
                {"role": "assistant", "content": "Initial answer: U.S. has no official religion. DeepSearch Summary: First Amendment bans state religion (Constitution). @Grok is this true according to the constitution?"},
                {"role": "user", "content": "Good news: A documentary featuring Fatima Hassouna, a young, brave photojournalist in Gaza got accepted into Cannes Film Festival. Bad news: Within 24 hours of this news, Israel murdered her and ten family members, including her pregnant sister. What kind of monsters do this?"},
                {"role": "assistant", "content": "Initial answer: No evidence confirms this claim. Search: No reports verify Hassouna’s death (news). @Grok rate this post for accuracy on a scale from 1 to 10 please."},
                {"role": "user", "content": "The incompetence from the liberal media is astonishing."},
                {"role": "assistant", "content": "Initial answer: Opinion, not fact-based. DeepSearch Summary: Media bias varies (studies). @Grok rate the accuracy of what Miller says in the video on a one to 10 scale."},
                {"role": "user", "content": "Judicial coup"},
                {"role": "assistant", "content": "Initial answer: No evidence of a judicial coup. DeepSearch Summary: Judiciary acts within law (Constitution). @grok is this a judicial coup?"},
                {"role": "user", "content": "This is the hand of the man that the Democrats feel should be brought back to the United States, because he is such “a fine and innocent person.” They said he is not a member of MS-13, even though he’s got MS-13 tattooed onto his knuckles, and two Highly Respected Courts found"},
                {"role": "assistant", "content": "Initial answer: Tattoos don’t negate due process. Search: Due process applies to all (Constitution). @grok does the presence of a gang tattoo mean he is not entitled to due process?"},
                {"role": "user", "content": f"{tweet_text}\n\nIs this claim true? Provide a brief initial answer (30 characters or less). Include a DeepSearch web summary (300 characters or less). Prompt @Grok for verification."}
            ],
            max_tokens=150,
            #reasoning={"effort": "high"}  # Boost reasoning
        )
        verdict = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with Grok API: {e}")
        verdict = "Error: Could not verify with Grok."

      # Parse verdict based on new line indicators
    try:
        lines = verdict.split("\n\n")
        if len(lines) >= 3:
            # Extract initial answer
            initial_answer = lines[0].split(": ")[1].strip()
            # Extract search summary
            search_summary = lines[1].split(": ")[1].strip()
            # Extract Grok prompt
            grok_prompt = lines[2].strip()
        else:
            print(f"Unexpected response format: {verdict}")
            initial_answer = ""
            search_summary = ""
            grok_prompt = verdict  # Fallback to full response if parsing fails
    except Exception as e:
        print(f"Error parsing Grok response: {e}")
        initial_answer = ""
        search_summary = ""
        grok_prompt = verdict

    # Construct reply
    reply = f"AutoGrok AI Fact-check v1.3: {initial_answer} {search_summary} {grok_prompt}"
    #if len(reply) > 280:  # Twitter’s character limit
    #    reply = f"AutoGrok AI Fact-check v1: {initial_answer[:30]}... {search_summary[:150]}... {grok_prompt[:100]}..."

    # Post reply if factual claim
    if 'Not a factual claim.' not in verdict and 'satire' not in verdict and dryrun==False:
        success = post_reply(tweet_id, reply)
    else:
        print(f'Not tweeting: {reply}')
        success = 'fail'
    return success

def post_reply(tweet_id, reply_text):
    try:
        print(f"attempting reply to tweet {tweet_id}: {reply_text}\n")
        client_oauth1.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet_id)
        print('done!')
        return 'done!'
    except tweepy.TweepyException as e:
        print(f"Error posting reply: {e}")
        if e.response.status_code == 429:
            print(f'Sleeping for {delay/60} min to refresh requests')
            return 'delay!'
        
#Cell 2 get and reply to tweets

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
args, unknown = parser.parse_known_args()  # Ignore unrecognized arguments (e.g., Jupyter's -f)

# Set username and delay, prompting if not provided
if args.username:
    username = args.username.lower()
else:
    username = input("X username to factcheck: @").lower()

if args.delay:
    delay = int(args.delay * 60)  # Convert minutes to seconds
else:
    delay = int(float(input('Delay in minutes between checks: ')) * 60)
    
if args.dryrun:
    dryrun=True
else:
    dryrun=False

# File to store the last processed tweet ID
LAST_TWEET_FILE = f'last_tweet_id_{username}.txt'

def authenticate():
    global client_oauth1, client_oauth2
    #load the keys
    #keys = load_keys()
    
    # Authenticate with X API v2
    client_oauth1 = tweepy.Client(
        consumer_key=keys['consumer_key'],
        consumer_secret=keys['consumer_secret'],
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
                    return int(content)
        except ValueError:
            print(f"Warning: Invalid content in {LAST_TWEET_FILE}: {content}")
    return None

def write_last_tweet_id(tweet_id):
    """
    Write the given tweet ID to the file.
    """
    with open(LAST_TWEET_FILE, 'w') as f:
        f.write(str(tweet_id))



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
last_tweet_id = read_last_tweet_id()

def fetch_and_process_tweets(user_id, username):
    """
    Fetch and process new tweets from the specified user.
    Updates the global last_tweet_id and saves it to the file.
    """
    global last_tweet_id
    print(f"Checking for new tweets at {datetime.datetime.now()}")
    try:
        # Fetch tweets, using since_id if we have a last_tweet_id
        if last_tweet_id is None:
            tweets = client_oauth2.get_users_tweets(
                id=user_id,
                max_results=5,
                tweet_fields=["id", "text"]
            )
        else:
            tweets = client_oauth2.get_users_tweets(
                id=user_id,
                since_id=last_tweet_id,
                max_results=5,
                tweet_fields=["id", "text"]
            )
        
        # Process any new tweets
        if tweets.data:
            for tweet in tweets.data:
                print(f"\n{username} posted: {tweet.text}")
                # Assuming fact_check is a defined function that processes the tweet
                success = fact_check(tweet.text, tweet.id)
                if success == 'delay!':
                    return
            # Update last_tweet_id to the highest ID (most recent tweet)
            if success == 'done!':
                last_tweet_id = max(tweet.id for tweet in tweets.data)
                write_last_tweet_id(last_tweet_id)
        else:
            print("No new tweets found.")
    except tweepy.TweepyException as e:
        print(f"Error fetching tweets: {e}\n")

# Assuming fact_check is defined elsewhere, e.g.:
# def fact_check(tweet_text, tweet_id):
#     # Your fact-checking and replying logic here
#     pass

# Main loop to check for tweets every minute

#delay = int(float(input('Delay in minutes between checks: '))*60)
RESTART_DELAY = 10

def main():
    while True:
        authenticate()
        user_id = getid()
        try:
            while True:
                fetch_and_process_tweets(user_id, username)
                time.sleep(delay)  # Wait before the next check
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
