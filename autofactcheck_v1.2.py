# Cell 1: Fact-check and reply functions
from openai import OpenAI

KEY_FILE = 'keys.txt'

def load_keys():
    """Load keys from the key file. Format:
    consumer_key=V0q_your_consumer_key_Fvn
    consumer_secret=m76B6_your_consumer_secret_fHuAXstAmxHwuEx84G9nHAO
    access_token=48365055_your_access_token_Py4EEaZ9K5AXd0LfAWVbNP4
    access_token_secret=x__your_access_toke_secret_T2xmh5TPTHaIhxJpb9
    bearer_token=AAAAAAAAA_your_bearer_token_AAPXkoDoc
    CHATGPT_API_KEY=your_chat_gpt_secret_key_
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
        
# Set up ChatGPT client
keys = load_keys()
client = OpenAI(api_key=keys['CHATGPT_API_KEY'])


def fact_check(tweet_text, tweet_id):
    # Ask ChatGPT if the tweet is true
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a fact-checking assistant for X posts. You are skeptical of authoritarianism. Your job is to prompt @Grok correctly to verify if something is true.\
             It is important that you verify the content of what is being claimed, not the fact that something was claimed. When in doubt, prompt @grok to rate the post for accuracy on a scale from 1 to 10."},          
            {"role": "user", "content":"Friendly reminder that the courts created a Constitutional crisis, not Trump When they stop him from exercising his legal right to deport illegals, they infringe on the executive branch’s constitutional power"},
            {"role": "assistant", "content": "@grok is he correct?"},
            {"role": "user", "content":"If you were an American falsely accused of wrongdoing on January 6th it wasn’t merely difficult to get “due process,” it was impossible. The entire system was rigged against you. All of it. Those persecuted Americans could only dream of the “due process” afforded illegal aliens."},
            {"role": "assistant", "content":"@Grok is @StephenM lying here?"},
            {"role": "user", "content":"America is a CHRISTIAN nation!"},
            {"role": "assistant", "content":"@Grok is this true according to the constitution?"},
            {"role": "user", "content":"Good news: A documentary featuring Fatima Hassouna, a young, brave photojournalist in Gaza got accepted into Cannes Film Festival. Bad news: Within 24 hours of this news, Israel murdered her and ten family members, including her pregnant sister. What kind of monsters do this?"},
            {"role": "assistant", "content":"@Grok rate this post for accuracy on a scale from 1 to 10 please."},
            {"role": "user", "content":"The incompetence from the liberal media is astonishing."},
            {"role": "assistant", "content":"@Grok rate the accuracy of what Miller says in the video on a one to 10 scale."},
            {"role": "user", "content":"Judicial coup"},
            {"role": "assistant", "content":"@grok is this a judicial coup?"},
            {"role": "user", "content":"This is the hand of the man that the Democrats feel should be brought back to the United States, because he is such “a fine and innocent person.” They said he is not a member of MS-13, even though he’s got MS-13 tattooed onto his knuckles, and two Highly Respected Courts found"},
            {"role": "assistant", "content":"@grok does the presence of a gang tattoo mean he is not entitled to due process?"},
            {"role": "user", "content": tweet_text}
        ],
        max_tokens=100
    )
    verdict = response.choices[0].message.content.strip()
    
    # Fake URL since ChatGPT doesn’t provide one
    #response_url = f"https://chatgpt.com/response/fake_id_{tweet_id}"
    
    reply = f"Auto Fact-check v1: {verdict}"# Details: {response_url}"
    #if len(reply) > 280:  # X’s character limit
        #reply = f"Fact-check: Asked ChatGPT. Response: {verdict[:500]}... Details: {response_url}"
    if verdict != 'Not a factual claim.':
        success = post_reply(tweet_id, reply)
    else:
        print(reply)
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
        
#Cell 2 get and reply to tweets

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

# Set the target username
username = input("X username to factcheck: @").lower()

# File to store the last processed tweet ID
LAST_TWEET_FILE = f'last_tweet_id_{username}.txt'

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
delay = int(float(input('Delay in minutes between checks: '))*60)
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
            print("\nStopping the tweet checker.")
            break

if __name__ == "__main__":
    main()
