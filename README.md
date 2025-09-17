[@ConSenseAI](https://x.com/ConSenseAI)
A multi-model Grok Alternative for X  
It combines answers from ChatGPT, Grok, and Claude to respond to messages it is tagged in. 

To Run:
1. Purchase X API for your organization account, a second account for the bot, and OpenAI, XAI API, and Grok keys.
3. Download Script or clone repo.
4. Create key file according to format in Load_keys() function.
5. Run script: $ python ConSenseAI_v1.2.py (change version as necissary)
6. Specify X user name and minutes between checking for more tweets from the user. This can also be done with the --username and --delay arguments
7. On the first run you will be taken though three-legged 0Ath flow before the key is stored in keys.txt 

$python ConSenseAI_v1.2 --help for list of arguments

At Basic Tier X API the script can check for more tweets 5 times every 15 minutes and post 100 tweets every 24 hours. 
It can be run in multiple terminals for an arbitrary amount of users according to these limits.
