A simple fact checking bot for X  

It currently uses Chat GPT to prompt Grok for the fact check. This might change with newer versions.

To Run:
1. Purchase X api and OpenAI keys.
2. Download Script or clone repo.
3. Create key file according to format in Load_keys() function.
4. Run script: $ python autofactcheck_v1.2.py
5. Specify X user name.
6. Specify delay in minutes between checking for more tweets from the user.

At Basic Tier X API the script can check for more tweets 15 times every 15 mintutes and post 100 tweets every 24 hours. 
It can be run in multiple terminals for an arbitrary amount of users acording to these limits. 

