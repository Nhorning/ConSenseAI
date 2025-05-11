A simple fact checking bot for X  

It can use ChatGPT or Grok 3 API to prompt Grok for a futher fact check.

To Run:
1. Purchase X API and OpenAI or XAI API keys.
2. Download Script or clone repo.
3. Create key file according to format in Load_keys() function.
4. Run script: $ python autogrok_v1.5.py (change version as necissary)
5. Specify X user name and minutes between checking for more tweets from the user. This can also be done with the --username and --delay arguments

$python autogrok_v1.5 --help for list of arguments

At Basic Tier X API the script can check for more tweets 5 times every 15 minutes and post 100 tweets every 24 hours. 
It can be run in multiple terminals for an arbitrary amount of users according to these limits. 

