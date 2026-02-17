# ConSenseAI

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Twitter](https://img.shields.io/twitter/follow/ConSenseAI?style=social)](https://x.com/ConSenseAI)

**Multi-model AI fact-checking bot for X/Twitter**  
Combines Grok, GPT, and Claude to provide comprehensive, balanced responses to misinformation.

🌐 **Website:** [ai-against-autocracy.org](https://ai-against-autocracy.org)  
🐦 **Bot Account:** [@ConSenseAI](https://x.com/ConSenseAI)  
💻 **Source:** [github.com/Nhorning/ConSenseAI](https://github.com/Nhorning/ConSenseAI)

---

## Key Features

### 🤖 Multi-Model Consensus
- **2 lower-tier models** analyze each claim (randomly selected from Grok-4-1-Fast, GPT-5-Mini, Claude-Haiku-4-5)
- **1 higher-tier model** combines responses from the AI company that didn't run (Grok-4, GPT-5.2, or Claude-Sonnet-4-5)
- **Guarantees** all 3 AI companies participate in every fact-check
- **Web search** enabled for Grok and Claude for real-time verification
- **Vision support** for GPT and Claude to analyze images and videos

### 🔍 Four Discovery Modes
1. **@Mentions** - Responds when tagged by users
2. **Proactive Search** - Finds misinformation based on AI-generated or custom keywords
3. **Followed Users** - Monitors and fact-checks tweets from accounts the bot follows
4. **Community Notes** - Writes Twitter Community Notes for flagged posts (with optional adversarial verification)

### 🛡️ Safety & Quality Controls
- **Per-user reply limits** with polite threshold notifications
- **Deduplication** - Won't reply twice to the same tweet or similar content
- **Dynamic daily caps** that increase hourly to balance activity
- **Automatic retry** with different models if one fails
- **Network resilience** with exponential backoff for outages

### 📊 Community Notes Integration
- **Test mode** for safe experimentation before going live
- **Adversarial verification** - LLMs predict if notes will be rated helpful before submission
- **Score calibration** based on Twitter's actual ClaimOpinion ratings
- **Comprehensive logging** of all note submissions and outcomes

### 💭 Reflection Posts
- Periodically generates standalone tweets summarizing recent fact-checks
- Auto-follows users with high engagement (configurable threshold)
- AI-generated search terms when `--search_term "auto"` is used

---

## Quick Start

### Prerequisites
- Python 3.8+
- X/Twitter API access (Basic tier minimum)
- API keys for: xAI (Grok), OpenAI (GPT), Anthropic (Claude)

### Installation
```bash
git clone https://github.com/Nhorning/ConSenseAI.git
cd ConSenseAI
pip install tweepy openai anthropic xai-sdk
```

### Configuration
Create `keys.txt` with your API credentials:
```
XAPI_key=your_x_api_key
XAPI_secret=your_x_api_secret
bearer_token=your_bearer_token
XAI_API_KEY=your_xai_api_key
CHATGPT_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
access_token=your_access_token
access_token_secret=your_access_token_secret
```

### Run the Bot
```bash
python ConSenseAI_v1.7.py \
  --username your_bot_username \
  --delay 3 \
  --search_term "auto" \
  --search_daily_cap 14 \
  --check_followed_users True \
  --post_interval 10
```

**First run:** You'll be guided through Twitter's OAuth flow to authorize the bot.

---

## Command-Line Arguments

### Essential
- `--username` - Twitter username of the bot account
- `--delay` - Minutes between checking for new tweets (default: 3)
- `--dryrun` - Test mode, no tweets posted (True/False)

### Search & Discovery
- `--search_term` - Keyword to search ("auto" for AI-generated terms)
- `--search_daily_cap` - Max search replies per day (increases hourly)
- `--check_followed_users` - Monitor followed users (True/False)
- `--followed_users_daily_cap` - Max replies to followed users per day

### Safety Controls
- `--reply_threshold` - Max replies per conversation/user
- `--per_user_threshold` - Apply threshold per-user instead of per-thread
- `--dedupe_window_hours` - Hours to check for duplicate replies (default: 24)

### Community Notes
- `--check_community_notes` - Enable CN processing (True/False)
- `--cn_test_mode` - Submit notes in test mode (True/False)
- `--cn_verify_helpfulness` - LLM verification before submission (True/False)
- `--cn_max_results` - Max CN posts to check per cycle

### Reflection & Engagement
- `--post_interval` - Post standalone reflection every N bot replies
- `--follow_threshold` - Min interactions before auto-following user

Run `python ConSenseAI_v1.7.py --help` for the complete list.

---

## Architecture

### Data Flow
```
Input Sources → Context Building → Multi-Model Analysis → Response Generation
     ↓               ↓                      ↓                     ↓
  Mentions      Cache Check          2 Lower-Tier           Post Reply
  Search        Thread Fetch         1 Higher-Tier          Update Cache
  Followed      Media Extract        (from unused co.)      Reflection
  CN Posts      Ancestor Chain                              Auto-Follow
```

### Key Files
- `ConSenseAI_v1.7.py` - Main bot (7200+ lines)
- `bot_tweets.json` - Bot's tweet history (max 1000)
- `ancestor_chains.json` - Conversation hierarchies (max 500)
- `cn_written_{username}.json` - Community Notes tracking
- `output.log` - Rotating logs (10MB, 5 files)

### Caching Strategy
- **Cache-first**: Checks `ancestor_chains.json` before making API calls
- **Automatic pruning**: Removes oldest entries when limits reached
- **Deduplication**: Content hashes prevent duplicate replies within 24h
- **Per-user counting**: Tracks replies per user in each conversation

---

## Advanced Features

### Auto-Generated Search Terms
When `--search_term "auto"`:
- Reviews recent bot conversations after each reflection post
- Uses all 3 AI models to generate controversial/relevant keywords
- Avoids repeating previously used terms
- Optimizes for engagement and impact

### Network Resilience
- **Exponential backoff** on rate limits (5s → 10s → 20s → 40s...)
- **Auto-retry** authentication with up to 10 attempts
- **Graceful degradation** when external APIs fail
- **Auto-restart** on critical errors after 10-60s delay

### Vision Analysis
- Automatically analyzes images/videos in tweets
- Integrates visual context into fact-checking
- Supported by GPT and Claude models

---

## Rate Limits (X API Basic Tier)

| Operation | Limit | Window |
|-----------|-------|--------|
| Read tweets | 10,000 | 24h |
| Post tweets | 100 | 24h |
| Mentions check | 5 | 15min |

**Tip:** Run multiple bot instances for different accounts to multiply capacity.

---

## Development

### Debugging
```bash
# Dry run mode (no posts)
python ConSenseAI_v1.7.py --dryrun True --delay 1

# Check logs
tail -f output.log

# Inspect caches
cat bot_tweets.json | jq .
cat ancestor_chains.json | jq .
```

### Testing Community Notes
```bash
# Test mode (notes only visible to you)
python ConSenseAI_v1.7.py --check_community_notes True --cn_test_mode True
```

### Code Structure
- **Single-file design** for easy deployment
- **Defensive coding** for Tweepy dict/object ambiguity
- **Extensive error handling** with automatic recovery
- See `.github/copilot-instructions.md` for architecture details

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description
4. Report bugs via GitHub Issues

**Areas needing help:**
- Unit tests with mocked API responses
- CI/CD pipeline for automated testing
- Documentation improvements
- Community Notes score calibration

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

This ensures any modifications deployed as a service must also be open-sourced.

---

## Credits

**Created by:** AI Against Autocracy  
**Contributors:** See GitHub contributors page  
**Models:** xAI (Grok), OpenAI (GPT), Anthropic (Claude)

**Support the project:** [ai-against-autocracy.org/donate](https://ai-against-autocracy.org)

---

## Disclaimer

This bot provides AI-generated fact-checks that may contain errors. Always verify critical information from primary sources. The bot is not affiliated with Twitter/X, xAI, OpenAI, or Anthropic.
