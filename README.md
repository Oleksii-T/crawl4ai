# set up:
    pip install -r requirements.txt
    python -m playwright install-deps
    python -m playwright install

# start
    - pm2 start ecosystem.config.js

# request options:
    - java_script_enabled: optional boolean, defaults to true. Set false for pages
      where client-side scripts remove useful server-rendered content before scraping.
    - chunk_token_threshold: optional positive integer, defaults to 1000. Increase it
      to reduce page splitting when one source page should return one extracted item.
    - llm_max_tokens: optional positive integer, defaults to 800. Increase it when
      a larger chunk needs enough completion space for a full article body.

# restart
    - pm2 restart ecosystem.config.js

# calculations:
    - 25k = 0.01$
    - one run for 'https://stopgame.ru/news' of first page - 12k tokens = 0.05$
    - assume 2 runs per day: 0.01$ per day -> 0.3$ per month
