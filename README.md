# set up:
    pip install -r requirements.txt
    python -m playwright install-deps
    python -m playwright install

# start
    - pm2 start ecosystem.config.js

# request options:
    - java_script_enabled: optional boolean, defaults to true. Set false for pages
      where client-side scripts remove useful server-rendered content before scraping.

# restart
    - pm2 restart ecosystem.config.js

# calculations:
    - 25k = 0.01$
    - one run for 'https://stopgame.ru/news' of first page - 12k tokens = 0.05$
    - assume 2 runs per day: 0.01$ per day -> 0.3$ per month
