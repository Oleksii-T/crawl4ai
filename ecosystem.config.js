module.exports = {
  apps: [
    {
      name: "crawl4ai",
      script: "/var/www/crawl4ai/venv/bin/uvicorn",
      args: "app:app --host 0.0.0.0 --port 8001 --workers 1",
      cwd: "/var/www/crawl4ai",
      interpreter: "none",
    },
  ],
};
