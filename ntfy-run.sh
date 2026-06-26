#!/usr/bin/env bash
# (Re)create the self-hosted ntfy push server on spark-1 (port 8090, topic nas-alerts).
# Used by nas-alert-bridge to deliver firing NAS Prometheus alerts to the ntfy phone app.
docker rm -f ntfy 2>/dev/null
docker run -d --name ntfy --restart unless-stopped -p 8090:80 \
  -e NTFY_BASE_URL=http://spark-1.tail620cfa.ts.net:8090 \
  -e NTFY_CACHE_FILE=/var/cache/ntfy/cache.db -e NTFY_CACHE_DURATION=24h \
  -v ntfy-cache:/var/cache/ntfy \
  binwiederhier/ntfy serve
