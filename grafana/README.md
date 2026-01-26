# Grafana vLLM Dashboard

## Export
```bash
curl -s -u admin:admin123 http://localhost:3000/api/dashboards/uid/vllm-spark \
  | jq '.dashboard' > vllm-dashboard.json
```

## Import
```bash
curl -X POST -u admin:admin123 -H "Content-Type: application/json" \
  -d "{\"dashboard\": $(cat vllm-dashboard.json), \"overwrite\": true}" \
  http://localhost:3000/api/dashboards/db
```
