# GEO Multi-Agent Pipeline — Final/

Merged GEO pipeline combining Project 1 (Geo/) and Project 2 agent infrastructure.

## Claude Desktop / VS Code MCP Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "geo-pipeline": {
      "command": "C:\\path\\python.exe",
      "args": ["D:\\path\\mcp_server.py"],
      "cwd": "D:\\path\\Final"
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `run_pipeline(query)` | Full LangGraph pipeline |
| `generate_prompts(domain)` | Agent 0 only — prompt generation |
| `analyze_entities(prompt_set_path)` | Agent 1 only — GEO entity analysis |
| `scrape_brands(brands)` | Agent 2 only — social profile scraping |
| `get_results()` | Read last pipeline_summary.json |
| `list_output_files()` | List CSVs/JSONs in output directory |

## MCP Resources

| Resource | Description |
|----------|-------------|
| `geo://overview` | Pipeline architecture and capabilities |
| `geo://models` | Available LLM models |
| `geo://last-run` | Last pipeline execution summary |

## Internal MCP Servers (mcp_servers/)

Used by the ReAct agent for tool-calling:

- `search_server.py` — Google Maps (SerpApi), TripAdvisor (DDG), DuckDuckGo
- `scrape_server.py` — Apify Instagram/Facebook scraping, website social extraction
- `wiki_server.py` — Wikipedia existence check (en/fr)

## Running

```bash
# Start external MCP server (stdio)
/c/path/python.exe mcp_server.py

# Run full pipeline
/c/path/python.exe pipeline.py --query "Restaurants in Sfax"
```
