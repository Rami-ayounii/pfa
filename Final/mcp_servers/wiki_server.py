"""MCP Wikipedia Server — existence check for en/fr Wikipedia pages.

Run as a standalone MCP server:
    python mcp_servers/wiki_server.py

Tools exposed:
    wikipedia_lookup(entity) -> dict
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wiki-server")


@mcp.tool()
def wikipedia_lookup(entity: str) -> dict:
    """Check if an entity has a Wikipedia page (English or French).

    Returns: {has_wikipedia: bool, found_in: 'en'|'fr'|None, url: str|None}
    """
    result = {"has_wikipedia": False, "found_in": None, "url": None}
    try:
        import wikipediaapi
        for lang in ["en", "fr"]:
            wiki = wikipediaapi.Wikipedia(user_agent="GEO-Pipeline/1.0", language=lang)
            for title in [entity, entity.title(), entity.replace(" ", "_")]:
                page = wiki.page(title)
                if page.exists():
                    result["has_wikipedia"] = True
                    result["found_in"]      = lang
                    result["url"]           = page.fullurl
                    return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
