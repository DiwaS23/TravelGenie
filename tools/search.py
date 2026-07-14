import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from tavily import TavilyClient
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

IRRELEVANT_DOMAINS = ["youtube.com", "youtu.be", "wikipedia.org", "reddit.com", "quora.com"]


@tool
def search_attractions(query: str) -> str:
    """
    Search for tourist attractions, restaurants, museums, or activities.
    Use this whenever the user asks what to see, do, or eat in a specific place.
    The input should be a clear search phrase, e.g. "top attractions in Kyoto"
    or "best restaurants near Shibuya, Tokyo".
    """
    try:
        response = tavily_client.search(query=query, max_results=8)
        results = response.get("results", [])

        filtered_results = [
            r for r in results
            if not any(bad_domain in r.get("url", "") for bad_domain in IRRELEVANT_DOMAINS)
        ]

        if not filtered_results:
            return f"No results found for '{query}'. Try a different search phrase."

        formatted_results = []
        for i, result in enumerate(filtered_results[:5], start=1):
            title = result.get("title", "Untitled")
            content = result.get("content", "No description available.")
            short_content = content[:200] + "..." if len(content) > 200 else content
            formatted_results.append(f"{i}. {title} — {short_content}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Search failed due to an error: {e}"


if __name__ == "__main__":
    print(search_attractions.invoke({"query": "top attractions in Kyoto"}))