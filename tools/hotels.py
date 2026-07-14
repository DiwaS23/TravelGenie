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
def search_hotels(city: str, max_price_per_night: float = None) -> str:
    """
    Search for real hotel options in any city worldwide, using live web search.
    Use this whenever the user asks about hotels, places to stay, or accommodation.

    Args:
        city: The name of the city to search hotels in, e.g. "Tokyo" or "Paris".
        max_price_per_night: Optional maximum budget per night in USD, if the user mentioned one.
    """
    if max_price_per_night:
        query = f"best hotels in {city} under ${int(max_price_per_night)} per night booking site"
    else:
        query = f"best hotels to stay in {city} booking site"

    try:
        response = tavily_client.search(query=query, max_results=8)
        results = response.get("results", [])

        filtered_results = [
            r for r in results
            if not any(bad_domain in r.get("url", "") for bad_domain in IRRELEVANT_DOMAINS)
        ]

        if not filtered_results:
            return f"No hotel results found for '{city}'. Try a different city name."

        formatted_results = []
        for i, result in enumerate(filtered_results[:5], start=1):
            title = result.get("title", "Untitled")
            content = result.get("content", "No description available.")
            url = result.get("url", "")
            short_content = content[:180] + "..." if len(content) > 180 else content
            formatted_results.append(f"{i}. {title} — {short_content}\n   Link: {url}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Hotel search failed due to an error: {e}"


if __name__ == "__main__":
    print(search_hotels.invoke({"city": "Bangkok", "max_price_per_night": 50}))