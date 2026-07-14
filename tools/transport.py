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
def search_transportation(origin: str, destination: str) -> str:
    """
    Search for transportation options (flights, trains, buses) between two locations,
    including approximate prices and typical travel times. Use this whenever the
    user asks how to get somewhere, or when building a full itinerary that needs
    transportation details.

    Args:
        origin: The starting city, e.g. "Delhi" or "New York". If unknown, use "your city".
        destination: The destination city, e.g. "Tokyo" or "Paris".
    """
    query = f"flights and trains from {origin} to {destination} price and duration booking"

    try:
        response = tavily_client.search(query=query, max_results=8)
        results = response.get("results", [])

        filtered_results = [
            r for r in results
            if not any(bad_domain in r.get("url", "") for bad_domain in IRRELEVANT_DOMAINS)
        ]

        if not filtered_results:
            return f"No transportation info found from {origin} to {destination}."

        formatted_results = []
        for i, result in enumerate(filtered_results[:5], start=1):
            title = result.get("title", "Untitled")
            content = result.get("content", "No description available.")
            url = result.get("url", "")
            short_content = content[:180] + "..." if len(content) > 180 else content
            formatted_results.append(f"{i}. {title} — {short_content}\n   Link: {url}")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Transportation search failed due to an error: {e}"


if __name__ == "__main__":
    print(search_transportation.invoke({"origin": "Delhi", "destination": "Bangkok"}))