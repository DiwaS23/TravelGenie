# pyrefly: ignore [missing-import]
from langchain_core.tools import tool

BASE_ITEMS = ["Passport/ID", "Phone charger", "Toiletries", "Underwear & socks", "Travel adapter"]

WEATHER_ITEMS = {
    "hot": ["Sunscreen", "Sunglasses", "Light breathable clothing", "Hat"],
    "cold": ["Warm jacket", "Gloves", "Scarf", "Thermal layers"],
    "rain": ["Umbrella", "Waterproof jacket", "Waterproof shoes"],
    "mild": ["Light jacket", "Comfortable walking shoes"],
}

STYLE_ITEMS = {
    "Budget": ["Reusable water bottle", "Snacks for travel"],
    "Balanced": ["Camera", "Day bag"],
    "Luxury": ["Formal outfit", "Dress shoes"],
}


@tool
def get_packing_list(weather_condition: str, travel_style: str) -> str:
    """
    Generate a packing list based on weather conditions and travel style.
    Use this whenever the user asks what to pack or bring for their trip.

    Args:
        weather_condition: One of "hot", "cold", "rain", or "mild" describing
                            the expected weather at the destination.
        travel_style: One of "Budget", "Balanced", or "Luxury".
    """
    weather_key = weather_condition.lower().strip()
    if weather_key not in WEATHER_ITEMS:
        weather_key = "mild"

    normalized_style = travel_style.strip().capitalize()
    style_items = STYLE_ITEMS.get(normalized_style, [])

    all_items = BASE_ITEMS + WEATHER_ITEMS[weather_key] + style_items

    formatted = "\n".join(f"- {item}" for item in all_items)
    return f"Recommended packing list:\n{formatted}"


if __name__ == "__main__":
    print(get_packing_list.invoke({"weather_condition": "cold", "travel_style": "Luxury"}))