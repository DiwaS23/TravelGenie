# pyrefly: ignore [missing-import]
from langchain_core.tools import tool

DAILY_RATES = {
    "Budget":   {"hotel": 50,  "food": 25, "activities": 15},
    "Balanced": {"hotel": 120, "food": 50, "activities": 35},
    "Luxury":   {"hotel": 300, "food": 100, "activities": 80},
}

ESTIMATED_FLIGHT_COST = 600


def _calculate_category_costs(style: str, days: int) -> dict:
    rates = DAILY_RATES[style]
    return {
        "hotel": rates["hotel"] * days,
        "food": rates["food"] * days,
        "activities": rates["activities"] * days,
    }


@tool
def estimate_budget(days: int, travel_style: str) -> str:
    """
    Estimate the total cost of a trip, including flights, hotels, food, and activities.
    Works for any destination worldwide. Use this whenever the user asks about
    budget, cost, how much a trip will cost, or wants a spending estimate.

    Args:
        days: The number of days for the trip.
        travel_style: Must be exactly one of "Budget", "Balanced", or "Luxury".
    """
    normalized_style = travel_style.strip().capitalize()

    if normalized_style not in DAILY_RATES:
        return (
            f"Invalid travel style '{travel_style}'. "
            f"Please choose one of: Budget, Balanced, or Luxury."
        )

    if days <= 0:
        return "Number of days must be greater than zero."

    category_costs = _calculate_category_costs(normalized_style, days)

    total_cost = (
        ESTIMATED_FLIGHT_COST
        + category_costs["hotel"]
        + category_costs["food"]
        + category_costs["activities"]
    )

    return (
        f"Estimated budget for a {days}-day {normalized_style.lower()} trip:\n"
        f"- Flights (round-trip estimate): ${ESTIMATED_FLIGHT_COST}\n"
        f"- Hotel ({days} nights): ${category_costs['hotel']}\n"
        f"- Food: ${category_costs['food']}\n"
        f"- Activities: ${category_costs['activities']}\n"
        f"- TOTAL: ${total_cost}"
    )


if __name__ == "__main__":
    print(estimate_budget.invoke({"days": 5, "travel_style": "Budget"}))