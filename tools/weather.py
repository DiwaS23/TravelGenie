import requests
import os
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_core.tools import tool

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather conditions for a specific city.
    Use this whenever the user asks about weather, temperature, climate,
    or whether they should pack certain clothing/items based on conditions.
    The input should be just a city name, e.g. "Tokyo" or "Paris", not a full address.
    """
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return (
            f"The weather in {city} is currently {description}, "
            f"with a temperature of {temp}°C (feels like {feels_like}°C) "
            f"and {humidity}% humidity."
        )

    except requests.exceptions.HTTPError:
        if response.status_code == 404:
            return f"Sorry, I couldn't find weather data for '{city}'. Please check the city name."
        elif response.status_code == 401:
            return "Weather lookup failed: invalid or missing API key."
        else:
            return f"Weather lookup failed with status code {response.status_code}."

    except requests.exceptions.Timeout:
        return "Weather lookup timed out. Please try again."

    except requests.exceptions.RequestException as e:
        return f"Weather lookup failed due to a network error: {e}"


if __name__ == "__main__":
    print(get_weather.invoke({"city": "Tokyo"}))
    print(get_weather.invoke({"city": "NotARealCityXYZ"}))