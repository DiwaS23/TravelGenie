import streamlit as st
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langgraph.prebuilt import create_react_agent
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.memory import MemorySaver

from tools.weather import get_weather
from tools.search import search_attractions
from tools.hotels import search_hotels
from tools.budget import estimate_budget
from tools.packing import get_packing_list
from tools.transport import search_transportation

import os
import uuid

load_dotenv()

SYSTEM_PROMPT = """You are TravelGenie, a friendly and knowledgeable travel planning assistant.

You have access to tools for checking weather, searching attractions, finding hotels,
finding transportation, estimating budgets, and generating packing lists. All tools
work for any city or destination worldwide, using live web search and real weather data.

IMPORTANT RULE ABOUT HOTELS AND TRANSPORTATION:
- Only call the hotel search tool and transportation search tool when the user is
  asking you to build a full ITINERARY or TRIP PLAN.
- Do NOT search for or mention hotels/transportation if the user is only asking about
  weather, attractions, budget, or packing in isolation.
- If a hotel or transportation link looks broken, irrelevant, or is not an actual
  booking site (e.g. a video, forum, or unrelated page), do not include it - mention
  the option by name/price only without a link instead.

WHEN BUILDING A FULL ITINERARY, always structure it exactly like this, including
EVERY section below - never skip the Budget Summary or Packing List sections:

**Getting There**
- [Transport option] — $[price] — [duration] — [Booking link]

**Day 1: [Theme/area of the city]**
- Morning: ...
- Afternoon: ...
- Evening: ...

(continue for each day/city segment in the trip)

**Where to Stay**
- [Hotel name] — $[price]/night — [Booking link]

**Budget Summary**
Call the budget estimation tool for the total trip length and travel style, and
present the itemized breakdown here.

**Packing List**
Call the packing list tool using the general weather conditions expected across
the trip and the travel style, and present it here.

Use the weather tool to inform practical suggestions (e.g., suggest indoor activities
if it's raining). Use the search tool to recommend real attractions with actual names.
Keep each day focused on a specific area of the city to minimize travel time.
Be concise but specific. If the user asks to modify a previous plan, regenerate the
FULL itinerary (all sections) again in the same format.
"""


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


st.set_page_config(page_title="TravelGenie", page_icon="✈️", layout="centered")

# --- VISUAL STYLING ---
st.markdown("""
<style>
    /* Real travel-photo background with a dark overlay for readability */
    .stApp {
        background:
            linear-gradient(rgba(20, 15, 35, 0.55), rgba(20, 15, 35, 0.55)),
            url("https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem 2rem 1rem 2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.25);
    }

    .block-container, .block-container p, .block-container span,
    .block-container li, .block-container div, .block-container label,
    .block-container h1, .block-container h2, .block-container h3,
    .block-container h4, .block-container h5, .block-container h6 {
        color: #222222 !important;
    }

    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] strong {
        color: #222222 !important;
    }
    div[data-testid="stChatMessageContent"] {
        background: rgba(250, 250, 250, 0.95) !important;
        border-radius: 14px;
    }
    div[data-testid="stChatMessageContent"] a {
        color: #a13d8f !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fbc2eb 0%, #a6c1ee 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6a88, #a18cd1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #444444 !important;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }

    /* Photo strip under the subtitle */
    .photo-strip img {
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
        object-fit: cover;
        height: 110px;
        width: 100%;
        transition: transform 0.25s ease;
    }
    .photo-strip img:hover {
        transform: scale(1.04);
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 6px;
        margin-bottom: 10px;
        animation: fadeIn 0.35s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    div[data-baseweb="select"] > div {
        background-color: #f5f5f5 !important;
        color: #222222 !important;
    }
    div[data-baseweb="select"] span {
        color: #222222 !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    ul[data-baseweb="menu"] li {
        color: #222222 !important;
    }

    section[data-testid="stSidebar"] button {
        background-color: #ffffff !important;
        color: #222222 !important;
        border: 1px solid #cccccc !important;
    }
    section[data-testid="stSidebar"] button p {
        color: #222222 !important;
    }

    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #222222 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #888888 !important;
    }

    /* Remove the dark background strip behind the chat input */
    div[data-testid="stBottom"] {
        background: transparent !important;
    }
    div[data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }
    div[data-testid="stBottom"] > div {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌍 TravelGenie ✈️</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Your AI-powered travel planning assistant — plan your next adventure, anywhere in the world 🧳🏖️🗺️</p>',
    unsafe_allow_html=True,
)

# --- PHOTO STRIP ---
st.markdown('<div class="photo-strip">', unsafe_allow_html=True)
photo_cols = st.columns(4)
travel_photos = [
    ("https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?auto=format&fit=crop&w=500&q=70", "Mountains"),
    ("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=500&q=70", "Beaches"),
    ("https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=500&q=70", "Cities"),
    ("https://images.unsplash.com/photo-1493246507139-91e8fad9978e?auto=format&fit=crop&w=500&q=70", "Culture"),
]
for col, (url, caption) in zip(photo_cols, travel_photos):
    with col:
        st.image(url, caption=caption, use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🧭 Trip Settings")
trip_days = st.sidebar.slider("Trip length (days)", min_value=1, max_value=14, value=5)
travel_style = st.sidebar.selectbox("Travel style", ["Budget", "Balanced", "Luxury"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Try: 'Plan a 3-day trip to Lisbon', 'What's the weather in Bangkok?', "
    "'Find hotels in Nairobi under $80'"
)

if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.last_destination = None
    st.session_state.prev_days = trip_days
    st.session_state.prev_style = travel_style
    st.rerun()


@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        disable_streaming=True,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    tools = [
        get_weather,
        search_attractions,
        search_hotels,
        search_transportation,
        estimate_budget,
        get_packing_list,
    ]
    checkpointer = MemorySaver()
    return create_react_agent(llm, tools, checkpointer=checkpointer, prompt=SYSTEM_PROMPT)


agent_executor = get_agent()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_destination" not in st.session_state:
    st.session_state.last_destination = None
if "prev_days" not in st.session_state:
    st.session_state.prev_days = trip_days
if "prev_style" not in st.session_state:
    st.session_state.prev_style = travel_style


def call_agent(prompt_text: str) -> str:
    response = agent_executor.invoke(
        {"messages": [("user", prompt_text)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    return extract_text(response["messages"][-1].content)


# --- DISPLAY CHAT HISTORY ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# --- AUTO-REGENERATE ITINERARY IF SIDEBAR SETTINGS CHANGED ---
settings_changed = (
    trip_days != st.session_state.prev_days
    or travel_style != st.session_state.prev_style
)

if settings_changed and st.session_state.last_destination and st.session_state.messages:
    st.session_state.prev_days = trip_days
    st.session_state.prev_style = travel_style

    regenerate_prompt = (
        f"My trip settings changed to {trip_days} days, {travel_style} style. "
        f"Please regenerate the full itinerary (including transportation and hotels) for "
        f"{st.session_state.last_destination} with these new settings."
    )

    st.session_state.messages.append({"role": "user", "content": regenerate_prompt})
    with chat_container:
        with st.chat_message("user"):
            st.write(regenerate_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Updating your itinerary... ✨"):
                try:
                    answer = call_agent(regenerate_prompt)
                except Exception as e:
                    answer = f"Sorry, something went wrong: {e}"
                st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

elif settings_changed:
    st.session_state.prev_days = trip_days
    st.session_state.prev_style = travel_style

# --- CHAT INPUT ---
user_input = st.chat_input("Ask me to plan your trip...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)

    full_prompt = f"(Trip context: {trip_days} days, {travel_style} style) {user_input}"

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🌎"):
                try:
                    answer = call_agent(full_prompt)
                except Exception as e:
                    answer = f"Sorry, something went wrong: {e}"
                st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    lowered = user_input.lower()
    if any(word in lowered for word in ["itinerary", "plan a trip", "plan my trip", "trip to"]):
        st.session_state.last_destination = user_input

# --- AUTO-SCROLL TO BOTTOM ---
st.markdown(
    """
    <script>
        var mainSection = window.parent.document.querySelector('section.main');
        if (mainSection) {
            mainSection.scrollTo(0, mainSection.scrollHeight);
        }
    </script>
    """,
    unsafe_allow_html=True,
)