# Import load_dotenv to read our .env file
from dotenv import load_dotenv

# Import the LangChain Gemini chat model wrapper
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

# Import ChatPromptTemplate, and MessagesPlaceholder for injecting history
# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import ChatMessageHistory - stores the growing list of messages
# pyrefly: ignore [missing-import]
from langchain_community.chat_message_histories import ChatMessageHistory

# Import RunnableWithMessageHistory - wraps our chain with automatic memory handling
# pyrefly: ignore [missing-import]
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables (our API keys)
load_dotenv()

# Create the Gemini chat model instance
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define our prompt template - NOTE the new MessagesPlaceholder("history") line.
# This is where all PREVIOUS messages will get automatically inserted.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are TravelGenie, a friendly and concise travel planning assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Build a simple chain: prompt -> llm (no output parser this time - we'll see why below)
chain = prompt_template | llm

# A dictionary to hold separate ChatMessageHistory objects, keyed by session ID
# In a real app, this might be a database instead of a plain Python dictionary
store = {}

# A function that returns the correct history object for a given session ID,
# creating a new empty one if this session hasn't been seen before
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap our chain with automatic memory handling
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- FIRST TURN: ask about Tokyo ---
response1 = chain_with_memory.invoke(
    {"input": "Plan a 5-day trip to Tokyo."},
    config={"configurable": {"session_id": "user-123"}}
)
print("--- Turn 1 ---")
print(response1.content)
print()

# --- SECOND TURN: a follow-up that ONLY makes sense with memory of turn 1 ---
response2 = chain_with_memory.invoke(
    {"input": "Actually, make it 3 days instead."},
    config={"configurable": {"session_id": "user-123"}}
)
print("--- Turn 2 (follow-up) ---")
print(response2.content)