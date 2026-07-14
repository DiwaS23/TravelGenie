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

import os

load_dotenv()


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


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    disable_streaming=True,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

tools = [get_weather, search_attractions, search_hotels, estimate_budget, get_packing_list]
checkpointer = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=checkpointer)


def run_agent(message: str, thread_id: str):
    response = agent_executor.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    final_answer = extract_text(response["messages"][-1].content)
    print("Final Answer:", final_answer)
    return final_answer


if __name__ == "__main__":
    run_agent("Plan a 3-day trip to Lisbon, budget style.", thread_id="demo")