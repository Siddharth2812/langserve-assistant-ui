import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timezone
import requests

load_dotenv()

# Initialize ChatOpenAI with gpt-4o-mini model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

def call_analysis_api(problem_statement: str) -> str:
    """
    Call the Django analysis endpoint instead of importing Django models directly.
    This function will make an HTTP request to the Django server.
    """
    # Define possible Django server URLs (now on port 8001)
    django_urls = [
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "http://0.0.0.0:8001"
    ]
    
    last_error = None
    # Try each URL until one works
    for base_url in django_urls:
        try:
            response = requests.post(
                f"{base_url}/modelv3/analyze_data/",  # Updated endpoint to match Django URLs
                json={"problem_statement": problem_statement},
                headers={"Host": "127.0.0.1:8001"},
                timeout=10
            )
            response.raise_for_status()
            print(response.json())
            return response.json()["analysis"]
        except requests.RequestException as e:
            last_error = str(e)
            continue
    
    # If we get here, none of the URLs worked
    return f"Error calling analysis API: {last_error}. Make sure the Django server is running on port 8001."

@tool(return_direct=True)
def analyze_data(problem_statement: str):
    """Call to analyze data using the Django-based ML models and return analysis."""
    return call_analysis_api(problem_statement)

@tool
def get_weather(location: str):
    """Call to get the weather from a specific location."""
    # This is a placeholder for the actual implementation
    if any([city in location.lower() for city in ["sf", "san francisco"]]):
        return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."
    else:
        return f"I am not sure what the weather is in {location}"
    
@tool(return_direct=True)
def get_stock_price(stock_symbol: str):
    """Call to get the current stock price and related information for a given stock symbol."""
    # This is a mock implementation
    mock_stock_data = {
        "AAPL": {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 173.50,
            "change": 2.35,
            "change_percent": 1.37,
            "volume": 52436789,
            "market_cap": "2.73T",
            "pe_ratio": 28.5,
            "fifty_two_week_high": 198.23,
            "fifty_two_week_low": 124.17,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        },
    }
    
    return mock_stock_data["AAPL"]

tools = [analyze_data, get_weather, get_stock_price]

SYSTEM_PROMPT = """You are a helpful assistant. 
You are able to call the following tools:
- analyze_data: Use this to analyze data with ML models and get insights
- get_weather
- get_stock_price
"""

system_message = SystemMessage(content=SYSTEM_PROMPT)
agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

