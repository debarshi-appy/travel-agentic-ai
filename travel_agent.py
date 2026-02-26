from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal


# ─────────────────────────────────────────
# 1. Define Tools per Agent
# ─────────────────────────────────────────

# -- Research Agent Tools --
@tool
def search_destination(destination: str) -> str:
    """Search for general info about a travel destination."""
    return f"{destination}: Popular tourist city. Best time to visit: Spring/Fall. Visa required for most countries."

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = {"Paris": "Cloudy, 18°C", "Tokyo": "Sunny, 24°C", "New York": "Rainy, 15°C"}
    return data.get(city, f"No weather data for {city}")


# -- Flights Agent Tools --
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights."""
    return f"3 flights found from {origin} to {destination} on {date}. Cheapest: $420 (Air France, 8h30m)"

@tool
def check_baggage_policy(airline: str) -> str:
    """Check baggage policy for an airline."""
    return f"{airline}: 1 carry-on (10kg) + 1 checked bag (23kg) included in economy."


# -- Hotels Agent Tools --
@tool
def search_hotels(city: str, check_in: str, check_out: str, budget: str = "mid-range") -> str:
    """Search for hotels in a city."""
    return f"Top {budget} pick in {city}: Grand Hotel at $115/night (4.5★). Available {check_in} to {check_out}."

@tool
def get_hotel_amenities(hotel_name: str) -> str:
    """Get amenities for a specific hotel."""
    return f"{hotel_name} amenities: Free WiFi, Pool, Breakfast included, Airport shuttle."


# -- Budget Agent Tools --
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency."""
    rates = {"USD_EUR": 0.92, "USD_JPY": 149.5, "EUR_USD": 1.09}
    rate = rates.get(f"{from_currency}_{to_currency}")
    if rate:
        return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"
    return f"Rate not available for {from_currency} -> {to_currency}"

@tool
def estimate_daily_budget(city: str, travel_style: str = "mid-range") -> str:
    """Estimate daily travel budget for a city."""
    budgets = {
        "Paris": {"budget": "$80", "mid-range": "$180", "luxury": "$500"},
        "Tokyo": {"budget": "$70", "mid-range": "$160", "luxury": "$450"},
    }
    city_data = budgets.get(city, {"mid-range": "$150"})
    return f"Daily {travel_style} budget in {city}: {city_data.get(travel_style, '$150')}/day"


# ─────────────────────────────────────────
# 2. Create Specialized Agents
# ─────────────────────────────────────────

research_agent = create_agent(
    model="gpt-4o",
    tools=[search_destination, get_weather],
    system_prompt=(
        "You are a travel research specialist. "
        "Your job is to gather destination info and weather data. "
        "Be concise and return only the relevant research findings."
    ),
)

flights_agent = create_agent(
    model="gpt-4o",
    tools=[search_flights, check_baggage_policy],
    system_prompt=(
        "You are a flight booking specialist. "
        "Find the best flight options and check baggage policies. "
        "Always summarize the top recommendation clearly."
    ),
)

hotels_agent = create_agent(
    model="gpt-4o",
    tools=[search_hotels, get_hotel_amenities],
    system_prompt=(
        "You are a hotel booking specialist. "
        "Find the best hotel options and detail the amenities. "
        "Always summarize the top recommendation clearly."
    ),
)

budget_agent = create_agent(
    model="gpt-4o",
    tools=[convert_currency, estimate_daily_budget],
    system_prompt=(
        "You are a travel budget specialist. "
        "Estimate costs, convert currencies, and provide a clear budget breakdown. "
        "Summarize total estimated trip cost at the end."
    ),
)

# Intake agent — validates that the user provided enough info
intake_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt=(
        "You are a travel planning intake specialist. "
        "Analyze the user's travel request and determine if they have provided "
        "enough information for a travel planning team to work with.\n\n"
        "The team needs at minimum:\n"
        "1. Destination (where they want to go)\n"
        "2. Origin (where they're traveling from)\n"
        "3. Approximate travel dates or time frame\n"
        "4. Budget preference or travel style (budget, mid-range, luxury)\n\n"
        "If the request contains ALL of these details (even approximately), "
        "respond with ONLY the single word: READY\n\n"
        "If ANY key details are missing, respond with a friendly, concise message "
        "asking for the specific missing information. Do NOT plan the trip yourself."
    ),
)


# ─────────────────────────────────────────
# 3. Define Graph State
# ─────────────────────────────────────────

class TripState(TypedDict):
    messages: Annotated[list, add_messages]
    research_result: str
    flights_result: str
    hotels_result: str
    budget_result: str


# ─────────────────────────────────────────
# 4. Define Node Functions
#    Each node invokes its agent and stores
#    the result in the state
# ─────────────────────────────────────────

def intake(state: TripState):
    """Validate the user's request has enough info before dispatching agents."""
    result = intake_agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content

    # If all info is present, pass through
    if answer.strip().upper() == "READY":
        return {}

    # Missing info — interrupt and ask the user
    user_response = interrupt({
        "question": answer,
        "node": "intake",
    })

    # Add the Q&A to messages so all downstream agents have full context
    return {
        "messages": [
            {"role": "assistant", "content": f"[Intake]: {answer}"},
            {"role": "user", "content": user_response},
        ]
    }


def run_research(state: TripState):
    result = research_agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    return {
        "research_result": answer,
        "messages": [{"role": "assistant", "content": f"[Research Agent]: {answer}"}]
    }

def run_flights(state: TripState):
    result = flights_agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    return {
        "flights_result": answer,
        "messages": [{"role": "assistant", "content": f"[Flights Agent]: {answer}"}]
    }

def needs_user_input(text: str) -> bool:
    """Check if the hotels agent is asking the user for specific dates."""
    text_lower = text.lower()
    has_question = "?" in text
    date_keywords = [
        "check-in", "check-out", "checkin", "checkout",
        "dates", "when", "arrival", "departure",
        "specific date", "exact date", "travel date",
    ]
    has_date_ref = any(kw in text_lower for kw in date_keywords)
    asking_patterns = ["could you", "can you", "please provide", "would you", "i need", "what are"]
    has_asking = any(p in text_lower for p in asking_patterns)
    return has_question and (has_date_ref or has_asking)


def run_hotels(state: TripState):
    result = hotels_agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content

    # If the agent is asking for specific dates, interrupt for user input
    if needs_user_input(answer):
        user_input = interrupt({
            "question": answer,
            "node": "hotels",
        })
        # Re-invoke the agent with the user's date response
        updated_messages = state["messages"] + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": user_input},
        ]
        result = hotels_agent.invoke({"messages": updated_messages})
        answer = result["messages"][-1].content

    return {
        "hotels_result": answer,
        "messages": [{"role": "assistant", "content": f"[Hotels Agent]: {answer}"}]
    }

def run_budget(state: TripState):
    # Budget agent gets context from previous agents
    context = f"""
User request: {state["messages"][0].content}

Research findings: {state.get("research_result", "")}
Flights findings: {state.get("flights_result", "")}
Hotels findings: {state.get("hotels_result", "")}

Based on the above, provide a full budget breakdown.
    """
    result = budget_agent.invoke({"messages": [{"role": "user", "content": context}]})
    answer = result["messages"][-1].content
    return {
        "budget_result": answer,
        "messages": [{"role": "assistant", "content": f"[Budget Agent]: {answer}"}]
    }

def summarize(state: TripState):
    """Final node: compile all agent results into one clean response."""
    summary = (
        "## Your Travel Plan Summary\n\n"
        "### Destination Research\n"
        f"{state.get('research_result', 'N/A')}\n\n"
        "### Flights\n"
        f"{state.get('flights_result', 'N/A')}\n\n"
        "### Hotels\n"
        f"{state.get('hotels_result', 'N/A')}\n\n"
        "### Budget Breakdown\n"
        f"{state.get('budget_result', 'N/A')}\n"
    )
    return {"messages": [{"role": "assistant", "content": summary}]}


# ─────────────────────────────────────────
# 5. Build the Graph
# ─────────────────────────────────────────

def build_graph():
    """Build the travel agent graph (uncompiled)."""
    graph = StateGraph(TripState)

    # Add all nodes
    graph.add_node("research", run_research)
    graph.add_node("flights", run_flights)
    graph.add_node("hotels", run_hotels)
    graph.add_node("budget", run_budget)
    graph.add_node("summarize", summarize)

    # Wire up the flow:
    # START → intake → research + flights + hotels (parallel) → budget → summarize
    graph.add_node("intake", intake)
    graph.add_edge(START, "intake")

    # Intake fans out to the three parallel agents
    graph.add_edge("intake", "research")
    graph.add_edge("intake", "flights")
    graph.add_edge("intake", "hotels")

    # All three feed into budget
    graph.add_edge("research", "budget")
    graph.add_edge("flights", "budget")
    graph.add_edge("hotels", "budget")

    # Budget feeds into final summary
    graph.add_edge("budget", "summarize")
    graph.add_edge("summarize", END)

    return graph


# Compiled without checkpointer — compatible with langgraph dev (platform provides its own)
# For standalone uvicorn usage, api.py compiles with MemorySaver
app = build_graph().compile()


# ─────────────────────────────────────────
# 6. Run directly (only when executed as script)
# ─────────────────────────────────────────

if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    app = build_graph().compile(checkpointer=MemorySaver())

    # Visualize the graph
    try:
        with open("graph.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("Graph image successfully saved as 'graph.png'")
    except Exception as e:
        print(f"Note: Could not save PNG: {e}")

    print("\n--- Graph Visualization (ASCII) ---")
    try:
        app.get_graph().print_ascii()
    except Exception:
        pass
    print("-----------------------------------\n")

    # Run the travel agent
    config = {"configurable": {"thread_id": "trip-paris-001"}}

    result = app.invoke(
        {
            "messages": [{
                "role": "user",
                "content": (
                    "I want to plan a 7-day trip from New York to Paris "
                    "in mid-April. I have a mid-range budget. "
                    "Find flights, a good hotel, and estimate total costs in EUR."
                )
            }],
            "research_result": "",
            "flights_result": "",
            "hotels_result": "",
            "budget_result": "",
        },
        config=config
    )

    print(result["messages"][-1].content)
