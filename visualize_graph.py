from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

# 1. Define the State (same as your original)
def add_messages(left, right):
    return left + right

class TripState(TypedDict):
    messages: Annotated[list, add_messages]
    research_result: str
    flights_result: str
    hotels_result: str
    budget_result: str

# 2. Mock Nodes (we only need the structure for visualization)
def run_research(state): return state
def run_flights(state): return state
def run_hotels(state): return state
def run_budget(state): return state
def summarize(state): return state

# 3. Build the Graph
graph = StateGraph(TripState)

graph.add_node("research", run_research)
graph.add_node("flights", run_flights)
graph.add_node("hotels", run_hotels)
graph.add_node("budget", run_budget)
graph.add_node("summarize", summarize)

graph.add_edge(START, "research")
graph.add_edge(START, "flights")
graph.add_edge(START, "hotels")

graph.add_edge("research", "budget")
graph.add_edge("flights", "budget")
graph.add_edge("hotels", "budget")

graph.add_edge("budget", "summarize")
graph.add_edge("summarize", END)

# 4. Compile and Save
app = graph.compile()

try:
    print("Generating graph image...")
    # This creates the PNG
    with open("graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("\nSuccess! Graph image saved as 'graph.png'")
    
    # Also print the ASCII version
    print("\n--- Graph Visualization (ASCII) ---")
    app.get_graph().print_ascii()
    print("-----------------------------------\n")
except Exception as e:
    print(f"Error generating graph: {e}")
    print("\nTry running: pip install pyppeteer pillow")
