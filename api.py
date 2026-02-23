"""
FastAPI server for the Travel Agent LangGraph.

Run with:
    uvicorn api:api --reload

Endpoints:
    POST  /plan          - Plan a trip (full response)
    POST  /plan/stream   - Plan a trip (streamed node-by-node)
    GET   /history/{id}  - Get conversation history for a thread
    GET   /graph         - View the graph structure as PNG
    GET   /health        - Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import json
import os

# Import the compiled graph from travel_agent.py
from travel_agent import app as travel_app


# ─────────────────────────────────────────
# FastAPI App Setup
# ─────────────────────────────────────────

api = FastAPI(
    title="Travel Agent API",
    description="AI-powered multi-agent travel planner built with LangGraph",
    version="1.0.0",
)

# Allow CORS for local development / testing from browser
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────

class TripRequest(BaseModel):
    """Request body for planning a trip."""
    message: str = Field(
        ...,
        description="Your travel planning request",
        examples=[
            "I want to plan a 7-day trip from New York to Paris in mid-April. "
            "I have a mid-range budget. Find flights, a good hotel, and estimate total costs in EUR."
        ],
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for conversation continuity. Leave empty for a new conversation.",
    )


class TripResponse(BaseModel):
    """Response body for a planned trip."""
    thread_id: str
    summary: str
    research: Optional[str] = None
    flights: Optional[str] = None
    hotels: Optional[str] = None
    budget: Optional[str] = None


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@api.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation link."""
    return """
    <html>
        <head>
            <title>Travel Agent API</title>
            <style>
                body {
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    max-width: 700px;
                    margin: 60px auto;
                    padding: 0 20px;
                    background: #0f0f0f;
                    color: #e0e0e0;
                }
                h1 { color: #60a5fa; }
                a { color: #93c5fd; }
                code {
                    background: #1e293b;
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 0.95em;
                }
                .endpoint {
                    background: #1e293b;
                    padding: 16px 20px;
                    border-radius: 8px;
                    margin: 12px 0;
                    border-left: 3px solid #60a5fa;
                }
                .method {
                    font-weight: bold;
                    color: #34d399;
                }
            </style>
        </head>
        <body>
            <h1>Travel Agent API</h1>
            <p>AI-powered multi-agent travel planner built with LangGraph.</p>
            <p>Interactive docs: <a href="/docs">/docs</a></p>
            <h3>Endpoints</h3>
            <div class="endpoint">
                <span class="method">POST</span> <code>/plan</code> &mdash; Plan a trip (full response)
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <code>/plan/stream</code> &mdash; Plan a trip (streamed)
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <code>/history/{thread_id}</code> &mdash; Get thread history
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <code>/graph</code> &mdash; View graph as PNG
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code> &mdash; Health check
            </div>
        </body>
    </html>
    """


@api.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Travel Agent API"}


@api.post("/plan", response_model=TripResponse)
async def plan_trip(request: TripRequest):
    """
    Plan a trip using the multi-agent travel planner.

    Sends your message through the full agent pipeline:
    Research, Flights, Hotels -> Budget -> Summary
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = travel_app.invoke(
            {
                "messages": [{"role": "user", "content": request.message}],
                "research_result": "",
                "flights_result": "",
                "hotels_result": "",
                "budget_result": "",
            },
            config=config,
        )

        # Extract the final summary from the last message
        final_message = result["messages"][-1]
        summary = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        return TripResponse(
            thread_id=thread_id,
            summary=summary,
            research=result.get("research_result"),
            flights=result.get("flights_result"),
            hotels=result.get("hotels_result"),
            budget=result.get("budget_result"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@api.post("/plan/stream")
async def plan_trip_stream(request: TripRequest):
    """
    Plan a trip with streaming — results are sent node-by-node
    as Server-Sent Events (SSE).

    Each event contains the node name and its output.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # Send thread_id first
            yield f"data: {json.dumps({'event': 'start', 'thread_id': thread_id})}\n\n"

            for event in travel_app.stream(
                {
                    "messages": [{"role": "user", "content": request.message}],
                    "research_result": "",
                    "flights_result": "",
                    "hotels_result": "",
                    "budget_result": "",
                },
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    # Extract message content
                    messages = node_output.get("messages", [])
                    content = ""
                    if messages:
                        msg = messages[-1]
                        if hasattr(msg, "content"):
                            content = msg.content
                        elif isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                        else:
                            content = str(msg)

                    payload = {
                        "event": "node_complete",
                        "node": node_name,
                        "content": content,
                    }

                    # Include agent-specific results if present
                    for key in ["research_result", "flights_result", "hotels_result", "budget_result"]:
                        if key in node_output:
                            payload[key] = node_output[key]

                    yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'event': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@api.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Retrieve the conversation history for a given thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = travel_app.get_state(config)
        if not state or not state.values:
            raise HTTPException(status_code=404, detail=f"No history found for thread '{thread_id}'")

        messages = []
        for msg in state.values.get("messages", []):
            if hasattr(msg, "content"):
                messages.append({"role": getattr(msg, "type", "unknown"), "content": msg.content})
            elif isinstance(msg, dict):
                messages.append(msg)

        return {
            "thread_id": thread_id,
            "message_count": len(messages),
            "messages": messages,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@api.get("/graph")
async def get_graph():
    """
    Returns the graph visualization as a PNG image.
    """
    graph_path = os.path.join(os.path.dirname(__file__), "graph.png")

    # Generate the graph if it doesn't exist yet
    if not os.path.exists(graph_path):
        try:
            with open(graph_path, "wb") as f:
                f.write(travel_app.get_graph().draw_mermaid_png())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not generate graph: {str(e)}")

    return FileResponse(graph_path, media_type="image/png", filename="travel_agent_graph.png")


# ─────────────────────────────────────────
# Run with: uvicorn api:api --reload
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)
