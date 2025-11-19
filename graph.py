import os
from typing import TypedDict, List, Dict, Union, Literal
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
USE_MOCK = os.getenv("USE_MOCK", "").lower() == "true"

# --- Pydantic Schema for Structured Output ---
class RoadmapStep(BaseModel):
    """A single, structured step in the personalized learning path."""
    topic: str = Field(description="The specific learning topic.")
    difficulty: Literal["Beginner", "Intermediate", "Advanced", "N/A"] = Field(description="The estimated difficulty level.")
    resource: str = Field(description="A recommended free or low-cost online resource (e.g., specific YouTube channel, documentation link).")
    resource_url: str = Field(description="Direct URL to the recommended resource. Must start with http or https.", default="")
    estimated_time_hours: int = Field(description="Estimated time to complete this topic, in hours (e.g., 20, 40).")
    confidence: int = Field(description="LLM confidence in this recommendation (0-100).", default=70)

class FinalPath(BaseModel):
    """Wrapper to return the list of roadmap steps."""
    path: List[RoadmapStep] = Field(description="The complete, ordered learning path.")

# --- Define the Graph State ---

class PathState(TypedDict):
    """Represents the shared data structure (state) passed between nodes."""
    user_skills: List[str]
    target_goal: str
    weekly_hours: int
    daily_hours: int
    study_days_per_week: int
    target_completion_days: int
    persona_style: str
    resource_preference: Literal["mixed", "video", "reading", "docs"]
    raw_topic_list: List[str]  # Intermediate list of topics
    learning_path: List[RoadmapStep] # Final structured output

# --- Initialize LLM with Gemini 2.5 Flash ---

try:
    # LLM will automatically use GEMINI_API_KEY environment variable
    llm = None if USE_MOCK else ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
except Exception:
    llm = None 
    print("Warning: GEMINI_API_KEY not found or invalid.")


# --- Helpers ---

VALID_DIFFICULTIES = {"Beginner", "Intermediate", "Advanced", "N/A"}

def ensure_llm():
    return llm is not None and not USE_MOCK

def mock_topics(goal: str) -> List[str]:
    return [
        "Foundations & Mindset",
        f"Core Tools for {goal}",
        "Projects & Portfolio",
        "Interview Readiness"
    ]

def mock_learning_path(state: PathState) -> List[RoadmapStep]:
    return [
        RoadmapStep(
            topic="[High Leverage] Strategy & Mindset",
            resource="Read the official documentation for your target role.",
            resource_url="https://www.example.com/guide",
            estimated_time_hours=10,
            difficulty="Beginner",
            confidence=85,
        ),
        RoadmapStep(
            topic="Hands-on Projects",
            resource="Build a capstone using open datasets.",
            resource_url="https://www.kaggle.com/",
            estimated_time_hours=30,
            difficulty="Intermediate",
            confidence=78,
        ),
    ]

def normalize_roadmap_step(step: RoadmapStep) -> RoadmapStep:
    topic = step.topic.strip().title()
    if topic.lower().startswith("[high leverage]"):
        topic = f"[High Leverage] {topic.split(']', 1)[-1].strip().title()}"
    difficulty = step.difficulty if step.difficulty in VALID_DIFFICULTIES else "Intermediate"
    url = step.resource_url.strip()
    if url and not url.lower().startswith(("http://", "https://")):
        url = f"https://{url.lstrip('/')}"
    return RoadmapStep(
        topic=topic,
        difficulty=difficulty,
        resource=step.resource.strip(),
        resource_url=url,
        estimated_time_hours=max(1, step.estimated_time_hours),
        confidence=max(0, min(100, step.confidence))
    )

def node_guard(node_name):
    def decorator(func):
        def wrapper(state: PathState) -> PathState:
            try:
                return func(state) or {}
            except Exception as error:
                print(f"[LangGraph][{node_name}] Error: {error}")
                return {}
        return wrapper
    return decorator

# --- LangGraph Nodes (Functions) ---

@node_guard("gap_analyzer")
def skill_gap_analyzer(state: PathState) -> PathState:
    """Compares user skills against the target goal to identify required topics."""
    if not ensure_llm():
        return {"raw_topic_list": mock_topics(state["target_goal"])}
    
    prompt = PromptTemplate.from_template(
        "You are an expert career counselor. Given the user's current skills: {skills}, "
        "and their target goal: '{goal}', "
        "what are the **essential, high-level topics** required to achieve this goal? "
        "Focus only on **knowledge gaps**. Return a comma-separated list of topics, **nothing else**."
    )
    
    chain = prompt | llm
    
    raw_topics_str = chain.invoke({
        "skills": ", ".join(state['user_skills']), 
        "goal": state['target_goal']
    }).content
    
    raw_topic_list = [t.strip() for t in raw_topics_str.split(',') if t.strip()]

    return {"raw_topic_list": raw_topic_list}

@node_guard("topic_sequencer")
def topic_sequencer(state: PathState) -> PathState:
    """Orders the raw topics based on prerequisites and learning flow."""
    if not ensure_llm():
        return {"raw_topic_list": mock_topics(state["target_goal"])}
    
    topics = ", ".join(state['raw_topic_list'])
    
    prompt = PromptTemplate.from_template(
        "The user needs to learn these topics for their goal of '{goal}': {topics}. "
        "Order these topics logically, from prerequisite to advanced, into a structured learning flow. "
        "Return the final list as a numbered list (1., 2., 3., etc.). Do not include any other text."
    )
    
    chain = prompt | llm
    
    ordered_topics_str = chain.invoke({"topics": topics, "goal": state['target_goal']}).content
    
    ordered_topics = [
        line.split('. ', 1)[-1].strip()
        for line in ordered_topics_str.split('\n')
        if line.strip() and line[0].isdigit()
    ]

    return {"raw_topic_list": ordered_topics}

@node_guard("resource_formatter")
def resource_finder_and_formatter(state: PathState) -> PathState:
    """Finds resources, estimates time, and formats the final roadmap using Pydantic."""
    if not ensure_llm():
        return {"learning_path": mock_learning_path(state)}
    
    list_structured_llm = llm.with_structured_output(FinalPath, method="json_mode")
    
    topics_list_str = "\n".join([f"- {topic}" for topic in state['raw_topic_list']])
    
    prompt = PromptTemplate.from_template(
        "You are acting as a {persona_style} mentor.\n"
        "For the goal: '{goal}', and the following **ordered** list of learning topics:\n{topics}\n"
        "The learner can dedicate roughly {daily_hours} hours per study day (~{weekly_hours} hours/week across {study_days_per_week} study days) "
        "and wants to finish within {target_days} days.\n"
        "They prefer resources that are {resource_preference}.\n"
        "Generate a structured learning path that respects this pacing. "
        "For each topic, provide: a recommended **Resource** description, a **Resource URL** (must be a fully-qualified `https://` link that works in a browser), "
        "an **Estimated Time (hours)** (use increments of 10 or 20 hours), the **Difficulty** ('Beginner', 'Intermediate', 'Advanced'), "
        "and a **Confidence score (0-100)** indicating how strong this recommendation is. "
        "Follow the 80/20 rule by flagging the most impactful foundational topics with a '[High Leverage]' prefix inside the topic name. "
        "Keep the resources specific and engaging. Return the output as a single list of JSON objects."
    )
    
    list_chain = prompt | list_structured_llm

    try:
        # Pydantic model ensures a clean, parsable list structure
        result: FinalPath = list_chain.invoke({
            "goal": state['target_goal'], 
            "topics": topics_list_str,
            "daily_hours": state['daily_hours'],
            "weekly_hours": state['weekly_hours'],
            "study_days_per_week": state['study_days_per_week'],
            "target_days": state['target_completion_days'],
            "persona_style": state['persona_style'],
            "resource_preference": state['resource_preference']
        })
        # The result is already a list of RoadmapStep Pydantic objects
        structured_path = result.path
    except Exception as e:
        print(f"Structured Output Error: {e}")
        structured_path = [
            RoadmapStep(topic=t, resource="LLM Error", resource_url="", estimated_time_hours=0, difficulty="N/A", confidence=0)
            for t in state['raw_topic_list']
        ]

    return {"learning_path": structured_path}

@node_guard("post_processor")
def post_process_learning_path(state: PathState) -> PathState:
    """Normalizes and deduplicates roadmap entries."""
    steps = state.get("learning_path", [])
    seen = set()
    cleaned: List[RoadmapStep] = []
    for step in steps:
        normalized = normalize_roadmap_step(step)
        key = (normalized.topic.lower(), normalized.resource_url.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return {"learning_path": cleaned}

# --- Build the LangGraph Workflow ---

def build_path_graph():
    """Compiles the StateGraph for the learning path generator."""
    workflow = StateGraph(PathState)

    workflow.add_node("gap_analyzer", skill_gap_analyzer)
    workflow.add_node("topic_sequencer", topic_sequencer)
    workflow.add_node("resource_formatter", resource_finder_and_formatter)
    workflow.add_node("post_processor", post_process_learning_path)

    workflow.set_entry_point("gap_analyzer")
    workflow.add_edge("gap_analyzer", "topic_sequencer")
    workflow.add_edge("topic_sequencer", "resource_formatter")
    workflow.add_edge("resource_formatter", "post_processor")
    workflow.add_edge("post_processor", END)

    return workflow.compile()

# Instantiate the compiled graph for use in app.py
app = build_path_graph()