import streamlit as st
import pandas as pd
from graph import app, PathState, RoadmapStep, llm 
from typing import List, Tuple, Optional
from datetime import date, timedelta
import math
import uuid
import tempfile
import os
import re
import time
try:
    import pyppeteer  # noqa: F401
    PYPPETEER_AVAILABLE = True
except ImportError:
    PYPPETEER_AVAILABLE = False
from fpdf import FPDF # Import the fpdf library (fpdf2 package)
from langchain_core.runnables.graph import CurveStyle, NodeStyles, MermaidDrawMethod

PRESET_LIBRARY = {
    "None (custom)": {
        "goal": "",
        "skills": "",
        "target_days": 120,
        "daily_hours": 3,
        "study_days_per_week": 5
    },
    "ML Engineer (NLP)": {
        "goal": "Machine Learning Engineer specializing in NLP",
        "skills": "Python, Basic Linux, Introductory Data Analysis",
        "target_days": 150,
        "daily_hours": 3,
        "study_days_per_week": 5
    },
    "Data Analyst": {
        "goal": "Data Analyst in FinTech",
        "skills": "Excel, Basic SQL, Business Reporting",
        "target_days": 120,
        "daily_hours": 2,
        "study_days_per_week": 5
    },
    "AI Product Manager": {
        "goal": "AI Product Manager for SaaS",
        "skills": "Product Strategy, Wireframing, Stakeholder Communication",
        "target_days": 90,
        "daily_hours": 2,
        "study_days_per_week": 4
    }
}

PDF_THEMES = {
    "Corporate": {
        "primary": (33, 37, 41),
        "secondary": (0, 123, 255),
        "table_fill": (224, 234, 248)
    },
    "Studio": {
        "primary": (44, 6, 62),
        "secondary": (255, 99, 71),
        "table_fill": (247, 228, 255)
    },
    "Engineering": {
        "primary": (12, 83, 138),
        "secondary": (0, 173, 181),
        "table_fill": (223, 244, 247)
    }
}

DEFAULT_PRESET = "ML Engineer (NLP)"

if "target_goal" not in st.session_state:
    st.session_state["target_goal"] = PRESET_LIBRARY[DEFAULT_PRESET]["goal"]
if "current_skills" not in st.session_state:
    st.session_state["current_skills"] = PRESET_LIBRARY[DEFAULT_PRESET]["skills"]
if "target_days" not in st.session_state:
    st.session_state["target_days"] = PRESET_LIBRARY[DEFAULT_PRESET]["target_days"]
if "daily_hours" not in st.session_state:
    st.session_state["daily_hours"] = PRESET_LIBRARY[DEFAULT_PRESET]["daily_hours"]
if "study_days_per_week" not in st.session_state:
    st.session_state["study_days_per_week"] = PRESET_LIBRARY[DEFAULT_PRESET]["study_days_per_week"]
if "pdf_theme" not in st.session_state:
    st.session_state["pdf_theme"] = "Corporate"
if "completed_steps" not in st.session_state:
    st.session_state["completed_steps"] = {}


def apply_preset_to_state(preset_key: str):
    preset = PRESET_LIBRARY[preset_key]
    st.session_state["target_goal"] = preset["goal"]
    st.session_state["current_skills"] = preset["skills"]
    st.session_state["target_days"] = preset["target_days"]
    st.session_state["daily_hours"] = preset["daily_hours"]
    st.session_state["study_days_per_week"] = preset["study_days_per_week"]

# --- Helper Functions for Enhancements ---

def get_graph_image_data(graph_app, max_retries: int = 5, retry_delay: float = 2.0):
    """Generates the LangGraph diagram as PNG bytes with retries and a Pyppeteer fallback."""
    graph = graph_app.get_graph()
    last_error = None
    for attempt in range(max_retries):
        try:
            return graph.draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
                curve_style=CurveStyle.LINEAR, 
                node_colors=NodeStyles(first="#ffc0cb", last="#90ee90", default="#add8e6")
            )
        except Exception as e:
            last_error = e
            time.sleep(retry_delay)
    if not PYPPETEER_AVAILABLE:
        st.warning(
            "Mermaid API failed repeatedly and Pyppeteer is not installed. "
            "Install it with `pip install pyppeteer` to enable local rendering."
        )
        return None
    st.warning("Mermaid API failed repeatedly. Trying local renderer (Pyppeteer)...")
    try:
        return graph.draw_mermaid_png(
            draw_method=MermaidDrawMethod.PYPPETEER,
            curve_style=CurveStyle.LINEAR, 
            node_colors=NodeStyles(first="#ffc0cb", last="#90ee90", default="#add8e6")
        )
    except Exception as fallback_error:
        st.error(f"Could not generate LangGraph diagram. Last error: {fallback_error or last_error}")
        return None

def generate_pdf(
    roadmap_rows: List[dict], 
    goal: str, 
    skills: List[str], 
    schedule_summary: dict,
    theme: str
) -> bytes:
    """Generates the roadmap as a professional PDF."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    palette = PDF_THEMES.get(theme, PDF_THEMES["Corporate"])
    
    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(*palette["primary"])
    pdf.cell(0, 10, "Personalized Learning Roadmap", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Goal: {goal}", 0, 1, "C")
    pdf.cell(0, 5, f"Starting Skills: {', '.join(skills)}", 0, 1, "C")
    pdf.cell(
        0, 
        5, 
        f"Timeline Target: {schedule_summary['target_days']} days · {schedule_summary['daily_hours']}h/day over {schedule_summary['study_days_per_week']} days (~{schedule_summary['weekly_capacity']}h/week)", 
        0, 
        1, 
        "C"
    )
    pdf.ln(10)

    # Table Header
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(*palette["table_fill"])
    col_widths = [55, 18, 75, 20, 22] # Total 190mm width
    
    pdf.cell(col_widths[0], 7, "Topic", 1, 0, "L", 1)
    pdf.cell(col_widths[1], 7, "Time", 1, 0, "C", 1)
    pdf.cell(col_widths[2], 7, "Resource", 1, 0, "L", 1)
    pdf.cell(col_widths[3], 7, "Difficulty", 1, 0, "C", 1)
    pdf.cell(col_widths[4], 7, "Confidence", 1, 1, "C", 1)

    # Table Rows
    pdf.set_font("Arial", "", 8)
    pdf.set_auto_page_break(auto=True, margin=12) # Auto page break
    
    for step in roadmap_rows:
        x = pdf.get_x()
        y = pdf.get_y()
        max_height = 5

        topic_label = step['topic']
        if step['is_high_leverage']:
            topic_label = f"[HL] {topic_label}"

        pdf.set_xy(x, y)
        pdf.set_fill_color(*(palette["secondary"] if step['is_high_leverage'] else (255, 255, 255)))
        pdf.multi_cell(col_widths[0], 5, topic_label, 1, "L", bool(step['is_high_leverage']))
        h1 = pdf.get_y() - y

        pdf.set_xy(x + col_widths[0], y)
        pdf.cell(col_widths[1], max_height, f"{step['estimated_time_hours']}h", 1, 0, "C") 
        
        pdf.set_xy(x + col_widths[0] + col_widths[1], y)
        resource_text = step['resource']
        if step.get('resource_url'):
            resource_text = f"{resource_text}\n{step['resource_url']}"
        pdf.multi_cell(col_widths[2], 5, resource_text, 1, "L", 0) 
        h2 = pdf.get_y() - y

        pdf.set_xy(x + col_widths[0] + col_widths[1] + col_widths[2], y)
        pdf.cell(col_widths[3], max_height, step['difficulty'], 1, 0, "C")

        pdf.set_xy(x + sum(col_widths[:4]), y)
        pdf.cell(col_widths[4], max_height, f"{step['confidence']}%", 1, 0, "C")

        pdf.set_y(y + max(h1, h2, max_height))
        pdf.ln(0)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 10)
    if schedule_summary['on_track']:
        pdf.multi_cell(
            0, 
            6, 
            f"Pacing looks good: projected completion in ~{schedule_summary['projected_days']} days "
            f"(target {schedule_summary['target_days']} days). Keep averaging {schedule_summary['daily_hours']}h/day "
            f"across {schedule_summary['study_days_per_week']} days."
        )
    else:
        pdf.multi_cell(
            0, 
            6, 
            f"Pacing alert: you need about {schedule_summary['recommended_daily_hours']}h/day across "
            f"{schedule_summary['study_days_per_week']} days to hit the {schedule_summary['target_days']}-day goal."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        temp_path = tmp_file.name
    try:
        pdf.output(temp_path)
        with open(temp_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
    finally:
        os.remove(temp_path)
    return pdf_bytes


def strip_high_leverage_tag(topic: str) -> Tuple[str, bool]:
    """Return clean topic name and whether it was marked high leverage."""
    normalized = topic.strip()
    tag = "[high leverage]"
    if normalized.lower().startswith(tag):
        # Remove first closing bracket instance
        clean = normalized.split("]", 1)[-1].strip()
        return clean, True
    return normalized, False


def format_roadmap_steps(roadmap_steps: List[RoadmapStep]) -> List[dict]:
    formatted = []
    for step in roadmap_steps:
        clean_topic, is_high = strip_high_leverage_tag(step.topic)
        formatted.append({
            "topic": clean_topic,
            "difficulty": step.difficulty,
            "resource": step.resource,
            "resource_url": getattr(step, "resource_url", ""),
            "estimated_time_hours": step.estimated_time_hours,
            "confidence": getattr(step, "confidence", 70),
            "is_high_leverage": is_high
        })
    return formatted


def summarize_schedule(
    roadmap_rows: List[dict], 
    daily_hours: int, 
    study_days_per_week: int, 
    target_days: int
) -> dict:
    total_hours = sum(step["estimated_time_hours"] for step in roadmap_rows)
    weekly_capacity = daily_hours * study_days_per_week

    if weekly_capacity > 0:
        weeks_needed = total_hours / weekly_capacity
        projected_days = max(1, math.ceil(weeks_needed * 7))
    else:
        weeks_needed = math.inf
        projected_days = math.inf

    available_weeks = target_days / 7 if target_days else 0
    total_sessions = max(1, math.ceil(available_weeks * study_days_per_week)) if available_weeks else study_days_per_week or 1

    recommended_daily_hours = math.ceil(total_hours / total_sessions) if total_sessions else total_hours
    projected_completion_date = (
        date.today() + timedelta(days=projected_days)
        if math.isfinite(projected_days)
        else None
    )
    on_track = math.isfinite(projected_days) and projected_days <= target_days
    timeline_gap = (
        projected_days - target_days if math.isfinite(projected_days) else None
    )

    progress_ratio = (
        min(1.0, target_days / projected_days)
        if math.isfinite(projected_days) and projected_days
        else 1.0
    )

    required_weekly_hours = (
        math.ceil(total_hours / available_weeks) if available_weeks else total_hours
    )

    return {
        "total_hours": total_hours,
        "weekly_capacity": weekly_capacity,
        "projected_days": projected_days if math.isfinite(projected_days) else None,
        "projected_completion_date": projected_completion_date,
        "on_track": on_track,
        "timeline_gap": timeline_gap,
        "progress_ratio": progress_ratio,
        "recommended_daily_hours": recommended_daily_hours,
        "daily_hours": daily_hours,
        "study_days_per_week": study_days_per_week,
        "target_days": target_days,
        "required_weekly_hours": required_weekly_hours,
        "available_weeks": available_weeks,
        "available_hours_window": weekly_capacity * available_weeks,
    }


def generate_weekly_breakdown(
    roadmap_rows: List[dict], weekly_capacity: int
) -> List[dict]:
    if weekly_capacity <= 0:
        return []
    breakdown = []
    current_week_hours = 0
    current_topics = []
    week_number = 1

    for step in roadmap_rows:
        step_hours = step["estimated_time_hours"]
        if current_week_hours + step_hours > weekly_capacity and current_week_hours > 0:
            breakdown.append({
                "week": week_number,
                "hours": current_week_hours,
                "topics": current_topics.copy()
            })
            week_number += 1
            current_week_hours = 0
            current_topics = []
        current_topics.append(step["topic"])
        current_week_hours += step_hours

    if current_topics:
        breakdown.append({
            "week": week_number,
            "hours": current_week_hours,
            "topics": current_topics.copy()
        })

    return breakdown


URL_PATTERN = re.compile(r"(https?://[^\s\]\)]+)")


def extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    match = URL_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def is_youtube_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return "youtube.com" in lowered or "youtu.be" in lowered


def to_youtube_embed(url: str) -> str:
    if not url:
        return ""
    if "watch?v=" in url:
        return url.replace("watch?v=", "embed/")
    if "youtu.be" in url:
        video_id = url.split("/")[-1]
        return f"https://www.youtube.com/embed/{video_id}"
    return url


def normalize_url(url: Optional[str]) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("www."):
        return f"https://{url}"
    return f"https://{url}"


@st.cache_data(show_spinner=False, ttl=3600)
def run_cached_workflow(
    goal: str,
    skills: Tuple[str, ...],
    daily_hours: int,
    study_days_per_week: int,
    target_days: int,
    weekly_hours: int
) -> dict:
    """Runs the LangGraph workflow with caching so repeated inputs are instant."""
    initial_state: PathState = {
        "user_skills": list(skills),
        "target_goal": goal,
        "weekly_hours": weekly_hours,
        "daily_hours": daily_hours,
        "study_days_per_week": study_days_per_week,
        "target_completion_days": target_days,
        "learning_path": [],
        "raw_topic_list": []
    }
    return app.invoke(initial_state)


def generate_weekly_ics(
    weekly_breakdown: List[dict],
    target_goal: str,
    start_date: Optional[date] = None
) -> Optional[str]:
    if not weekly_breakdown:
        return None
    start_date = start_date or date.today()
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//LangGraph Learning Path//EN"
    ]
    for block in weekly_breakdown:
        block_start = start_date + timedelta(days=7 * (block["week"] - 1))
        block_end = block_start + timedelta(days=6)
        uid = uuid.uuid4()
        description = "; ".join(block["topics"])
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}@langgraph",
            f"DTSTAMP:{date.today().strftime('%Y%m%dT%H%M%SZ')}",
            f"DTSTART;VALUE=DATE:{block_start.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{(block_end + timedelta(days=1)).strftime('%Y%m%d')}",
            f"SUMMARY:Week {block['week']} - {target_goal}",
            f"DESCRIPTION:{description}",
            "END:VEVENT"
        ])
    lines.append("END:VCALENDAR")
    return "\n".join(lines)

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="LangGraph Learning Path Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 AI-Powered Personalized Learning Path Generator")
st.markdown("Generates a structured roadmap using **Gemini 2.5 Flash** and a multi-step reasoning workflow orchestrated by **LangGraph**.")

if not llm:
    st.error("🚨 **Configuration Error:** The Gemini LLM is not initialized. Please ensure your `GEMINI_API_KEY` environment variable is set. **This is crucial for Hugging Face deployment.**")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🎯 Your Goals & Skills")

    preset_options = list(PRESET_LIBRARY.keys())
    preset_index = preset_options.index(DEFAULT_PRESET) if DEFAULT_PRESET in preset_options else 0
    preset_choice = st.selectbox(
        "⚡ Quick Persona Preset",
        preset_options,
        index=preset_index
    )
    if st.button("Load Preset", key="load_preset"):
        apply_preset_to_state(preset_choice)
    
    target_goal = st.text_input(
        "Target Career/Goal",
        key="target_goal"
    )
    
    current_skills_input = st.text_area(
        "Your Current Skills (comma-separated, e.g., Python, SQL, Git)",
        key="current_skills"
    )
    
    target_days = st.number_input(
        "Desired Completion Timeline (days)",
        min_value=30,
        max_value=365,
        step=10,
        key="target_days"
    )

    study_days_per_week = st.slider(
        "Study Days per Week",
        min_value=3,
        max_value=7,
        key="study_days_per_week"
    )

    daily_hours = st.slider(
        "Focus Hours per Study Day",
        min_value=1,
        max_value=10,
        key="daily_hours"
    )

    weekly_hours = daily_hours * study_days_per_week
    st.caption(f"≈ {weekly_hours} hours/week available ({daily_hours}h × {study_days_per_week} days)")

    pdf_theme_choice = st.selectbox(
        "PDF Theme",
        list(PDF_THEMES.keys()),
        key="pdf_theme"
    )
    
    generate_button = st.button("🚀 Generate Roadmap", type="primary", disabled=not llm)
    
    st.markdown("---")
    # LangGraph Visualization added to the sidebar for a continuous portfolio pop
    st.subheader("LangGraph Workflow")
    graph_img = get_graph_image_data(app)
    if graph_img:
        st.image(graph_img, caption="LangGraph State Machine Flow", use_column_width=True)
    
    st.markdown("---")
    st.markdown("**80/20 Focus**")
    st.caption("We'll flag the highest-leverage topics so you can start where impact is greatest.")
    with st.sidebar.expander("🧭 Onboarding Wizard", expanded=False):
        st.markdown(
            "- Pick a preset or enter your own goal & skills.\n"
            "- Tell us how fast you want to finish and your real study cadence.\n"
            "- Generate the path, tweak any steps, then track progress.\n"
            "- Cached runs reuse identical inputs instantly so you can iterate fast."
        )
    st.caption("Caching enabled: rerunning with the same inputs returns the saved result instantly.")

# --- Main Area Output ---
if generate_button and llm:
    if not target_goal or not current_skills_input:
        st.error("Please enter a Target Goal and your Current Skills.")
        st.stop()
    
    # Prepare LangGraph Input
    user_skills = [s.strip() for s in current_skills_input.split(',') if s.strip()]
    skills_tuple = tuple(user_skills)
    
    st.subheader("Process Status")
    status_container = st.container()
    
    with status_container:
        st.info("Starting **LangGraph** workflow (cached).")
        with st.spinner("Running LangGraph pipeline..."):
            final_state = run_cached_workflow(
                goal=target_goal,
                skills=skills_tuple,
                daily_hours=daily_hours,
                study_days_per_week=study_days_per_week,
                target_days=target_days,
                weekly_hours=weekly_hours
            )
        topics_needed = final_state.get('raw_topic_list', [])
        st.success(f"1. **Skill Gap Analysis** complete. Identified {len(topics_needed)} topics.")
        st.success("2. **Topic Sequencing** complete. Dependencies ordered.")
        st.success("3. **Resource Finder** complete. Final roadmap generated.")

    # --- Display Final Output ---
    roadmap_data: List[RoadmapStep] = final_state['learning_path']

    if roadmap_data:
        formatted_steps = format_roadmap_steps(roadmap_data)

        st.markdown("### ✏️ Customize & Track")
        editor_ready = [
            {
                "Priority": idx + 1,
                "Topic": step["topic"],
                "Difficulty": step["difficulty"],
                "Recommended Resource": step["resource"],
                "Resource URL": step["resource_url"],
                "Estimated Time (h)": step["estimated_time_hours"],
                "Focus": "High Leverage" if step["is_high_leverage"] else "Supporting",
                "Confidence": step["confidence"]
            }
            for idx, step in enumerate(formatted_steps)
        ]

        edited_df = st.data_editor(
            pd.DataFrame(editor_ready),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Priority": st.column_config.NumberColumn(min_value=1, step=1),
                "Estimated Time (h)": st.column_config.NumberColumn(min_value=1, step=5),
                "Focus": st.column_config.SelectboxColumn(options=["High Leverage", "Supporting"]),
                "Difficulty": st.column_config.SelectboxColumn(options=["Beginner", "Intermediate", "Advanced", "N/A"]),
                "Confidence": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
                "Resource URL": st.column_config.TextColumn(help="Paste full https:// link for the resource")
            },
            key="roadmap_editor"
        )

        editable_records = edited_df.sort_values("Priority").to_dict("records")
        if not editable_records:
            st.warning("Please keep at least one roadmap step.")
            st.stop()
        customized_steps = []
        for record in editable_records:
            topic_name = str(record.get("Topic", "")).strip()
            if not topic_name:
                continue
            estimated_hours = record.get("Estimated Time (h)", 1)
            try:
                estimated_hours = max(1, int(estimated_hours))
            except (ValueError, TypeError):
                estimated_hours = 1
            confidence_score = record.get("Confidence", 70)
            try:
                confidence_score = max(0, min(100, int(confidence_score)))
            except (ValueError, TypeError):
                confidence_score = 70
            resource_url = str(record.get("Resource URL", "")).strip()
            if not resource_url:
                resource_url = extract_first_url(record.get("Recommended Resource", ""))
            resource_url = normalize_url(resource_url) if resource_url else ""
            customized_steps.append({
                "topic": topic_name,
                "difficulty": record.get("Difficulty", "Intermediate"),
                "resource": record.get("Recommended Resource", ""),
                "resource_url": resource_url or "",
                "estimated_time_hours": estimated_hours,
                "is_high_leverage": record.get("Focus", "Supporting") == "High Leverage",
                "confidence": confidence_score
            })
        if not customized_steps:
            st.warning("Your edits removed every topic. Please keep at least one step.")
            st.stop()

        schedule_summary = summarize_schedule(
            customized_steps, daily_hours, study_days_per_week, target_days
        )
        weekly_breakdown = generate_weekly_breakdown(
            customized_steps, schedule_summary['weekly_capacity']
        )

        st.markdown("### 🎬 Roadmap Modules (collapsible)")
        for idx, step in enumerate(customized_steps):
            header = f"{idx + 1}. {step['topic']} · {step['difficulty']} · {step['estimated_time_hours']}h"
            with st.expander(header, expanded=(idx == 0)):
                st.write(
                    "High-leverage focus 💡" if step["is_high_leverage"] else "Supporting concept"
                )
                st.markdown(f"- **Confidence:** {step['confidence']}%")
                resource_url = step.get("resource_url") or extract_first_url(step["resource"])
                resource_url = normalize_url(resource_url) if resource_url else ""
                if resource_url:
                    if is_youtube_url(resource_url):
                        st.video(to_youtube_embed(resource_url))
                    st.link_button("Open resource", resource_url, help="Opens the recommended learning resource in a new tab")
                st.markdown(f"**Resource details:** {step['resource']}")

        st.header(f"✅ Your Personalized Roadmap for: **{target_goal}**")
        
        # Before vs. After Visualization (Skills -> Gaps)
        with st.expander("🔬 See the Gap Analysis (Before vs. After)", expanded=True):
            col_b, col_a = st.columns(2)
            
            col_b.subheader("BEFORE: Your Input")
            col_b.markdown(f"**Goal:** `{target_goal}`")
            col_b.markdown("**Skills Provided:**")
            col_b.code(", ".join(user_skills))
            col_b.markdown("**Timeline Request:**")
            col_b.code(f"{target_days} days · {daily_hours}h/day × {study_days_per_week} days/week")
            
            col_a.subheader("AFTER: Identified Gaps (Topics)")
            col_a.markdown("**Calculated Topics (80/20 aware):**")
            col_a.code(", ".join([step["topic"] for step in customized_steps]))
        
        st.markdown("---")

        total_time = schedule_summary['total_hours']
        projected_completion = (
            schedule_summary['projected_completion_date'].strftime("%b %d")
            if schedule_summary['projected_completion_date']
            else "N/A"
        )

        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Total Study Hours", f"{total_time} h")
        metrics_cols[1].metric(
            "Weekly Capacity", f"{schedule_summary['weekly_capacity']} h/week"
        )
        metrics_cols[2].metric("Projected Completion", projected_completion)

        progress_text = (
            "Timeline on track" if schedule_summary['on_track']
            else "Needs more daily focus"
        )
        st.progress(schedule_summary['progress_ratio'])
        st.caption(progress_text)

        if schedule_summary['on_track']:
            ahead_by = abs(schedule_summary['timeline_gap']) if schedule_summary['timeline_gap'] else 0
            st.success(
                f"You're pacing to finish **~{schedule_summary['projected_days']} days** "
                f"(~{ahead_by} day cushion). Keep averaging {daily_hours} hrs/day."
            )
        else:
            gap_days = schedule_summary['timeline_gap'] if schedule_summary['timeline_gap'] else 0
            st.warning(
                f"At the current pace you'll need ~{schedule_summary['projected_days']} days "
                f"(+{gap_days} days). Aim for **{schedule_summary['recommended_daily_hours']} hrs/day** to stay on track."
            )

        st.markdown("### ✅ Progress Tracker")
        completed_map = st.session_state.get("completed_steps", {})
        active_keys = set()
        remaining_steps = []
        for idx, step in enumerate(customized_steps):
            progress_key = f"{step['topic']}_{idx}"
            active_keys.add(progress_key)
            ui_key = f"progress_{progress_key}"
            if ui_key not in st.session_state:
                st.session_state[ui_key] = completed_map.get(progress_key, False)
            completed = st.checkbox(
                f"{step['topic']} ({step['estimated_time_hours']}h)",
                key=ui_key
            )
            st.session_state["completed_steps"][progress_key] = completed
            if not completed:
                remaining_steps.append(step)

        for stored_key in list(completed_map.keys()):
            if stored_key not in active_keys:
                st.session_state["completed_steps"].pop(stored_key, None)

        completed_count = len(customized_steps) - len(remaining_steps)
        if completed_count == len(customized_steps):
            st.balloons()
            st.success("Amazing! You completed every topic in this roadmap.")
        elif completed_count:
            st.info(f"{completed_count}/{len(customized_steps)} topics checked off. Keep the streak going!")

        remaining_schedule = (
            summarize_schedule(remaining_steps, daily_hours, study_days_per_week, target_days)
            if remaining_steps else None
        )
        if remaining_schedule:
            st.caption(
                f"Remaining load: {remaining_schedule['total_hours']}h · projected finish in "
                f"~{remaining_schedule['projected_days']} days if you keep {daily_hours}h/day."
            )

        priority_tab, schedule_tab, data_tab = st.tabs([
            "🔥 Priority Roadmap (80/20)", 
            "📅 Weekly Pacing Plan", 
            "📊 Full Dataset"
        ])

        high_leverage_steps = [step for step in customized_steps if step["is_high_leverage"]]

        with priority_tab:
            st.subheader("High-Leverage Topics (Top 20%)")
            if high_leverage_steps:
                for step in high_leverage_steps:
                    st.markdown(
                        f"**{step['topic']}** · {step['difficulty']}  \n"
                        f"{step['resource']}  \n"
                        f"_Est. {step['estimated_time_hours']} hrs_"
                    )
                    st.divider()
            else:
                st.info("No topics were flagged as high leverage. Try regenerating with a narrower goal.")

            st.subheader("Supporting Topics")
            supporting = [step for step in customized_steps if not step["is_high_leverage"]]
            if supporting:
                st.markdown(
                    ", ".join([f"`{step['topic']}`" for step in supporting])
                )

        with schedule_tab:
            st.subheader("Weekly Momentum Plan")
            st.markdown(
                f"- Weekly capacity: **{schedule_summary['weekly_capacity']} hrs** "
                f"({daily_hours} hrs × {study_days_per_week} days)"
            )
            st.markdown(
                f"- Required to hit goal: **~{schedule_summary['required_weekly_hours']} hrs/week**"
            )
            if weekly_breakdown:
                for week in weekly_breakdown[:6]:  # show first 6 weeks
                    st.markdown(
                        f"**Week {week['week']}** · {week['hours']} hrs  \n"
                        f"{', '.join(week['topics'])}"
                    )
                ics_payload = generate_weekly_ics(weekly_breakdown, target_goal)
                if ics_payload:
                    st.download_button(
                        "📥 Add weekly plan to calendar (.ics)",
                        data=ics_payload,
                        file_name=f"{target_goal.replace(' ', '_')}_weekly_plan.ics",
                        mime="text/calendar"
                    )
            else:
                st.info("Add more study hours to see a week-by-week plan.")

        with data_tab:
            df = pd.DataFrame([
                {
                    "Topic": step["topic"],
                    "Difficulty": step["difficulty"],
                    "Recommended Resource": step["resource"],
                    "Resource URL": step["resource_url"],
                    "Estimated Time (h)": step["estimated_time_hours"],
                    "Focus": "High Leverage" if step["is_high_leverage"] else "Supporting",
                    "Confidence": f"{step['confidence']}%"
                }
                for step in customized_steps
            ])

            def color_difficulty(val):
                if val == 'Beginner': return 'background-color: #d4edda'
                if val == 'Intermediate': return 'background-color: #fff3cd'
                if val == 'Advanced': return 'background-color: #f8d7da'
                return ''

            def color_focus(val):
                if val == "High Leverage": return 'background-color: #ffe5ec'
                return ''

            st.dataframe(
                df.style
                .applymap(color_difficulty, subset=['Difficulty'])
                .applymap(color_focus, subset=['Focus']),
                hide_index=True, 
                use_container_width=True
            )

        pdf_buffer = generate_pdf(
            customized_steps, 
            target_goal, 
            user_skills, 
            schedule_summary,
            pdf_theme_choice
        )
        st.download_button(
            label="⬇️ Download Roadmap as Professional PDF",
            data=pdf_buffer,
            file_name=f"Learning_Path_{target_goal.replace(' ', '_').replace('/', '_')}.pdf",
            mime="application/pdf",
            type="secondary"
        )
        
    else:
        st.warning("The generator finished but returned an empty path. Please check the console for LLM errors and ensure the input is clear.")