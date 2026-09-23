import time
import traceback

import os
import streamlit as st

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ[key] = value.strip()
except Exception:
    pass

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain
from pipeline import message_text

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

STEPS = [
    ("🔍", "Search Agent", "Finds recent, reliable sources"),
    ("📖", "Reader Agent", "Scrapes the most relevant page"),
    ("✍️", "Writer", "Drafts the research report"),
    ("🧐", "Critic", "Reviews and gives feedback"),
]

EXAMPLE_TOPICS = [
    "Latest advances in solid-state batteries",
    "Impact of generative AI on software engineering jobs",
    "State of quantum computing in 2026",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "elapsed" not in st.session_state:
    st.session_state.elapsed = {}
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""


def set_example(text: str):
    st.session_state.topic_input = text


# ---------------------------------------------------------------------------
# Pipeline (same logic as pipeline.py, but reports progress to the UI)
# ---------------------------------------------------------------------------
def collect_tool_output(result: dict) -> str:
    """Join tool messages if present, otherwise fall back to the last message."""
    tool_messages = [
        m for m in result["messages"] if getattr(m, "type", "") == "tool"
    ]
    messages_to_use = tool_messages or result["messages"][-1:]
    return "\n\n".join(message_text(m.content) for m in messages_to_use)


def run_pipeline_with_ui(topic: str) -> dict:
    state: dict = {}
    timings: dict = {}

    progress = st.progress(0, text="Starting pipeline...")

    # Step 1 - Search
    with st.status("🔍 Step 1 — Search agent is working...", expanded=True) as status:
        t0 = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = collect_tool_output(search_result)
        timings["Search"] = time.time() - t0
        st.write("Search complete.")
        status.update(label="🔍 Step 1 — Search complete", state="complete", expanded=False)
    progress.progress(25, text="Search done. Reading top sources...")

    # Step 2 - Reader
    with st.status("📖 Step 2 — Reader agent is scraping top resources...", expanded=True) as status:
        t0 = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about {topic} "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = collect_tool_output(reader_result)
        timings["Reader"] = time.time() - t0
        st.write("Scraping complete.")
        status.update(label="📖 Step 2 — Scraping complete", state="complete", expanded=False)
    progress.progress(50, text="Sources read. Drafting report...")

    # Step 3 - Writer
    with st.status("✍️ Step 3 — Writer is drafting the report...", expanded=True) as status:
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}\n\n"
        )
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        timings["Writer"] = time.time() - t0
        st.write("Draft complete.")
        status.update(label="✍️ Step 3 — Report drafted", state="complete", expanded=False)
    progress.progress(75, text="Draft ready. Critic is reviewing...")

    # Step 4 - Critic
    with st.status("🧐 Step 4 — Critic is reviewing the report...", expanded=True) as status:
        t0 = time.time()
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        timings["Critic"] = time.time() - t0
        st.write("Review complete.")
        status.update(label="🧐 Step 4 — Review complete", state="complete", expanded=False)
    progress.progress(100, text="All done ✅")

    st.session_state.elapsed = timings
    return state


def as_text(value) -> str:
    """Chains may return strings or message objects; normalise to text."""
    if hasattr(value, "content"):
        return message_text(value.content)
    return message_text(value)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Research System")
    st.caption("A multi-agent pipeline: search → read → write → critique.")

    st.markdown("### Pipeline")
    for icon, name, desc in STEPS:
        st.markdown(f"**{icon} {name}**  \n{desc}")

    st.divider()
    st.markdown("### Try an example")
    for i, ex in enumerate(EXAMPLE_TOPICS):
        st.button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,), use_container_width=True)

    if st.session_state.state:
        st.divider()
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.state = None
            st.session_state.topic = ""
            st.session_state.elapsed = {}
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Multi-Agent Research System")
st.write("Enter a topic and let the agents research it and produce a reviewed report.")

with st.form("research_form", clear_on_submit=False):
    topic = st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="e.g. Latest advances in solid-state batteries",
    )
    submitted = st.form_submit_button("🚀 Run research", type="primary")

if submitted:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.state = None
        try:
            result = run_pipeline_with_ui(topic.strip())
            st.session_state.state = result
            st.session_state.topic = topic.strip()
        except Exception as exc:
            st.error(f"The pipeline failed: {exc}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
state = st.session_state.state
if state:
    st.divider()
    st.subheader(f"Results for: {st.session_state.topic}")

    if st.session_state.elapsed:
        cols = st.columns(len(st.session_state.elapsed) + 1)
        total = 0.0
        for col, (name, secs) in zip(cols, st.session_state.elapsed.items()):
            col.metric(name, f"{secs:.1f}s")
            total += secs
        cols[-1].metric("Total", f"{total:.1f}s")

    report_text = as_text(state["report"])
    feedback_text = as_text(state["feedback"])

    tab_report, tab_feedback, tab_search, tab_scrape = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Search Results", "📖 Scraped Content"]
    )

    with tab_report:
        st.markdown(report_text)
        st.download_button(
            "⬇️ Download report (.md)",
            data=report_text,
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(feedback_text)

    with tab_search:
        st.text_area("Raw search output", state["search_results"], height=400)

    with tab_scrape:
        st.text_area("Raw scraped content", state["scraped_content"], height=400)