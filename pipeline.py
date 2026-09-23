from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

def run_research_pipeline(topic: str) -> dict:
    state = {}

    # Search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    tool_messages = [
        message for message in search_result["messages"]
        if getattr(message, "type", "") == "tool"
    ]
    messages_to_use = tool_messages or search_result["messages"][-1:]
    state["search_results"] = "\n\n".join(
        message_text(message.content) for message in messages_to_use
    )
    print("\nSearch result\n", state["search_results"])


    # Reader agent working
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about {topic}"
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    reader_tool_messages = [
        message for message in reader_result["messages"]
        if getattr(message, "type", "") == "tool"
    ]
    messages_to_use = reader_tool_messages or reader_result["messages"][-1:]
    state["scraped_content"] = "\n\n".join(
        message_text(message.content) for message in messages_to_use
    )
    print("\nScraped content\n", state["scraped_content"])


    # Writer chain 
    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}\n\n"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFinal Report\n",state["report"])


    # Critic report 
    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCritic Report\n",state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)