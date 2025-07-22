#! /usr/bin/env python

import operator
import os
import asyncio
import time
import json
import logging
from typing import Annotated, TypedDict, Union, List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re # Added this import
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.agents import AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enhanced Web Browser Tools Setup ---

# Initialize Google Search with better error handling



# --- Enhanced Models with Better Configuration ---

# Model configurations with optimized parameters
# Initialize models with better configurations
# model_rina = ChatOllama(
#     model="llama3.2:3b",
#     streaming=True,
#     # base_url="https://saved-sympathy-mart-period.trycloudflare.com/"
# )

# model_emilia = ChatOllama(
#     model="llama3-2.3b:latest", 
#     streaming=True,
#     # base_url="https://saved-sympathy-mart-period.trycloudflare.com/"
# )



# model_supervisor = ChatOllama(
#     model="hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
#     streaming=True,
#     # base_url="https://saved-sympathy-mart-period.trycloudflare.com/"
# )

# import os
# from dotenv import load_dotenv

# # Load environment variables from a .env file
# load_dotenv()




# # Get API keys from environment variables (now loaded from .env if present)
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search = GoogleSerperAPIWrapper(
    serper_api_key=SERPER_API_KEY
)

model_rina = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
)

model_emilia = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 

)

model_supervisor = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",

)


search = GoogleSerperAPIWrapper()

def google_search(query: str) -> str:
    # Run the search multiple times for consistency
    try:
        result = search.run(query)
        return result
    except Exception as e:
        return f"Error during Google Search: {str(e)}"

web_tools = [
    Tool(
        name="google_search",
        description="Search Google for the most accurate, up-to-date information. Results are verified for consistency.",
        func=google_search,
    ),
]

# --- Enhanced Difficulty Assessment ---

def is_difficult_question(question: str) -> Dict[str, Union[bool, str, List[str], int]]:
    """Enhanced difficulty assessment with detailed analysis."""
    difficult_keywords = [
        "current", "latest", "recent", "today", "2024", "2025", "news",
        "stock price", "weather", "events", "what happened", "when did",
        "latest version", "recent developments", "breaking news",
        "research", "study", "scientific", "technical", "complex",
        "detailed explanation", "in-depth", "comprehensive", "compare",
        "analysis", "statistics", "data", "trends", "forecast",
        "implementation", "code", "programming", "algorithm", "architecture"
    ]
    
    question_lower = question.lower()
    detected_keywords = []
    complexity_score = 0
    
    # Check for difficult keywords
    for keyword in difficult_keywords:
        if keyword in question_lower:
            detected_keywords.append(keyword)
            complexity_score += 1
    
    # Check for question complexity
    word_count = len(question.split())
    if word_count > 15:
        complexity_score += 2
    elif word_count > 10:
        complexity_score += 1
    
    # Check for multiple questions
    question_count = question.count('?')
    if question_count > 1:
        complexity_score += 2
    
    # Check for technical terms
    technical_terms = ["api", "sdk", "framework", "library", "database", "server", "client", "protocol"]
    for term in technical_terms:
        if term in question_lower:
            complexity_score += 1
            detected_keywords.append(term)
    
    # Determine if tools are needed
    use_tools = complexity_score >= 2
    
    return {
        "use_tools": use_tools,
        "complexity_score": complexity_score,
        "detected_keywords": detected_keywords,
        "word_count": word_count,
        "question_count": question_count
    }

# --- Enhanced Agent Prompts ---

# Rina: Enhanced General Knowledge Agent
rina_simple_prompt = PromptTemplate.from_template(
    """
Riwayat Percakapan:
{chat_history}

Permintaan Pengguna:
{input}

Tanggapan Anda (jawab dalam Bahasa Indonesia):"""
)

rina_tool_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
Saat menggunakan alat:
- Cari informasi yang paling relevan dan terbaru
- Verifikasi fakta dari beberapa sumber jika memungkinkan
- Sintesis informasi dari berbagai sumber
- Berikan konteks dan jelaskan signifikansi temuan

Selalu balas dalam Bahasa Indonesia."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Emilia: Enhanced Technical Implementation Agent
emilia_simple_prompt = PromptTemplate.from_template(
    """

Riwayat Percakapan:
{chat_history}

Permintaan Pengguna:
{input}

Tanggapan Anda (jawab dalam Bahasa Indonesia):"""
)

emilia_tool_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
Saat menggunakan alat:
- Cari dokumentasi resmi dan spesifikasi terkait
- Cari praktik terbaik dan rekomendasi dari komunitas
- Verifikasi kompatibilitas versi dan persyaratan
- Temukan contoh kerja dan tutorial

Selalu balas dalam Bahasa Indonesia."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Enhanced Supervisor Prompt (Bahasa Indonesia)
supervisor_routing_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Anda adalah supervisor cerdas yang mengelola tim berisi dua agen spesialis:

PANDUAN ROUTING:
- Rute ke Rina untuk: ["sains", "teknologi", "humaniora", "seni", "bisnis", "riset", "analisis"]
- Rute ke Emilia untuk: ["pemrograman", "implementasi", "teknis", "arsitektur", "pengembangan"]

FORMAT KEPUTUSAN:
- Untuk merutekan: "ROUTE_TO: [NamaAgen] - [Alasan singkat dan langsung untuk routing]"
- Untuk menyelesaikan: "FINISH - [Kesimpulan komprehensif yang merangkum poin-poin utama]"

PENTING: Selalu rute ke anggota tim jika tugas memerlukan pekerjaan lanjutan. Hanya gunakan FINISH jika semua aspek sudah sepenuhnya dijawab. Pertimbangkan keahlian dan beban kerja/keseimbangan dalam keputusan Anda.

Beban Kerja Saat Ini:
Rina: {rina_workload} tugas
Emilia: {emilia_workload} tugas

Riwayat Percakapan:
{messages}

Selalu balas dalam Bahasa Indonesia."""),
    ("user", "{input}"),
])

# --- Enhanced Agent Creation ---

# Create tool-calling agents with better error handling
rina_tool_agent = create_tool_calling_agent(model_rina, web_tools, rina_tool_prompt)
rina_tool_executor = AgentExecutor(
    agent=rina_tool_agent, 
    tools=web_tools, 
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate"
)

emilia_tool_agent = create_tool_calling_agent(model_emilia, web_tools, emilia_tool_prompt)
emilia_tool_executor = AgentExecutor(
    agent=emilia_tool_agent, 
    tools=web_tools, 
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate"
)

# Enhanced simple runnables with better streaming
rina_simple_runnable = rina_simple_prompt | model_rina | StrOutputParser()
emilia_simple_runnable = emilia_simple_prompt | model_emilia | StrOutputParser()

# --- Enhanced Graph State ---

class AgentState(TypedDict):
    messages: Annotated[list[Union[HumanMessage, AIMessage]], operator.add]
    next_agent: str
    metadata: Dict[str, Any]  # For tracking conversation metadata

# --- Enhanced Agent Nodes ---

def agent_node(state: AgentState, agent_simple, agent_tools, name: str):
    """Enhanced agent node with better error handling, performance optimization, and true streaming."""
    import threading
    import queue
    current_question = state["messages"][-1].content
    difficulty_analysis = is_difficult_question(current_question)
    use_tools = difficulty_analysis["use_tools"]
    if name in ["Rina", "Emilia"]:
        use_tools = True
    logger.info(f"[{name}] Processing question: {current_question[:50]}...")
    logger.info(f"[{name}] Difficulty analysis: {difficulty_analysis}")
    start_time = time.time()
    metadata = state.get("metadata", {})
    # Streaming logic
    def stream_tool_agent():
        full_content = ""
        tool_started = False
        tool_finished = False
        tool_queue = queue.Queue()
        def tool_worker():
            try:
                for chunk in agent_tools.stream({
                    "input": current_question,
                    "chat_history": [msg for msg in state["messages"][:-1]]
                }):
                    if chunk and isinstance(chunk, dict) and "output" in chunk:
                        tool_queue.put(chunk["output"])
                tool_queue.put(None)  # Signal end
            except Exception as e:
                tool_queue.put(e)
        t = threading.Thread(target=tool_worker)
        t.start()
        # Wait for first output or timeout
        try:
            first = tool_queue.get(timeout=2)
            if first is None:
                tool_finished = True
            elif isinstance(first, Exception):
                raise first
            else:
                # Yield system message: tool started
                yield {"type": "system", "content": f"{name} is searching for information...", "sender": "system"}
                tool_started = True
                full_content += first
                yield {"type": "chunk", "content": first, "sender": name}
        except queue.Empty:
            # Tool is slow, yield system message
            yield {"type": "system", "content": f"{name} is still searching, please wait...", "sender": "system"}
        # Continue streaming rest
        while not tool_finished:
            next_item = tool_queue.get()
            if next_item is None:
                tool_finished = True
                break
            elif isinstance(next_item, Exception):
                raise next_item
            else:
                full_content += next_item
                yield {"type": "chunk", "content": next_item, "sender": name}
        # Yield system message: tool finished
        yield {"type": "system", "content": f"{name} has finished searching.", "sender": "system"}
        yield {"type": "complete", "content": full_content, "sender": name}
    def stream_simple_agent():
        full_content = ""
        for chunk in agent_simple.stream({
            "input": current_question,
            "chat_history": state["messages"],
        }):
            if chunk:
                full_content += chunk
                yield {"type": "chunk", "content": chunk, "sender": name}
        yield {"type": "complete", "content": full_content, "sender": name}
    try:
        if use_tools:
            logger.info(f"[{name}] Using tools for difficult question (streaming)")
            for item in stream_tool_agent():
                yield item
        else:
            logger.info(f"[{name}] Using simple mode (streaming)")
            for item in stream_simple_agent():
                yield item
        processing_time = time.time() - start_time
        metadata[f"{name.lower()}_processing_time"] = processing_time
        metadata[f"{name.lower()}_used_tools"] = use_tools
        metadata[f"{name.lower()}_difficulty_score"] = difficulty_analysis["complexity_score"]
        return {
            "messages": [AIMessage(content="", name=name)],  # Content is streamed, so leave blank
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"[{name}] Critical error: {str(e)}")
        error_message = f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question or ask for a different approach."
        yield {"type": "system", "content": error_message, "sender": "system-error"}
        return {
            "messages": [AIMessage(content=error_message, name=name)],
            "metadata": {"error": str(e)}
        }

# --- Enhanced Supervisor Node ---

members = ["Rina", "Emilia"]

def supervisor_node(state: AgentState):
    """Enhanced supervisor node with dynamic, fair task distribution."""
    last_message = state["messages"][-1].content
    metadata = state.get("metadata", {})
    # Defensive: always provide integer values for workloads
    rina_workload = int(metadata.get("rina_workload", 0) or 0)
    emilia_workload = int(metadata.get("emilia_workload", 0) or 0)
    last_agent = metadata.get("last_agent", None)

    logger.info("[Supervisor] Making routing decision")
    start_time = time.time()

    try:
        supervisor_input = {
            "messages": state["messages"],
            "input": last_message,
            "members": ", ".join(members),
            "rina_workload": rina_workload,
            "emilia_workload": emilia_workload,
        }
        full_decision = ""
        for chunk in model_supervisor.stream(supervisor_routing_prompt.format_messages(**supervisor_input)):
            if hasattr(chunk, 'content') and chunk.content:
                full_decision += chunk.content

        # Remove <think>...</think> tags
        cleaned_content = re.sub(r"<think>.*?</think>\n?", "", full_decision, flags=re.DOTALL).strip()

        supervisor_message = AIMessage(content=cleaned_content, name="Supervisor")
        next_agent = _parse_supervisor_decision(cleaned_content)

        # Workload balancing: if ambiguous, alternate or pick less-busy agent
        if next_agent not in members and next_agent != "FINISH":
            # Alternate if last_agent exists, else pick less-busy
            if last_agent == "Rina":
                next_agent = "Emilia"
            elif last_agent == "Emilia":
                next_agent = "Rina"
            else:
                next_agent = "Rina" if rina_workload <= emilia_workload else "Emilia"

        processing_time = time.time() - start_time
        logger.info(f"[Supervisor] Decision made in {processing_time:.2f}s: {next_agent}")

        # Update workload
        if next_agent == "Rina":
            rina_workload += 1
        elif next_agent == "Emilia":
            emilia_workload += 1
        if next_agent in members:
            metadata["last_agent"] = next_agent
        metadata["rina_workload"] = rina_workload
        metadata["emilia_workload"] = emilia_workload
        metadata["supervisor_processing_time"] = processing_time
        metadata["supervisor_decision"] = next_agent

        return {
            "messages": [supervisor_message],
            "next_agent": next_agent,
            "metadata": metadata
        }

    except Exception as e:
        logger.error(f"[Supervisor] Error in decision making: {str(e)}")
        # Default to alternate or less-busy agent
        if last_agent == "Rina":
            fallback_agent = "Emilia"
        elif last_agent == "Emilia":
            fallback_agent = "Rina"
        else:
            fallback_agent = "Rina" if rina_workload <= emilia_workload else "Emilia"
        return {
            "messages": [AIMessage(content=f"ROUTE_TO: {fallback_agent} - Default routing due to decision error", name="Supervisor")],
            "next_agent": fallback_agent,
            "metadata": {"error": str(e)}
        }

def _enhanced_supervisor_decision(state: AgentState):
    """Enhanced supervisor decision with better prompt engineering."""
    supervisor_input = {
        "messages": state["messages"],
        "input": state["messages"][-1].content,
        "members": ", ".join(members),
    }
    
    full_decision = ""
    for chunk in model_supervisor.stream(supervisor_routing_prompt.format_messages(**supervisor_input)):
        if hasattr(chunk, 'content') and chunk.content:
            full_decision += chunk.content
    
    return full_decision

def _parse_supervisor_decision(decision: str) -> str:
    """Enhanced decision parsing with better logic."""
    decision_lower = decision.lower()
    
    # Check for explicit routing
    if "route_to:" in decision_lower:
        for member in members:
            if member.lower() in decision_lower:
                return member
    
    # Check for finish
    if "finish" in decision_lower:
        return "FINISH"
    
    # Fallback: check for agent names in the decision
    for member in members:
        if member.lower() in decision_lower:
            return member
    
    # Default to Rina for general questions
    return "Rina"

# --- Enhanced Graph Construction ---

builder = StateGraph(AgentState)

# Add nodes with enhanced error handling
builder.add_node("Rina", lambda state: agent_node(state, rina_simple_runnable, rina_tool_executor, "Rina"))
builder.add_node("Emilia", lambda state: agent_node(state, emilia_simple_runnable, emilia_tool_executor, "Emilia"))
builder.add_node("Supervisor", supervisor_node)

# Add edges
for member in members:
    builder.add_edge(member, "Supervisor")

builder.add_conditional_edges(
    "Supervisor",
    lambda state: state.get("next_agent"),
    {"Rina": "Rina", "Emilia": "Emilia", "FINISH": END},
)

builder.set_entry_point("Supervisor")

app = builder.compile()

# --- Performance Monitoring ---

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0

performance_monitor = PerformanceMonitor()

# --- Enhanced Execution ---

if __name__ == "__main__":
    question = "jadwal rilis iron heart"
    initial_input = HumanMessage(content=question)
    inputs = {"messages": [initial_input], "metadata": {}}
    
    num_printed_messages = 0
    
    print(f"User:\n{initial_input.content}")
    print("\n--- Starting Enhanced Multi-Agent Conversation ---")
    
    start_time = time.time()
    
    for state in app.stream(inputs, stream_mode="values"):
        current_messages = state['messages']
        
        if len(current_messages) > num_printed_messages:
            new_messages = current_messages[num_printed_messages:]
            
            for msg in new_messages:
                if isinstance(msg, HumanMessage):
                    print(f"\n{msg.name if msg.name else 'User'}:")
                    print(msg.content)
                elif isinstance(msg, AIMessage):
                    print(f"\n{msg.name}:")
                    print(msg.content)
            
            num_printed_messages = len(current_messages)
        
        if "next_agent" in state:
            if state["next_agent"] == "FINISH":
                print("\n--- Supervisor decided to FINISH the conversation ---")
                break
            elif state["next_agent"] in members:
                print(f"\n--- Supervisor routing to: {state['next_agent']} ---")
    
    total_time = time.time() - start_time
    print(f"\n--- Enhanced Conversation Complete in {total_time:.2f}s ---")
    
    # Print performance metrics
    if "metadata" in state:
        print(f"\nPerformance Metrics:")
        for key, value in state["metadata"].items():
            print(f"  {key}: {value}")