#! /usr/bin/env python
from PIL import Image # For handling image files
import os

import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()
import operator
import os
import time
import json
import logging
from typing import Annotated, TypedDict, Union, List, Dict, Any
import re # Added this import
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

from os import getenv
from baseUrl import ollama_url
nama_model_Riset = getenv("MODEL_RISET")
nama_model_Implementasi = getenv("MODEL_IMPLEMENTASI")
nama_model_supervisor = getenv("MODEL_SUPERVISOR")


# Model configurations with optimized parameters
# Initialize models with better configurations
model_Riset = ChatOllama(
    model=nama_model_Riset,
    streaming=True,
    base_url=ollama_url
    # base_url="https://saved-sympathy-mart-period.trycloudflare.com/"
)

model_Implementasi = ChatOllama(
    model=nama_model_Implementasi, 
    streaming=True,
    base_url=ollama_url
    # base_url="https://saved-sympathy-mart-period.trycloudflare.com/"
)



model_supervisor = ChatOllama(
    model=nama_model_supervisor,
    streaming=True,
    base_url=ollama_url
    # base_url="https://saved-sympathy-mart-period.trycloudflare.com/"
)





# import os
# from dotenv import load_dotenv

# # Load environment variables from a .env file
# load_dotenv()

import os
import base64
import io
import json
from PIL import Image
from openai import OpenAI
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

# Load environment variables
nebius_api_key = os.getenv("NEBIUS_API_KEY")
image_generation_model = os.getenv("IMAGE_GENERATION")




# # Get API keys from environment variables (now loaded from .env if present)
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search = GoogleSerperAPIWrapper(
    serper_api_key=SERPER_API_KEY
)


if not nebius_api_key:
    raise ValueError("NEBIUS_API_KEY environment variable is not set.")
if not image_generation_model or not isinstance(image_generation_model, str):
    raise ValueError("IMAGE_GENERATION environment variable is not set or is not a string.")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=nebius_api_key
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
        "terkini", "terbaru", "hari ini", "2024", "2025", "berita",
        "harga saham", "cuaca", "acara", "apa yang terjadi", "kapan",
        "versi terbaru", "perkembangan terbaru", "berita terkini",
        "riset", "studi", "ilmiah", "teknis", "kompleks",
        "penjelasan detail", "mendalam", "komprehensif", "bandingkan",
        "analisis", "statistik", "data", "tren", "prediksi",
        "implementasi", "kode", "pemrograman", "algoritma", "arsitektur"
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

# Riset: Enhanced General Knowledge Agent
Riset_simple_prompt = PromptTemplate.from_template(
    """
Riwayat Percakapan:
{chat_history}

Permintaan Pengguna:
{input}

Tanggapan Anda (jawab dalam Bahasa Indonesia):"""
)

Riset_tool_prompt = ChatPromptTemplate.from_messages([
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


# Implementasi: Enhanced Technical Implementation Agent
Implementasi_simple_prompt = PromptTemplate.from_template(
    """

Riwayat Percakapan:
{chat_history}

Permintaan Pengguna:
{input}

Tanggapan Anda (jawab dalam Bahasa Indonesia):"""
)

Implementasi_tool_prompt = ChatPromptTemplate.from_messages([
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
- Rute ke Riset untuk: ["sains", "teknologi", "humaniora", "seni", "bisnis", "riset", "analisis"]
- Rute ke Implementasi untuk: ["pemrograman", "implementasi", "teknis", "arsitektur", "pengembangan"]
- Rute ke Creator untuk: ["pembuatan gambar", "membuat gambar", "menghasilkan gambar", "gambar", "buat gambar"]

FORMAT KEPUTUSAN:
- Untuk merutekan: "ROUTE_TO: [NamaAgen] - [Alasan singkat dan langsung untuk routing]"
- Untuk menyelesaikan: "FINISH - [Kesimpulan komprehensif yang merangkum poin-poin utama]"

PENTING: Selalu rute ke anggota tim jika tugas memerlukan pekerjaan lanjutan. Hanya gunakan FINISH jika semua aspek sudah sepenuhnya dijawab. Pertimbangkan keahlian dan beban kerja/keseimbangan dalam keputusan Anda.

Beban Kerja Saat Ini:
Riset: {Riset_workload} tugas
Implementasi: {Implementasi_workload} tugas

Riwayat Percakapan:
{messages}

Selalu balas dalam Bahasa Indonesia."""),
    ("user", "{input}"),
])

# --- Enhanced Agent Creation ---

# Create tool-calling agents with better error handling
Riset_tool_agent = create_tool_calling_agent(model_Riset, web_tools, Riset_tool_prompt)
Riset_tool_executor = AgentExecutor(
    agent=Riset_tool_agent, 
    tools=web_tools, 
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate"
)

Implementasi_tool_agent = create_tool_calling_agent(model_Implementasi, web_tools, Implementasi_tool_prompt)
Implementasi_tool_executor = AgentExecutor(
    agent=Implementasi_tool_agent, 
    tools=web_tools, 
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate"
)

# Enhanced simple runnables with better streaming
Riset_simple_runnable = Riset_simple_prompt | model_Riset | StrOutputParser()
Implementasi_simple_runnable = Implementasi_simple_prompt | model_Implementasi | StrOutputParser()

# Creator: Image Generation Agent
creator_simple_prompt = PromptTemplate.from_template(
    """Anda adalah agen pembuat gambar. Tugas Anda adalah menghasilkan gambar berdasarkan deskripsi pengguna.
Riwayat Percakapan:
{chat_history}

Permintaan Pengguna:
{input}

Tanggapan Anda (jawab dalam Bahasa Indonesia):"""
)

creator_simple_runnable = creator_simple_prompt | model_Implementasi | StrOutputParser() # Using model_Implementasi for now, can be changed if a specific image generation model is needed later.

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
    if name in ["Riset", "Implementasi"]:
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

def creator_agent_node(state: AgentState, agent_simple, name: str):
    """Creator agent node for image generation."""
    current_question = state["messages"][-1].content
    logger.info(f"[{name}] Processing image generation request: {current_question[:50]}...")
    start_time = time.time()
    metadata = state.get("metadata", {})
    from deep_translator import GoogleTranslator
    translate = GoogleTranslator(source='auto', target='en').translate(current_question)
    logger.info(f"[{name}] Translating image generation request: {translate}")

    prompt = translate # Use the user\'s prompt for image generation

    try:
        response = client.images.generate(
            model=image_generation_model,
            response_format="b64_json",
            extra_body={
                "response_extension": "png",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "negative_prompt": "",
                "seed": -1
            },
            prompt=prompt
        )

        # Parse the JSON response
        image_data_json = response.to_json()
        image_data = json.loads(image_data_json)

        # Extract and decode the base64 image
        b64_image = image_data['data'][0]['b64_json']
        decoded_image_bytes = base64.b64decode(b64_image)
        image_stream = io.BytesIO(decoded_image_bytes)

        # Save image to static\image_generation
        output_dir = os.path.join("static", "image_generation")
        os.makedirs(output_dir, exist_ok=True)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"generated_image_{timestamp}.png")
        image = Image.open(image_stream)
        image.save(output_path)
        
        success_message = f"Gambar berhasil dibuat dan disimpan di: {output_path}"
        yield {"type": "complete", "content": success_message, "sender": name}

        processing_time = time.time() - start_time
        metadata[f"{name.lower()}_processing_time"] = processing_time
        metadata[f"{name.lower()}_output_path"] = output_path
        
        return {
            "messages": [AIMessage(content=success_message, name=name)],
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"[{name}] Critical error during image generation: {str(e)}")
        error_message = f"Saya mohon maaf, tetapi terjadi kesalahan saat membuat gambar: {str(e)}. Mohon coba lagi atau berikan deskripsi yang berbeda."
        yield {"type": "system", "content": error_message, "sender": "system-error"}
        return {
            "messages": [AIMessage(content=error_message, name=name)],
            "metadata": {"error": str(e)}
        }

# --- Enhanced Supervisor Node ---

members = ["Riset", "Implementasi", "Creator"]

def supervisor_node(state: AgentState):
    """Enhanced supervisor node with dynamic, fair task distribution."""
    last_message = state["messages"][-1].content
    metadata = state.get("metadata", {})
    # Defensive: always provide integer values for workloads
    Riset_workload = int(metadata.get("Riset_workload", 0) or 0)
    Implementasi_workload = int(metadata.get("Implementasi_workload", 0) or 0)
    last_agent = metadata.get("last_agent", None)

    logger.info("[Supervisor] Making routing decision")
    start_time = time.time()

    try:
        supervisor_input = {
            "messages": state["messages"],
            "input": last_message,
            "members": ", ".join(members),
            "Riset_workload": Riset_workload,
            "Implementasi_workload": Implementasi_workload,
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
            if last_agent == "Riset":
                next_agent = "Implementasi"
            elif last_agent == "Implementasi":
                next_agent = "Riset"
            else:
                next_agent = "Riset" if Riset_workload <= Implementasi_workload else "Implementasi"

        processing_time = time.time() - start_time
        logger.info(f"[Supervisor] Decision made in {processing_time:.2f}s: {next_agent}")

        # Update workload
        if next_agent == "Riset":
            Riset_workload += 1
        elif next_agent == "Implementasi":
            Implementasi_workload += 1
        if next_agent in members:
            metadata["last_agent"] = next_agent
        metadata["Riset_workload"] = Riset_workload
        metadata["Implementasi_workload"] = Implementasi_workload
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
        if last_agent == "Riset":
            fallback_agent = "Implementasi"
        elif last_agent == "Implementasi":
            fallback_agent = "Riset"
        else:
            fallback_agent = "Riset" if Riset_workload <= Implementasi_workload else "Implementasi"
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
    
    # Default to Riset for general questions
    return "Riset"

# --- Enhanced Graph Construction ---

builder = StateGraph(AgentState)

# Add nodes with enhanced error handling
builder.add_node("Riset", lambda state: agent_node(state, Riset_simple_runnable, Riset_tool_executor, "Riset"))
builder.add_node("Implementasi", lambda state: agent_node(state, Implementasi_simple_runnable, Implementasi_tool_executor, "Implementasi"))
builder.add_node("Creator", lambda state: creator_agent_node(state, creator_simple_runnable, "Creator"))
builder.add_node("Supervisor", supervisor_node)

# Add edges
for member in members:
    builder.add_edge(member, "Supervisor")

builder.add_conditional_edges(
    "Supervisor",
    lambda state: state.get("next_agent"),
    {"Riset": "Riset", "Implementasi": "Implementasi", "FINISH": END},
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