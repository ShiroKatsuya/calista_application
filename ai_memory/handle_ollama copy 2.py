import os
import json
from datetime import datetime
from uuid import uuid4 # Keep for potential future use, though not primary session ID now
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai.types import (GenerateContentConfig
)



client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = os.getenv("GEMINI_MODEL")

google_search_tool = Tool(
    google_search = GoogleSearch()
)

# Define search config once
search_config = GenerateContentConfig(
    tools=[google_search_tool],
    response_modalities=["TEXT"]
)

# Note: zero_shot_prompt is defined but not used in get_realtime_data.
# It might be intended for a different part of the system or future use.
zero_shot_prompt = """You are a helpful AI assistant with access to Google Search. When using the search tool:
1. Extract the key information from search results
2. Present the information in a clear, organized way
3. Focus on factual, up-to-date details
4. Avoid redundant information
5. No need to quote from sources """


def get_realtime_data(query: str) -> str:
    """
    Fetches real-time data using Google Search via Gemini API.

    Args:
        query: The user's query string.

    Returns:
        A string containing the search results or an error message.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fetching real-time data for query: '{query}' at {current_time}")
    try:
        # No need to truncate query here unless specifically required by the API
        # query = query[:some_limit] # Example if truncation is needed

        response = client.models.generate_content(
            model=model_id,
            contents=query, # Pass the user query directly
            config=search_config # Use the defined config
        )

        # Defensive check for response structure
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            search_results = response.candidates[0].content.parts[0].text
            print("Search results obtained.")
            # Return only the text content of the search results
            print("realtime data :",search_results)
            return f"As of {current_time}, here's the information found regarding '{query}':\n{search_results}"
        else:
            print("Warning: Unexpected response structure from Gemini API.")
            return f"Sorry, I couldn't retrieve information for '{query}' at this time (unexpected API response)."

    except Exception as e:
        print(f"Error fetching real-time data: {str(e)}")
        # Return an error message string
        return f"Sorry, I encountered an error while trying to fetch real-time information for '{query}': {str(e)}"


# --- Configuration ---
TEMPERATURE = 0.7 # Controls creativity (0.0 = deterministic, 1.0 = more creative)
MEMORY_KEY = "history"

MAX_WINDOW_TURNS = 60  # Stores last 60 pairs of human/ai messages (Adjusted from 120000)
MEMORY_DIR = "memory_files" # Directory to store conversation history files

# --- Global Store for Session History (Initialized Once) ---
chat_session_store = {}
_current_session_id = None
_current_memory_file = None

# --- Initialization Block (Runs Once) ---
if 'chat_session_store' not in globals() or not chat_session_store: # Ensure initialization runs
    chat_session_store = {} # Ensure it's initialized

    try:
        # --- Ensure Memory Directory Exists ---
        os.makedirs(MEMORY_DIR, exist_ok=True)
        print(f"Memory directory set to: {MEMORY_DIR}")

        # --- Always Create a New Session ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _current_session_id = f"session_{timestamp}"
        _current_memory_file = os.path.join(MEMORY_DIR, f"{_current_session_id}.json")

        print(f"Starting new session for this run: {_current_session_id}")
        print(f"History for this session will be saved to: {_current_memory_file}")
        print(f"Previous session files in '{MEMORY_DIR}' are preserved but not loaded.")

        # Create an empty memory buffer for the new session
        new_memory = ConversationBufferWindowMemory(
            memory_key=MEMORY_KEY,
            return_messages=True,
            k=MAX_WINDOW_TURNS,
        )
        # Store the new, empty chat history object
        chat_session_store[_current_session_id] = new_memory.chat_memory
        # Note: The actual file is created only when history is first saved.

    except Exception as e:
        print(f"FATAL: Error during memory initialization: {e}")
        # Fallback mechanism if session creation fails
        _current_session_id = f"session_fallback_{uuid4()}" # Use UUID if timestamp fails
        _current_memory_file = None # Indicate saving might fail
        # Initialize empty store to prevent crashes later
        if _current_session_id not in chat_session_store:
             fallback_memory = ConversationBufferWindowMemory(memory_key=MEMORY_KEY, return_messages=True, k=MAX_WINDOW_TURNS)
             chat_session_store[_current_session_id] = fallback_memory.chat_memory
        print(f"Warning: Proceeding with fallback session ID {_current_session_id} and potentially no persistence.")


# --- History Persistence Functions ---
def save_current_conversation_history():
    """Saves the current session's history to its dedicated JSON file."""
    global _current_session_id, _current_memory_file, chat_session_store

    if not _current_session_id or not _current_memory_file:
        print("Error: Cannot save history. Current session ID or file path is not set.")
        return

    if _current_session_id not in chat_session_store:
        print(f"Error: History for session {_current_session_id} not found in store.")
        return

    try:
        chat_memory_history = chat_session_store[_current_session_id]
        messages = []
        # Access messages directly from the BaseChatMessageHistory object
        for message in chat_memory_history.messages:
            # Serialize according to the specified format
            if isinstance(message, HumanMessage):
                messages.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"type": "ai", "content": message.content})
            # Add other types if needed (e.g., SystemMessage, ToolMessage)

        # Ensure directory exists before writing
        os.makedirs(os.path.dirname(_current_memory_file), exist_ok=True)

        with open(_current_memory_file, 'w') as f:
            json.dump(messages, f, indent=2) # Save the list directly
            # print(f"Saved conversation history to {_current_memory_file}") # Reduce console noise
    except Exception as e:
        print(f"Error saving conversation history to {_current_memory_file}: {e}")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves the chat history for the given session ID from the global store.
       Called by RunnableWithMessageHistory. Expects the session_id passed during invoke/stream.
    """
    global chat_session_store
    if session_id not in chat_session_store:
        # This case should ideally only be hit if the _current_session_id somehow
        # differs from the one passed in config, or if the fallback mechanism was triggered.
        print(f"Warning: Session ID '{session_id}' not found in store during get_session_history. Creating new temporary buffer.")
        # Create an empty history defensively.
        temp_memory = ConversationBufferWindowMemory(
            memory_key=MEMORY_KEY,
            return_messages=True,
            k=MAX_WINDOW_TURNS,
        )
        chat_session_store[session_id] = temp_memory.chat_memory
    # Return the BaseChatMessageHistory object for the session
    return chat_session_store[session_id]


def handle_ollama_conversation(user_input: str, model_name: str, voice_func):
    """
    Handles the conversation logic with Ollama, optionally fetching real-time data,
    using a new timestamped session for each run, LLM interaction, streaming, and voice output.

    Args:
        user_input: The raw input string from the user.
        model_name: The name of the Ollama model to use.
        voice_func: The function to call for text-to-speech output.
    """
    global _current_session_id # Access the globally managed current session ID

    # --- Define Keywords for Real-time Data Fetching ---
    realtime_keywords = ["latest", "current", "new", "recent", "upcoming", "2024", "2025", "today", "now", "price", "weather", "news"]

    # --- Define Base Prompt Template ---
    # This prompt guides the LLM on how to behave generally.
    # We will add context (like search results) dynamically if needed.
    prompt_template_str = """You are an AI assistant focused on clear and natural conversation.
Your primary goal is to provide answers that are **concise**, **to-the-point**, and use **simple, everyday language**.
**Avoid technical jargon** and complex sentence structures. Ensure every response is easily understandable.
Crucially, **always link your response directly to the user's *last* message**, referencing it explicitly if helpful, to maintain a smooth conversational flow.
**Avoid repeating your previous questions or statements.** If the user confirms a previous question (e.g., with 'Yes'), acknowledge the confirmation and move the conversation forward based on that understanding. Do not ask the same question again.
Remember the conversation history provided below to maintain context.
If the user's input seems unclear or ambiguous *and you haven't just asked a clarifying question*, **ask a clarifying question** before giving a detailed answer.
If relevant real-time information is provided below, incorporate it naturally into your response to the user's query."""

    # --- Inner Try/Except Block for Ollama Interaction ---
    try:
        # --- 0. Check if Initialization Succeeded ---
        if not _current_session_id:
             raise RuntimeError("Memory system failed to initialize. Cannot proceed.")

        # --- 1. Initialize LLM ---
        if not model_name:
            print("Error: model_name not provided or empty.")
            raise ValueError("model_name is required.")

        print(f"Initializing Ollama model: {model_name}")
        llm = OllamaLLM(
            model=model_name,
            temperature=TEMPERATURE,
        )
        print("LLM Initialized.")

        # --- 2. Determine if Real-time Data is Needed ---
        # Check the actual user_input string passed to the function
        needs_search = any(keyword in user_input.lower() for keyword in realtime_keywords)
        input_for_ollama = user_input # Default input is the user's raw query

        if needs_search:
            print("Keywords detected, attempting to fetch real-time data...")
            search_results_content = get_realtime_data(user_input) # Pass the user's query

            # Prepend the search results (or error message) to the user input for the LLM
            # This gives the LLM context to work with.
            input_for_ollama = f"User query: '{user_input}'\n\nRelevant information found:\n{search_results_content}"
            print("Real-time data fetched (or error occurred), preparing augmented input for LLM.")
        else:
            print("No real-time keywords detected, proceeding with standard input.")


        # --- 3. Define Enhanced Prompt Template ---
        # This template structure allows injecting history and the potentially augmented input.
        conversation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template_str), # General instructions
                MessagesPlaceholder(variable_name=MEMORY_KEY), # Injects history messages
                ("human", "{input}"), # User's input (potentially augmented with search results)
            ]
        )
        print("Enhanced Chat Prompt Template Defined.")

        # --- 4. Create Runnable with History ---
        runnable = conversation_prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,    # Function to load history for the specific session_id
            input_messages_key="input", # Key for user input in the prompt
            history_messages_key=MEMORY_KEY, # Key for history messages in the prompt ("history")
        )
        print("RunnableWithMessageHistory Created with Context Awareness.")

        # --- 5. Interaction Loop (Streaming) ---
        def ollama_stream_generator(stream):
            """Processes and yields Ollama stream chunks for audio output, handling persistence."""
            print("\nAssistant:", end=' ', flush=True) # Console indicator for AI response start
            full_response = ""
            try:
                for chunk in stream:
                    content = ""
                    # Handle different possible chunk structures from Ollama/Langchain stream
                    if isinstance(chunk, str):
                        content = chunk
                    elif hasattr(chunk, 'content'): # AIMessageChunk etc.
                        content = chunk.content
                    elif isinstance(chunk, dict):
                        # Ollama raw stream might have {'message': {'content': '...'}}
                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                        # Older Ollama versions might just have {'response': '...'}
                        elif 'response' in chunk:
                             content = chunk['response']
                        else:
                            # Fallback for unexpected dict structure
                            content = str(chunk)
                    else:
                        print(f"\n[Debug] Unexpected chunk type: {type(chunk)}, {chunk}")
                        content = str(chunk) # Attempt to convert to string

                    # Basic cleaning (can be expanded)
                    cleaned_chunk = content.replace('*', '') # Remove markdown emphasis

                    if cleaned_chunk:
                        full_response += cleaned_chunk
                        print(cleaned_chunk, end='', flush=True)
                        yield cleaned_chunk # Yield for real-time voice output

            except Exception as e:
                print(f"\nError reading from Ollama stream: {e}")
                yield "Maaf, terjadi kesalahan saat menghasilkan respons." # Yield error message
            finally:
                print("\n(End of response stream)")
                # Save history after the full response is generated/streamed for the current session
                save_current_conversation_history() # Call the updated save function

        # --- Execute the chain and stream the response ---
        print(f"Using session ID: {_current_session_id}")
        print(f"Input sent to Ollama chain: {input_for_ollama[:200]}...") # Log truncated input

        # Pass the potentially augmented input to the chain
        response_stream = chain_with_history.stream(
            {"input": input_for_ollama}, # Pass the prepared input under the "input" key
            config={"configurable": {"session_id": _current_session_id}} # Crucial: Pass the correct session ID
        )

        # Pass the generator to the voice function for real-time TTS output
        voice_func(ollama_stream_generator(response_stream)) # Use function parameter

    except Exception as e:
        print(f"\nUnexpected error during Ollama interaction: {str(e)}")
        # Fallback to a simple voice message if anything above fails
        try:
            # Attempt to use a predefined simple voice function for error feedback
            # Keep import local to avoid circular dependencies or issues if output_audio is unavailable
            from output_audio.voice import voice as simple_voice
            simple_voice("Maaf, saya mengalami kendala saat memproses permintaan Anda.")
        except ImportError:
            print("Fallback voice function ('simple_voice') not available.")
        except Exception as e_voice:
            print(f"Error calling fallback voice function: {e_voice}")