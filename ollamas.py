import os
import json
import re
from datetime import datetime
from uuid import uuid4
from langchain.memory.vectorstore_token_buffer_memory import ConversationVectorStoreTokenBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from voice_natural import stream_voice
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



query_instruction=(
        "You are a retrieval_oriented embedding model. "
        "Transform the user's natural_language query into a compact, keyword_rich "
        "representation optimized for semantic search."
    )

embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
print("embedder",embedder)

chroma = Chroma(collection_name="demo",
                embedding_function=embedder,
                collection_metadata={"hnsw:space": "cosine"},
                )

retriever = chroma.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            'k': 5,
            'score_threshold': 0.75,
        },
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


    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Fetching real-time data for query: '{query}' at {current_time}")
    try:
        # No need to truncate query here unless specifically required by the API
        # query = query[:some_limit] # Example if truncation is needed

        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=search_config
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
TEMPERATURE = 0.7
MEMORY_KEY = "history"

MAX_WINDOW_TURNS = 120000
MEMORY_DIR = "memory_files"

model_realtime = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
)

# --- Define a default LLM for fallback ---
# DEFAULT_OLLAMA_MODEL_NAME = os.getenv("MODEL_NAME_GENERAL_MODE")
try:
    default_llm_for_memory = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=TEMPERATURE)
    print(f"Default LLM for memory fallback initialized: {default_llm_for_memory}")
except Exception as e:
    print(f"CRITICAL WARNING: Failed to initialize default LLM ('{default_llm_for_memory}'). Memory fallback might fail. Error: {e}")
    default_llm_for_memory = None

# # --- Define a default LLM for fallback ---
# DEFAULT_OLLAMA_MODEL_NAME = os.getenv("MODEL_NAME_GENERAL_MODE")
# try:
#     default_llm_for_memory = OllamaLLM(model=DEFAULT_OLLAMA_MODEL_NAME, temperature=TEMPERATURE)
#     print(f"Default LLM for memory fallback initialized: {DEFAULT_OLLAMA_MODEL_NAME}")
# except Exception as e:
#     print(f"CRITICAL WARNING: Failed to initialize default LLM ('{DEFAULT_OLLAMA_MODEL_NAME}'). Memory fallback might fail. Error: {e}")
#     default_llm_for_memory = None

# --- Global Store for Session History (Initialized Once) ---
chat_session_store = {}
_current_session_id = None
_current_memory_file = None

# --- Initialization Block (Runs Once) ---
if 'chat_session_store' not in globals() or not chat_session_store:
    chat_session_store = {}

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
        new_memory = ConversationVectorStoreTokenBufferMemory(
            llm=default_llm_for_memory,
            return_messages=True,
            retriever=retriever,
            max_token_limit=MAX_WINDOW_TURNS,
        )
        # Store the new, empty chat history object
        chat_session_store[_current_session_id] = new_memory.chat_memory

    except Exception as e:
        print(f"FATAL: Error during memory initialization: {e}")
        _current_session_id = f"session_fallback_{uuid4()}"
        _current_memory_file = None
        if _current_session_id not in chat_session_store:
            if default_llm_for_memory:
                fallback_memory = ConversationVectorStoreTokenBufferMemory(
                    llm=default_llm_for_memory,
                    return_messages=True,
                    retriever=retriever,
                    max_token_limit=MAX_WINDOW_TURNS
                )
                chat_session_store[_current_session_id] = fallback_memory.chat_memory
            else:
                print(f"Error: Cannot create fallback ConversationVectorStoreTokenBufferMemory for session {_current_session_id} because default LLM failed to initialize.")
                chat_session_store[_current_session_id] = None
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
        for message in chat_memory_history.messages:
            if isinstance(message, HumanMessage):
                messages.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"type": "ai", "content": message.content})

        os.makedirs(os.path.dirname(_current_memory_file), exist_ok=True)

        with open(_current_memory_file, 'w') as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        print(f"Error saving conversation history to {_current_memory_file}: {e}")

def create_new_session():
    """Creates a new chat session and returns the session ID."""
    global _current_session_id, _current_memory_file, chat_session_store
    
    try:
        # Save current session before creating new one
        if _current_session_id and _current_session_id in chat_session_store:
            save_current_conversation_history()
        
        # Create new session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_session_id = f"session_{timestamp}"
        new_memory_file = os.path.join(MEMORY_DIR, f"{new_session_id}.json")
        
        # Create new memory buffer
        new_memory = ConversationVectorStoreTokenBufferMemory(
            llm=default_llm_for_memory,
            return_messages=True,
            retriever=retriever,
            max_token_limit=MAX_WINDOW_TURNS,
        )
        
        # Update global variables
        _current_session_id = new_session_id
        _current_memory_file = new_memory_file
        chat_session_store[_current_session_id] = new_memory.chat_memory
        
        print(f"Created new session: {_current_session_id}")
        print(f"New session file: {_current_memory_file}")
        
        return {
            "session_id": _current_session_id,
            "timestamp": timestamp,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error creating new session: {e}")
        return {
            "session_id": None,
            "timestamp": None,
            "status": "error",
            "message": str(e)
        }

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves the chat history for the given session ID from the global store.
       Called by RunnableWithMessageHistory. Expects the session_id passed during invoke/stream.
    """
    global chat_session_store
    if session_id not in chat_session_store:
        print(f"Warning: Session ID '{session_id}' not found in store during get_session_history. Creating new temporary buffer.")
        temp_memory = ConversationVectorStoreTokenBufferMemory(
            return_messages=True,
            retriever=retriever,
            max_token_limit=MAX_WINDOW_TURNS
        )
        chat_session_store[session_id] = temp_memory.chat_memory
    return chat_session_store[session_id]


def handle_ollama_conversation(user_input: str, model_name: str):
    global _current_session_id
    prompt_template_str = """You are an AI assistant focused on clear and natural conversation.
Your primary goal is to provide answers that are **concise**, **to-the-point**, and use **simple, everyday language**.
**Avoid technical jargon** and complex sentence structures. Ensure every response is easily understandable.
Crucially, **always link your response directly to the user's *last* message**, referencing it explicitly if helpful, to maintain a smooth conversational flow.
**Avoid repeating your previous questions or statements.** If the user confirms a previous question (e.g., with 'Yes'), acknowledge the confirmation and move the conversation forward based on that understanding. Do not ask the same question again.
Remember the conversation history provided below to maintain context.
If the user's input seems unclear or ambiguous *and you haven't just asked a clarifying question*, **ask a clarifying question** before giving a detailed answer.
Contextual information, potentially including real-time data relevant to the user's query, may be provided before the user's message. **Integrate this contextual information naturally into your response** to provide the most accurate and up-to-date answer possible. If the context doesn't seem relevant to the user's *specific* last question, prioritize answering the question directly based on your knowledge and the conversation history.
"""
    try:
        if not _current_session_id:
             raise RuntimeError("Memory system failed to initialize. Cannot proceed.")

        if not model_name:
            print("Error: model_name not provided or empty.")
            raise ValueError("model_name is required.")

        print(f"Initializing Ollama model: {model_name}")
        # llm = OllamaLLM(
        #     model=model_name,
        #     temperature=TEMPERATURE,
        # )
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",

        )
        print("LLM Initialized.")

        needs_search = False
        search_results_content = "No real-time data fetched as it wasn't deemed necessary for this query."
        try:
            pre_check_prompt = f'''Analyze the following user query. Determine if fetching up-to-date, real-time information (like current events, weather, prices, future dates, recent, 2025, 2026 developments) is likely necessary to provide an accurate and relevant answer. Respond with ONLY "SEARCH_NEEDED" if real-time data is likely required, or "NO_SEARCH_NEEDED" otherwise. Do not provide any explanation.

User Query:
{user_input}'''
            pre_check_response = client.models.generate_content(
                model=model_id,
                contents=pre_check_prompt,
                config=GenerateContentConfig(response_modalities=["TEXT"])
            )

            if pre_check_response.candidates and pre_check_response.candidates[0].content and pre_check_response.candidates[0].content.parts:
                decision = pre_check_response.candidates[0].content.parts[0].text.strip()
                print(f"Pre-check decision: {decision}")
                if decision == "SEARCH_NEEDED":
                    needs_search = True
            else:
                print("Warning: Could not get a clear decision from pre-check model.")

        except Exception as e:
            print(f"Error during real-time data pre-check: {e}")

        if needs_search:
            print("Pre-check indicated need for real-time data. Attempting to fetch...")
            search_results_content = get_realtime_data(user_input)
            input_for_ollama = f"Contextual Information:\n{search_results_content}\n\nUser's Query:\n{user_input}"
            print("Real-time data fetched, preparing augmented input for LLM.")
        else:
            print("No real-time data needed based on pre-check or pre-check failed.")
            input_for_ollama = user_input

        conversation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template_str),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("human", "{input}"),
            ]
        )
        print("Enhanced Chat Prompt Template Defined.")

        runnable = conversation_prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key=MEMORY_KEY,
        )
        print("RunnableWithMessageHistory Created with Context Awareness.")

        # --- 6. Interaction Loop (Streaming) --- Stream LLM response and process audio immediately
        print(f"Using session ID: {_current_session_id}")
        print(f"Input sent to Ollama chain: {input_for_ollama[:200]}...")

        # Stream the LLM response and process audio chunks immediately with optimizations
        current_text_buffer = ""
        sentence_end_pattern = re.compile(r'[.!?]\s*')
        pending_sentences = []  # Queue for sentences to process
        processing_sentences = set()  # Track sentences being processed
        
        # Use stream() instead of invoke() to get streaming response
        for chunk in chain_with_history.stream(
            {"input": input_for_ollama},
            config={"configurable": {"session_id": _current_session_id}}
        ):
            # Extract text content from chunk
            chunk_text = ""
            if isinstance(chunk, str):
                chunk_text = chunk
            elif hasattr(chunk, 'content'):
                chunk_text = chunk.content
            elif isinstance(chunk, dict):
                if 'message' in chunk and 'content' in chunk['message']:
                    chunk_text = chunk['message']['content']
                elif 'response' in chunk:
                    chunk_text = chunk['response']
                else:
                    chunk_text = str(chunk)
            else:
                chunk_text = str(chunk)
            
            if chunk_text:
                current_text_buffer += chunk_text
                
                # Check if we have complete sentences to process
                sentences = sentence_end_pattern.split(current_text_buffer)
                if len(sentences) > 1:  # We have at least one complete sentence
                    # Process all complete sentences except the last one (which might be incomplete)
                    for i in range(len(sentences) - 1):
                        sentence = sentences[i].strip()
                        if sentence:
                            # Clean the sentence
                            cleaned_sentence = sentence.replace('*', '').strip()
                            if cleaned_sentence and cleaned_sentence not in processing_sentences:
                                # Add to processing queue
                                pending_sentences.append(cleaned_sentence)
                                processing_sentences.add(cleaned_sentence)
                    
                    # Keep the last (potentially incomplete) sentence in buffer
                    current_text_buffer = sentences[-1]
                
                # Process pending sentences immediately (non-blocking)
                while pending_sentences:
                    sentence_to_process = pending_sentences.pop(0)
                    try:
                        # Stream audio for this sentence immediately
                        for audio_base64, subtitle in stream_voice(sentence_to_process):
                            yield audio_base64, subtitle
                    except Exception as e:
                        print(f"Error processing sentence: {e}")
                        yield "", f"Error processing: {sentence_to_process[:50]}..."
                    finally:
                        processing_sentences.discard(sentence_to_process)
        
        # Process any remaining text in buffer
        if current_text_buffer.strip():
            cleaned_remaining = current_text_buffer.replace('*', '').strip()
            if cleaned_remaining:
                try:
                    for audio_base64, subtitle in stream_voice(cleaned_remaining):
                        yield audio_base64, subtitle
                except Exception as e:
                    print(f"Error processing remaining text: {e}")
                    yield "", f"Error processing remaining text"

    except Exception as e:
        print(f"\nUnexpected error during Ollama interaction: {str(e)}")
        yield "", f"Error: {str(e)}"
    finally:
        save_current_conversation_history()
