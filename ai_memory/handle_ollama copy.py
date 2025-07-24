import os
import json
from datetime import datetime
from uuid import uuid4 # Keep for potential future use, though not primary session ID now
from langchain_ollama.llms import OllamaLLM
from langchain.memory.vectorstore_token_buffer_memory import ConversationVectorStoreTokenBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from google.genai.types import (GenerateContentConfig
)
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import google.generativeai as genai_embed
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time # Add time import for metrics
# embedder = HuggingFaceInstructEmbeddings(
#     model_name="hkunlp/instructor-large",  # Specify the model name
#     model_kwargs={'device': 'gpu'},       # Specify model arguments (e.g., device)
#     encode_kwargs={'normalize_embeddings': True} # Specify encoding arguments
#     # The original query_instruction is removed as the instructions provide a complete example
#     # instantiation which doesn't include it, implying it should be replaced or defaulted.
# )
query_instruction=(
        "You are a retrieval_oriented embedding model. "
        "Transform the user's natural_language query into a compact, keyword_rich "
        "representation optimized for semantic search."
    )

# embedder = genai_embed.embed_content(
#                 model="models/embedding-001",
#                 content=query_instruction,
#                 task_type="retrieval_query"
#             )
# print("embedder",embedder)

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

MAX_WINDOW_TURNS = 300  # Stores last 60 pairs of human/ai messages (Adjusted from 120000)
MEMORY_DIR = "memory_files" # Directory to store conversation history files

# --- Define a default LLM for fallback ---
# You might want to use an environment variable or choose a different default model
DEFAULT_OLLAMA_MODEL_NAME = os.getenv("MODEL_NAME") # Default model name
try:
    default_llm_for_memory = OllamaLLM(model=DEFAULT_OLLAMA_MODEL_NAME, temperature=TEMPERATURE)
    print(f"Default LLM for memory fallback initialized: {DEFAULT_OLLAMA_MODEL_NAME}")
except Exception as e:
    print(f"CRITICAL WARNING: Failed to initialize default LLM ('{DEFAULT_OLLAMA_MODEL_NAME}'). Memory fallback might fail. Error: {e}")
    default_llm_for_memory = None # Set to None if initialization fails

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
        new_memory = ConversationVectorStoreTokenBufferMemory(
            llm=default_llm_for_memory, 
            return_messages=True,
            retriever=retriever,
            max_token_limit=MAX_WINDOW_TURNS, 
        )
        # Store the new, full memory object
        chat_session_store[_current_session_id] = new_memory
        # Note: The actual file is created only when history is first saved.

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
                chat_session_store[_current_session_id] = fallback_memory # Store full object
            else:
                print(f"Error: Cannot create fallback ConversationVectorStoreTokenBufferMemory for session {_current_session_id} because default LLM failed to initialize.")
                chat_session_store[_current_session_id] = None 
        print(f"Warning: Proceeding with fallback session ID {_current_session_id} and potentially no persistence.")


def get_full_memory_object(session_id: str) -> ConversationVectorStoreTokenBufferMemory | None:
    """Retrieves the full ConversationVectorStoreTokenBufferMemory object for the session."""
    global chat_session_store
    memory_object = chat_session_store.get(session_id)
    if isinstance(memory_object, ConversationVectorStoreTokenBufferMemory):
        return memory_object
    elif memory_object is None:
        print(f"Warning: No memory object found for session_id '{session_id}'. It might be a fallback scenario where memory init failed.")
    else:
        print(f"Warning: Unexpected object type in chat_session_store for session_id '{session_id}': {type(memory_object)}")
    return None

# --- History Persistence Functions ---
def save_current_conversation_history():
    """Saves the current session's history to its dedicated JSON file."""
    global _current_session_id, _current_memory_file, chat_session_store

    if not _current_session_id or not _current_memory_file:
        print("Error: Cannot save history. Current session ID or file path is not set.")
        return

    memory_object = chat_session_store.get(_current_session_id)

    if not memory_object:
        print(f"Error: Memory object for session {_current_session_id} not found in store.")
        return

    if not isinstance(memory_object, ConversationVectorStoreTokenBufferMemory):
        print(f"Error: Object in store for session {_current_session_id} is not a ConversationVectorStoreTokenBufferMemory instance. Type: {type(memory_object)}")
        return

    try:
        # Access .chat_memory to get the BaseChatMessageHistory object, then .messages
        chat_history = memory_object.chat_memory 
        messages = []
        # Access messages directly from the BaseChatMessageHistory object
        for message in chat_history.messages: # Corrected: Access .messages on chat_history
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
    except AttributeError as ae:
        print(f"Error accessing messages for session {_current_session_id}: {ae}. Memory object type: {type(memory_object)}, chat_memory type: {type(memory_object.chat_memory if hasattr(memory_object, 'chat_memory') else None)}")
    except Exception as e:
        print(f"Error saving conversation history to {_current_memory_file}: {e}")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves the chat history for the given session ID from the global store.
       Called by RunnableWithMessageHistory. Expects the session_id passed during invoke/stream.
    """
    global chat_session_store, default_llm_for_memory, retriever, MAX_WINDOW_TURNS
    
    memory_object = chat_session_store.get(session_id)

    if not isinstance(memory_object, ConversationVectorStoreTokenBufferMemory):
        print(f"Warning: Session ID '{session_id}' not found or invalid type in store during get_session_history. Type: {type(memory_object)}. Creating new temporary memory object.")
        if default_llm_for_memory is None:
            print(f"CRITICAL ERROR: default_llm_for_memory is None. Cannot create new ConversationVectorStoreTokenBufferMemory for session {session_id}.")
            # Fallback to a basic history, though this is not ideal for ConversationVectorStoreTokenBufferMemory's functionality
            from langchain.memory import ChatMessageHistory as BasicChatMessageHistory
            temp_basic_history = BasicChatMessageHistory()
            # Storing this basic history might lead to type errors if full memory object features are assumed later for this session_id
            # A proper fix would be to ensure default_llm_for_memory is always available or handle this state more robustly.
            return temp_basic_history 

        temp_memory_obj = ConversationVectorStoreTokenBufferMemory(
            llm=default_llm_for_memory,
            return_messages=True,
            retriever=retriever,
            max_token_limit=MAX_WINDOW_TURNS
        )
        chat_session_store[session_id] = temp_memory_obj
        memory_object = temp_memory_obj

    if not hasattr(memory_object, 'chat_memory'):
        # This case should ideally not be hit if chat_session_store always holds ConversationVectorStoreTokenBufferMemory instances or None
        print(f"ERROR: Object for session {session_id} in store (type: {type(memory_object)}) does not have 'chat_memory' attribute.")
        # Fallback to prevent immediate crash, but indicates a deeper issue.
        from langchain.memory import ChatMessageHistory as BasicChatMessageHistory
        return BasicChatMessageHistory()

    return memory_object.chat_memory


def handle_ollama_conversation(user_input: str, model_name: str, voice_func):

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
        # --- 0. Check if Initialization Succeeded ---
        if not _current_session_id:
             raise RuntimeError("Memory system failed to initialize. Cannot proceed.")

        # --- Metrics Collection Setup ---
        initial_buffer_tokens = 0
        # initial_vector_store_docs = 0 # Removed
        simulated_retrieval_time_ms = 0
        simulated_avg_relevance_score_percent = 0
        simulated_num_retrieved_docs = 0

        cvs_memory = get_full_memory_object(_current_session_id)

        if cvs_memory and hasattr(cvs_memory, 'llm') and cvs_memory.llm:
            try:
                # Initial Buffer Tokens
                buffer_messages_content = [msg.content for msg in cvs_memory.chat_memory.messages]
                if buffer_messages_content:
                    initial_buffer_tokens = sum(cvs_memory.llm.get_num_tokens(content) for content in buffer_messages_content)

                # Initial Vector Store Documents - REMOVED
                # if hasattr(cvs_memory, 'retriever') and hasattr(cvs_memory.retriever, 'vectorstore') and hasattr(cvs_memory.retriever.vectorstore, '_collection'):
                #     initial_vector_store_docs = cvs_memory.retriever.vectorstore._collection.count()
                # else:
                #     print("Warning: Could not get initial vector store doc count directly (retriever, vectorstore, or _collection not found).")
            except Exception as e:
                print(f"Error getting initial memory stats: {e}")

            # Simulate Retrieval for Metrics
            current_buffer_messages = cvs_memory.chat_memory.messages
            if current_buffer_messages:
                query_for_retriever = "\n".join([msg.content for msg in current_buffer_messages]).strip()
                if query_for_retriever:
                    try:
                        print(f"Simulating retrieval for metrics with query from buffer: '{query_for_retriever[:100]}...'")
                        
                        metrics_calc_start_time = time.perf_counter() 
                        
                        if not (hasattr(cvs_memory, 'retriever') and hasattr(cvs_memory.retriever, 'vectorstore')):
                            raise AttributeError("Retriever or its vectorstore not found on ConversationVectorStoreTokenBufferMemory object.")

                        search_kwargs = cvs_memory.retriever.search_kwargs if hasattr(cvs_memory.retriever, 'search_kwargs') else {}
                        k_val = search_kwargs.get('k', 5)
                        score_thresh = search_kwargs.get('score_threshold')

                        retrieved_docs_with_distance_scores = cvs_memory.retriever.vectorstore.similarity_search_with_score(
                            query_for_retriever,
                            k=k_val
                        )
                        base_simulated_retrieval_time_ms = (time.perf_counter() - metrics_calc_start_time) * 1000
                        
                        processed_docs_for_metrics = []
                        adjusted_score_threshold_for_distance = None

                        if score_thresh is not None and hasattr(cvs_memory.retriever, 'vectorstore') and \
                           hasattr(cvs_memory.retriever.vectorstore, 'collection_metadata') and \
                           cvs_memory.retriever.vectorstore.collection_metadata.get("hnsw:space") == "cosine":
                            adjusted_score_threshold_for_distance = 1.0 - score_thresh
                            
                        for doc, distance_val_score in retrieved_docs_with_distance_scores:
                            doc_score_to_use = distance_val_score 
                            passes_filter = False

                            if adjusted_score_threshold_for_distance is not None: 
                                if distance_val_score <= adjusted_score_threshold_for_distance:
                                    passes_filter = True
                                    doc_score_to_use = 1.0 - distance_val_score 
                            else:
                                passes_filter = True 
                                doc_score_to_use = distance_val_score

                            if passes_filter:
                                processed_docs_for_metrics.append((doc, doc_score_to_use))
                        
                        tokenization_time_for_retrieved_docs_ms = 0.0
                        total_tokens_in_retrieved_docs = 0

                        if processed_docs_for_metrics:
                            for doc, _ in processed_docs_for_metrics: # Original doc_score_value not used here
                                doc_content = doc.page_content
                                num_tokens = 0
                                doc_tokenization_start_indiv_time = time.perf_counter()
                                try:
                                    if cvs_memory.llm: # Check LLM is available
                                        num_tokens = cvs_memory.llm.get_num_tokens(doc_content)
                                except Exception as e_tokens:
                                    print(f"Warning: Could not get token count for a retrieved document: {e_tokens}")
                                tokenization_time_for_retrieved_docs_ms += (time.perf_counter() - doc_tokenization_start_indiv_time) * 1000
                                
                                if num_tokens > 0:
                                    total_tokens_in_retrieved_docs += num_tokens
                        
                        simulated_num_retrieved_docs = len(processed_docs_for_metrics)

                        # --- NEW: Generate simulated response and score it ---
                        simulated_response_eval_time_ms = 0.0
                        actual_simulated_avg_relevance_score_value = 0.0 

                        sim_resp_eval_start_time = time.perf_counter()
                        # Use top 3 (or fewer if less available) from already filtered/scored docs for context
                        sim_context_docs_content = [doc.page_content for doc, score_val in processed_docs_for_metrics[:3]]
                        
                        simulated_llm_response_text = ""
                        # Only attempt generation if LLM is available and there's context from processed docs
                        if cvs_memory.llm and sim_context_docs_content: 
                            sim_context_text = "\n\n".join(sim_context_docs_content)
                            sim_prompt = f"Based on the following context, answer the query.\n\nContext:\n{sim_context_text}\n\nQuery:\n'''{query_for_retriever}'''\n\nAnswer:"
                            try:
                                simulated_llm_response_text = cvs_memory.llm.invoke(sim_prompt)
                            except Exception as e_sim_llm:
                                print(f"Error generating simulated LLM response for metrics: {e_sim_llm}")

                        if simulated_llm_response_text: # Only score if response was generated
                            try:
                                gemini_relevance_prompt = f"""Evaluate the relevance of the 'Generated Text' to the 'Query'.
Respond with a single floating-point number between 0.0 (not relevant) and 1.0 (highly relevant). Only provide the number.

Query:
'''{query_for_retriever}'''

Generated Text:
'''{simulated_llm_response_text}'''"""

                                relevance_response = client.models.generate_content(
                                    model=model_id, 
                                    contents=gemini_relevance_prompt,
                                    config=GenerateContentConfig(response_modalities=["TEXT"])
                                )
                                if relevance_response.candidates and relevance_response.candidates[0].content and relevance_response.candidates[0].content.parts:
                                    relevance_score_text = relevance_response.candidates[0].content.parts[0].text.strip()
                                    try:
                                        actual_simulated_avg_relevance_score_value = float(relevance_score_text)
                                    except ValueError:
                                        print(f"Could not parse relevance score from Gemini: '{relevance_score_text}'")
                                else:
                                    print("Warning: No valid response from Gemini for relevance scoring of simulated response.")
                            except Exception as e_gemini_score:
                                print(f"Error during Gemini relevance scoring for simulated response: {e_gemini_score}")
                        
                        simulated_response_eval_time_ms = (time.perf_counter() - sim_resp_eval_start_time) * 1000
                        simulated_avg_relevance_score_percent = actual_simulated_avg_relevance_score_value * 100.0
                        # --- END NEW ---
                        
                        simulated_retrieval_time_ms = base_simulated_retrieval_time_ms + tokenization_time_for_retrieved_docs_ms + simulated_response_eval_time_ms

                        print(f"Simulated metrics: {simulated_num_retrieved_docs} docs processed (total tokens: {total_tokens_in_retrieved_docs}), "
                              f"Time: {simulated_retrieval_time_ms:.2f}ms (vec_search: {base_simulated_retrieval_time_ms:.2f}ms, "
                              f"doc_tokenization: {tokenization_time_for_retrieved_docs_ms:.2f}ms, "
                              f"sim_resp_eval: {simulated_response_eval_time_ms:.2f}ms), "
                              f"Simulated Response Relevance to Query: {simulated_avg_relevance_score_percent:.2f}%")

                    except Exception as e:
                        print(f"Error during simulated metrics calculation: {e}")
                else:
                     print("Simulated retrieval: No query content from buffer, skipping.")
            else:
                print("Simulated retrieval: Buffer is empty, skipping.")
        elif not cvs_memory:
             print("Warning: Full memory object not available for metrics pre-calculation.")


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

        # --- 2. Pre-check if Real-time Data is Needed ---
        needs_search = False
        search_results_content = "No real-time data fetched as it wasn't deemed necessary for this query." # Default
        try:
            pre_check_prompt = f'''Analyze the following user query. Determine if fetching up-to-date, real-time information (like current events, weather, prices, future dates, recent developments) is likely necessary to provide an accurate and relevant answer. Respond with ONLY "SEARCH_NEEDED" if real-time data is likely required, or "NO_SEARCH_NEEDED" otherwise. Do not provide any explanation.

User Query:
{user_input}'''
            # Use the Gemini client for the pre-check
            pre_check_response = client.models.generate_content(
                model=model_id, # Use the configured Gemini model
                contents=pre_check_prompt,
                # No search tool needed for this classification task
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
            # Proceed without search if pre-check fails

        # --- 3. Get Contextual Data (Potentially Real-time) ---
        if needs_search:
            print("Pre-check indicated need for real-time data. Attempting to fetch...")
            # Pass the original user_input to the function that performs the search
            search_results_content = get_realtime_data(user_input)
            input_for_ollama = f"Contextual Information:\n{search_results_content}\n\nUser's Query:\n{user_input}"
            print("Real-time data fetched, preparing augmented input for LLM.")
        else:
            print("No real-time data needed based on pre-check or pre-check failed.")
            input_for_ollama = user_input # Use original input if no search was done

        # --- 4. Define Enhanced Prompt Template ---
        # This template structure allows injecting history and the potentially augmented input.
        conversation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template_str), # General instructions
                MessagesPlaceholder(variable_name=MEMORY_KEY), # Injects history messages
                ("human", "{input}"), # User's input (augmented with context)
            ]
        )
        print("Enhanced Chat Prompt Template Defined.")

        # --- 5. Create Runnable with History ---
        runnable = conversation_prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,    # Function to load history for the specific session_id
            input_messages_key="input", # Key for user input in the prompt
            history_messages_key=MEMORY_KEY, # Key for history messages in the prompt ("history")
        )
        print("RunnableWithMessageHistory Created with Context Awareness.")

        # --- 6. Interaction Loop (Streaming) ---
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
                        # print("cleaned_chunk",cleaned_chunk)
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

        # --- Metrics Collection (Post Response) ---
        final_buffer_tokens = 0
        # final_vector_store_docs = 0 # Removed
        cvs_memory_after = get_full_memory_object(_current_session_id)

        if cvs_memory_after and hasattr(cvs_memory_after, 'llm') and cvs_memory_after.llm:
            try:
                buffer_messages_content_after = [msg.content for msg in cvs_memory_after.chat_memory.messages]
                if buffer_messages_content_after:
                    final_buffer_tokens = sum(cvs_memory_after.llm.get_num_tokens(content) for content in buffer_messages_content_after)
                
                # Final Vector Store Documents - REMOVED
                # if hasattr(cvs_memory_after, 'retriever') and hasattr(cvs_memory_after.retriever, 'vectorstore') and hasattr(cvs_memory_after.retriever.vectorstore, '_collection'):
                #     final_vector_store_docs = cvs_memory_after.retriever.vectorstore._collection.count()
                # else:
                #     print("Warning: Could not get final vector store doc count directly (retriever, vectorstore, or _collection not found).")
            except Exception as e:
                print(f"Error getting final memory stats: {e}")
        elif not cvs_memory_after:
            print("Warning: Full memory object not available for metrics post-calculation.")

        print("\n--- ConversationVectorStoreTokenBufferMemory Metrics ---")
        print(f"User Input: \"{user_input[:100]}...\"")
        print(f"(1) Tokens in Buffer (Initial): {initial_buffer_tokens}")
        # Clarification: Reporting number of documents as a proxy for tokens in VectorStore - REMOVED
        # print(f"(1) Documents in VectorStore (Initial): {initial_vector_store_docs} (Note: This is document count, not token count)")
        print(f"(2) Tokens in Buffer (Final): {final_buffer_tokens}")
        # print(f"(2) Documents in VectorStore (Final): {final_vector_store_docs} (Note: This is document count, not token count)") # REMOVED
        print(f"(3) Simulated Processing Time (ms): {simulated_retrieval_time_ms:.2f}") # Renamed for clarity
        print(f"    - Documents Processed (Simulated): {simulated_num_retrieved_docs}")
        print(f"(4) Simulated Response Relevance to Query (%): {simulated_avg_relevance_score_percent:.2f}") # Renamed for clarity
        print("---------------------------------------------------\n")

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