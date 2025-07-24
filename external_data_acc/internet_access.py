# import google.generativeai as genai
from langchain_community.document_loaders import AsyncChromiumLoader
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSerperAPIWrapper
from functools import lru_cache
import re
import os
import concurrent.futures
import torch
from input_audio.recording import process_audio, pause_audio_processing, resume_audio_processing, record_audio, process_audio
from output_audio.voice import voice
from dataclasses import dataclass
import time
import random
import json
from external_data_acc.open_website import embed_app
from google import genai
import ollama
from dotenv import load_dotenv
import requests
from retrying import retry
import hashlib

load_dotenv()
import logging


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = os.getenv("GEMINI_MODEL")

ollama_model = os.getenv("MODEL_NAME_INTERNET_TOOLS")

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


@dataclass
class Document:
    page_content: str

def simulate_network_delay():
    """Simulate network latency"""
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)
    return delay
def process_url(args):
    """Helper function to process URLs in parallel"""
    url, use_simulation = args
    return get_and_transform_page(url, use_simulation)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    handlers=[logging.FileHandler("scraping.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# --- Define retry strategy for network errors ---
def retry_if_network_error(exception):
    """Return True if we should retry, False otherwise"""
    return isinstance(exception, (requests.exceptions.Timeout, 
                                   requests.exceptions.ConnectionError,
                                   requests.exceptions.HTTPError)) # Retry on 5xx errors potentially

@retry(retry_on_exception=retry_if_network_error, stop_max_attempt_number=3, wait_fixed=2000)
def fetch_with_requests(url):
    """Fetch URL using requests with retry logic."""
    logger.info(f"Attempting to fetch {url} with requests")
    headers = {'User-Agent': 'Mozilla/5.0 ...'} # Add a realistic User-Agent
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    logger.info(f"Successfully fetched {url} with requests")
    return response.text

def fetch_with_browser(url):
    """Fetch URL using browser automation."""
    logger.info(f"Falling back to browser fetch for {url}")
    loader = AsyncChromiumLoader([url]) 
    html_doc = loader.load()
    if not html_doc:
        logger.error(f"Browser fetch failed for {url}")
        raise ConnectionError(f"Browser loader failed to load {url}")
    logger.info(f"Successfully fetched {url} with browser")
    return html_doc[0].page_content if isinstance(html_doc, list) and html_doc else str(html_doc)


def _get_and_transform_page_uncached(url, use_simulation=False, already_validated=False):
    """Helper to retrieve and transform a page without caching, with optimizations."""
    start_time = time.time()
    page_content_raw = None
    fetch_method = "requests"
    error_message = None
    status = "pending"
    
    # --- Interactive Prompt (Keep existing logic) ---
    if use_simulation and not already_validated:
        voice("Do you want to continue? Please say 'Yes' for an explanation or 'No' to skip.")
        resume_audio_processing()
        audio_processor = process_audio()
        translate = next(audio_processor)
        accepted_responses = {"yes", "sure", "ok", "okay", "yup", "yep"}
        if not any(resp in translate.lower() for resp in accepted_responses):
            voice("Enjoy For Read Article")
            return Document(page_content="")  # Aborts processing if not confirmed.
        print("Validated response received. Continuing...")

    # --- Fetching ---
    try:
        # 1. Try with requests
        page_content_raw = fetch_with_requests(url)
        # Basic check if content seems too small (might indicate JS needed)
        if len(page_content_raw) < 500: # Adjust threshold as needed
             logger.warning(f"Short content from requests for {url}. Trying browser.")
             raise ValueError("Content too short, likely needs browser.")

    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Requests fetch failed for {url}: {e}. Trying browser.")
        fetch_method = "browser"
        try:
            # 2. Fallback to browser (AsyncChromiumLoader)
            if use_simulation:
                 delay = simulate_network_delay()
                 try:
                     embed_app([url]) # Keep existing simulation logic
                     logger.info(f"\nAccessing: {url} | Page load took {delay:.2f} seconds (simulation)")
                 except Exception as emb_e:
                     logger.error(f"Embed error: {emb_e}")
            
            page_content_raw = fetch_with_browser(url)
            
        except Exception as browser_e:
            logger.error(f"Browser fetch failed for {url}: {browser_e}")
            error_message = str(browser_e)
            status = "fetch_error"
            page_content_raw = None # Ensure it's None if fetch fails

    # --- Parsing & Extraction (if fetch succeeded) ---
    extracted_data = {
        'headline': None,
        'date': None,
        'author': None,
        'summary': None,
        'full_text': None,
        'content_hash': None,
    }

    if page_content_raw:
        parse_start_time = time.time()
        try:
            if use_simulation and fetch_method == "browser": # Simulate parsing delay only if browser was used
                 print("Extracting content...")
                 delay = simulate_network_delay()
                 print(f"Content extraction took {delay:.2f} seconds (simulation)")

            soup = BeautifulSoup(page_content_raw, 'html.parser')

            # Remove noisy elements (navigation, headers, footers, scripts, styles, ads, comments)
            for selector in ['nav', 'footer', 'aside', 'header', 'form', 'script', 'style',
                             '[class*="ad"]', '[id*="ad"]', '[class*="comment"]', '[id*="comment"]',
                             '.sidebar', '.ads', '.modal', '.popup']:
                for element in soup.find_all(selector):
                    element.decompose()

            # 2. Information Filtering & Summarization
            # Target specific elements - these are examples, adjust selectors based on common patterns
            # Expanded list of potential main content selectors
            content_selectors = [
                'article', 'main', '[role="main"]', '[itemprop="articleBody"]',
                '.story-content', '.article-content', '.post-content', '.entry-content',
                '.main-content', '.content-body', '.richtext', '.body-content', '.page-content',
                '.td-post-content', '.post', '.single-post-content', '.news-body', '.item-page',
                '.blog-post', '.post-area', '.article-body', '.g-article', '.s-content',
                '#content', '#main-content', '#article', '#post', '#story', '#primary',
                '#page-content', '#mainbar', '#article-content', '#single-post-content',
                '#news-body', '#item-page', '#blog-post', '#post-area', '#article-body',
                '#g-article', '#s-content'
            ]
            
            main_content_tag = None
            best_content_score = 0

            for selector in content_selectors:
                found_tags = soup.select(selector) # Use select for CSS selectors
                for tag in found_tags:
                    text_content = tag.get_text(separator=' ', strip=True)
                    # Simple heuristic: density of alphanumeric characters relative to total length
                    # and minimum word count
                    alphanum_chars = sum(c.isalnum() for c in text_content)
                    total_chars = len(text_content)
                    word_count = len(text_content.split())
                    
                    if total_chars > 0 and word_count >= 50: # Minimum words to consider as valid content
                        density = alphanum_chars / total_chars
                        # Score based on density and length (favor longer, denser content)
                        score = density * word_count
                        
                        if score > best_content_score:
                            best_content_score = score
                            main_content_tag = tag


            
            if main_content_tag:
                 full_text_cleaned = re.sub(r'\s{2,}', ' ', main_content_tag.get_text(separator='\n', strip=True))
            else: # Fallback if no main tag found
                 full_text_cleaned = re.sub(r'\s{2,}', ' ', soup.get_text(separator='\n', strip=True))

            # Basic quality check: minimum word count for full_text
            MIN_WORDS = 100 # Adjust as needed
            if len(full_text_cleaned.split()) < MIN_WORDS:
                logger.warning(f"Extracted content for {url} is too short ({len(full_text_cleaned.split())} words). Marking as parse_error.")
                status = "parse_error"
                error_message = "Content too short after parsing."
                page_content_raw = None # Indicate failure
            else:
                extracted_data['headline'] = soup.find('h1').get_text(strip=True) if soup.find('h1') else None
                time_tag = soup.find('time')
                extracted_data['date'] = time_tag.get('datetime') or time_tag.get_text(strip=True) if time_tag else None
                author_tag = soup.find(class_=re.compile(r'author', re.I)) or soup.find(rel='author')
                extracted_data['author'] = author_tag.get_text(strip=True) if author_tag else None
                extracted_data['full_text'] = full_text_cleaned
                words = full_text_cleaned.split()
                # Prioritize meta description as summary, fallback to first 150 words
                meta_description = soup.find('meta', attrs={'name': 'description'})
                if meta_description and meta_description.get('content'):
                    extracted_data['summary'] = meta_description['content'].strip()
                else:
                    extracted_data['summary'] = " ".join(words[:150]) + ('...' if len(words) > 150 else '')
                
                # 3. Accuracy - Content Hashing
                extracted_data['content_hash'] = hashlib.sha256(full_text_cleaned.encode('utf-8')).hexdigest()

                status = "success"
                logger.info(f"Successfully parsed {url}. Parse time: {time.time() - parse_start_time:.2f}s")

        except Exception as parse_e:
            logger.error(f"Parsing failed for {url}: {parse_e}")
            error_message = str(parse_e)
            status = "parse_error"
            # Keep raw content if parsing fails but fetch succeeded? Optional.
            # extracted_data['full_text'] = page_content_raw 

    # --- Structured Output Preparation ---
    execution_time = time.time() - start_time
    result = {
        "url": url,
        "status": status,
        "fetch_method": fetch_method,
        "headline": extracted_data['headline'],
        "date": extracted_data['date'],
        "author": extracted_data['author'],
        "summary": extracted_data['summary'],
        'full_text': extracted_data['full_text'], # Include full text for the LLM
        "content_hash": extracted_data['content_hash'],
        "error": error_message,
        "fetch_timestamp": time.time(),
        "processing_duration_seconds": round(execution_time, 2),
    }
    
    
    if status == "success":
         # Return the full_text, which seems most relevant for the LLM context
         return Document(page_content=result['full_text'] or "") # Pass full_text to LLM
         # Alternatively, return the structured dict if you refactor downstream
         # return result 
    else:
         # Simulate the original code's behavior on failure/opt-out
         return Document(page_content="") 

# You'll also need to adjust get_and_transform_page and the caching logic
# The @lru_cache should ideally cache the *structured result* if possible, 
# or at least the summary string returned.

@lru_cache(maxsize=100)
def perform_serper_search(query, use_simulation=False):
    """Perform DDG search and process results with simulation option"""
    if use_simulation:
        print("\nSimulating search process...")
        # ... (keep simulation logic) ...
    
    # Use the imported DDGS class directly
    # results = DDGS().text(query, max_results=3) 
    # urls = [r['href'] for r in results if isinstance(r, dict) and 'href' in r]

    # Initialize GoogleSerperAPIWrapper
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        raise ValueError("SERPER_API_KEY environment variable not set.")
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key, k=3)

    # Perform search using GoogleSerperAPIWrapper
    search_results = search.results(query)
    urls = [r['link'] for r in search_results.get('organic', []) if 'link' in r]

    print(f"DEBUG: Number of URLs found by Serper API: {len(urls)}") # Added for debugging
    logger.info(f"Google Serper search for '{query}' found URLs: {urls}")

    processed_docs_structured = [] # Store structured results if needed later

    if use_simulation:
        print(f"\nFound {len(urls)} relevant pages to analyze")
        embed_app(urls)
        
        # Loop for print confirmation (existing logic)
        while True:
            voice("Do you want to continue? Please say 'Yes' for an explanation or 'stop or skip' to skip.")
            resume_audio_processing()
            audio_processor = process_audio()
            translate = next(audio_processor)
            accepted_responses = {"yes", "sure", "ok", "okay", "yup", "yep"}
            negative_responses = {"nope", "no", "nah", "not really", "skip", "stop", "abort", "cancel"}

            if any(resp in translate.lower() for resp in accepted_responses):
                voice("Validated response received. Continuing with page processing...")
                break
            elif any(resp in translate.lower() for resp in negative_responses):
                print("User opted out. Aborting processing of search results.")
                logger.warning(f"User opted out of processing search results for query '{query}'.")
                return [] # Return empty list if user opts out here
            else:
                print("Response not recognized. Please try again.")

        # Process pages sequentially (to avoid concurrently running multiple print prompts)
        docs_summaries = []
        for url in urls:
             # Ensure get_and_transform_page is called correctly
             doc_result = get_and_transform_page(url, use_simulation=True, already_validated=True) 
             if doc_result and doc_result.page_content: # Check if content exists (not error/opt-out)
                 docs_summaries.append(doc_result.page_content) # This is the summary string
             # Optionally store the full structured result if you adapt _get_and_transform_page_uncached to return it
             # processed_docs_structured.append(structured_result_from_get_page) 
        
        # If user opted out during the loop, docs_summaries might be empty or partial
        if not docs_summaries: # Handle case where user said "No" to all pages
             logger.warning("User opted out of processing all search result pages.")
             return [] # Match original behavior

    else: # Non-simulation mode (parallel processing)
        url_args = [(url, False) for url in urls] # use_simulation=False
        docs_objects = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # executor.map returns the results in order
            docs_objects = list(executor.map(process_url, url_args)) # process_url needs to call get_and_transform_page

        # Extract summaries from the returned Document objects
        docs_summaries = [doc.page_content for doc in docs_objects if doc and doc.page_content]
        # Optionally collect structured data if process_url is adapted

    # Pass the summaries to the LLM prompt creator
    # Clean up empty strings just in case
    content_for_llm = [summary for summary in docs_summaries if summary] 
    logger.info(f"Passing {len(content_for_llm)} summaries to LLM prompt for query '{query}'")
    return content_for_llm # Return list of summaries for create_prompt

# get_and_transform_page remains largely the same structure, calling cached/uncached versions
def get_and_transform_page(url, use_simulation=False, already_validated=False):
    """Retrieve and transform using optimized methods."""
    logger.debug(f"Getting page {url}, simulation: {use_simulation}, validated: {already_validated}")
    if use_simulation:
        # Direct call, no caching for interactive mode to ensure prompts
        return _get_and_transform_page_uncached(url, use_simulation=True, already_validated=already_validated)
    else:
        # Use caching for non-interactive mode
        return perform_serper_search(url)

def truncate(text, word_limit=400):
    """Limit text to specified number of words"""
    words = text.split()
    return " ".join(words[:word_limit])

def create_prompt(query, search_results):
    """Create a formatted prompt with search context"""
    # Truncate each search result to a manageable length to fit within context window
    # and provide concise, relevant information.
    truncated_results = [truncate(res, word_limit=400) for res in search_results]

    prompt = (
        "Please provide a detailed, accurate, and synthesized explanation about the following topic based on the provided context.\n"
        "Analyze the context thoroughly and resolve any apparent contradictions if possible. If the answer to the question cannot be found within the provided context, please state that clearly.\n"
        "Note: If the query relates to stores, products, shopping, or purchasing, provide only basic factual information without detailed explanations.\n\n"
        f"Context:\n{'\n\n---\n\n'.join(truncated_results)}\n\n"
        f"Question: {query}\n\nDetailed Answer:"
    )
    return prompt

def create_completion_gemini(prompt, use_simulation=True):
    """Generate completion using Gemini model"""
    if use_simulation:
        print("\nGenerating response using AI model...")
        voice("Information is being processed, please wait.")
        delay = simulate_network_delay()
        print(f"AI processing took {delay:.2f} seconds")

    chat = client.start_chat(history=[])
    response = chat.send_message(prompt)
    return response

def create_completion_ollama(prompt, use_simulation=True):
    """Generate completion using Ollama model"""
    if use_simulation:
        print("\nGenerating response using AI model...")
        delay = simulate_network_delay()
        print(f"AI processing took {delay:.2f} seconds")

    try:
        response_ollama = ollama.generate(model=ollama_model, prompt=prompt)
        
        response = response_ollama['response']
        return response
        
    except Exception as e:
        print(f"Error generating Ollama response: {str(e)}")
        return f"Error: {str(e)}"

def main():
    if torch.cuda.is_available():
        print("CUDA is available. Utilizing GPU for processing.")
    else:
        print("CUDA is not available. Proceeding with CPU.")
        
    audio_processor = process_audio()
    resume_audio_processing()
    
    all_structured_results = [] # List to store results across multiple queries

    try:
        while True:
            try:
                translate = next(audio_processor)
                if translate:
                    pause_audio_processing()
                    
                    # Check if the user said a stop command.
                    if any(keyword in translate.lower() for keyword in [
                        "stop internet", "hentikan internet", "matikan internet"]):
                        print("Menghentikan akses internet...")
                        resume_audio_processing()
                        return None

                    query = translate
                    search_summaries = perform_serper_search(query, use_simulation=True)
                    
                    # If user response was negative (i.e. "No"), search_summaries will be empty
                    if not search_summaries:
                        voice("Enjoy For Read Article")
                        logger.warning(f"Skipping AI processing for query '{query}' due to opt-out or extraction failure.")
                        resume_audio_processing()
                        continue

                    prompt = create_prompt(query, search_summaries)
                    response_ollama = create_completion_ollama(prompt, use_simulation=True)
                    # Check if create_completion_ollama returned an error string
                    if isinstance(response_ollama, str) and response_ollama.startswith("Error:"):
                        print(response_ollama) # Print the error message
                        # Optionally decide how to handle the error, e.g., continue or exit
                        resume_audio_processing()
                        continue # Or return None, depending on desired behavior
                    
                    response = response_ollama # Corrected line: response_ollama is already the string
                    print(response)
                    cleaned_response = response.replace('*', '').replace('\n\n', '\n')
                    voice(cleaned_response)
                    # For simplicity, this example only logs the LLM interaction.
                    query_result_summary = {
                        "query": query,
                        "llm_response": response_ollama if not (isinstance(response_ollama, str) and response_ollama.startswith("Error:")) else "Error",
                        "timestamp": time.time(),
                        # Add summaries used if desired: "context_summaries": search_summaries 
                    }
                    all_structured_results.append(query_result_summary)
            except StopIteration:
                continue

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.exception(f"An unhandled error occurred in main loop: {e}") # Log full traceback
        # print(f"\nAn error occurred: {str(e)}") # Keep user message
        resume_audio_processing() # Keep existing error handling
    finally:
        # --- Save final structured output ---
        if all_structured_results:
             try:
                 output_filename = f"query_results_{int(time.time())}.json"
                 with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(all_structured_results, f, indent=2, ensure_ascii=False)
                 logger.info(f"Saved {len(all_structured_results)} query results to {output_filename}")
             except Exception as save_e:
                 logger.error(f"Failed to save results to JSON: {save_e}")
if __name__ == "__main__":
    main()