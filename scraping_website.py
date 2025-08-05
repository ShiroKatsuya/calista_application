from langchain_community.utilities import GoogleSerperAPIWrapper
import requests
from bs4 import BeautifulSoup
import re
import hashlib
import time
import logging
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

def extract_keywords_from_url(url: str) -> str:
    """
    Extract meaningful keywords from a URL path for search purposes.
    Converts URL slugs into searchable keywords.
    """
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        
        if not path:
            return ""
        
        # Remove file extensions and common URL patterns
        path = re.sub(r'\.(html|htm|php|asp|aspx)$', '', path)
        path = re.sub(r'/\d{4}/\d{2}/\d{2}/', '/', path)  # Remove date patterns
        path = re.sub(r'/\d+$', '', path)  # Remove trailing numbers
        
        # Split by common separators and clean up
        keywords = re.split(r'[-_/]', path)
        keywords = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip()) > 2]
        
        # Remove common stop words and short terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        keywords = [kw for kw in keywords if kw.lower() not in stop_words and len(kw) > 2]
        
        # Join keywords with spaces for search
        search_query = ' '.join(keywords)
        
        logger.info(f"Extracted keywords from URL {url}: {search_query}")
        return search_query
        
    except Exception as e:
        logger.error(f"Error extracting keywords from URL {url}: {e}")
        return ""

def is_minimal_content(article_text: str, min_words: int = 100) -> bool:
    """
    Check if the extracted article content is minimal or insufficient.
    """
    if not article_text:
        return True
    
    word_count = len(article_text.split())
    return word_count < min_words

def search_related_articles(original_url: str, agent_executor) -> dict:
    """
    Search for related articles when the original scraping fails or provides minimal content.
    Uses multiple search strategies to get a variety of information from different sources.
    """
    try:
        # Extract keywords from the original URL
        keywords = extract_keywords_from_url(original_url)
        
        if not keywords:
            logger.warning(f"Could not extract meaningful keywords from URL: {original_url}")
            return {
                "content": "Tidak dapat mengekstrak kata kunci yang bermakna dari URL asli.",
                "urls": []
            }
        
        # Create multiple search queries for comprehensive coverage
        search_queries = [
            f"latest news {keywords} recent developments",
            f"breaking news {keywords} today",
            f"{keywords} latest updates",
            f"recent developments {keywords}",
            f"news analysis {keywords}"
        ]
        
        logger.info(f"Searching for related articles with multiple queries for: {keywords}")
        
        # Use the agent to search for related information with multiple strategies
        search_prompt = f"""
        Cari informasi terbaru dan artikel terkait tentang: {keywords}
        
        Gunakan beberapa strategi pencarian untuk mendapatkan informasi yang beragam:
        1. Berita terbaru dan perkembangan terkini
        2. Analisis dan opini dari berbagai sumber
        3. Fakta-fakta terbaru dan data terkini
        4. Perspektif dari berbagai media dan sumber
        
        Pastikan untuk memberikan informasi yang:
        - Akurat dan terkini
        - Berasal dari berbagai sumber terpercaya
        - Memberikan perspektif yang beragam
        - Mencakup aspek-aspek berbeda dari topik tersebut
        """
        
        response = agent_executor.invoke({
            "input": search_prompt, 
            "chat_history": []
        })
        
        # Get the URLs from the search results
        search_result = google_search(f"latest news {keywords} recent developments")
        
        return {
            "content": response.get("output", "Tidak dapat menemukan informasi terkait."),
            "urls": search_result.get("urls", [])
        }
        
    except Exception as e:
        logger.error(f"Error searching for related articles: {e}")
        return {
            "content": f"Terjadi kesalahan saat mencari artikel terkait: {str(e)}",
            "urls": []
        }

def process_article_with_fallback(url: str, agent_executor=None) -> dict:
    """
    Main function to process an article with automatic fallback to related article search.
    
    Args:
        url: The URL of the article to process
        agent_executor: The agent executor to use for fallback search (optional)
    
    Returns:
        dict: Contains the processing result and method used
    """
    if agent_executor is None:
        agent_executor = Implementasi_tool_executor
    
    result = {
        'original_url': url,
        'method_used': 'unknown',
        'content': '',
        'status': 'failed',
        'error': None
    }
    
    try:
        # First, attempt to scrape the original article
        scrape_result = scrape_website(url)
        
        # Check if scraping was successful and content is sufficient
        if scrape_result.get('status') == 'success' and not is_minimal_content(scrape_result.get('full_text', '')):
            # Original scraping worked well
            article_text = scrape_result.get('full_text', '')
            
            # Process with no-tools agent to avoid google_search
            user_question = f"Ringkaslah artikel berikut dan berikan fakta terbaru terkait topik ini:\n{article_text}\n"
            
            agent_response = Implementasi_no_tools_agent.invoke({
                "input": user_question, 
                "chat_history": []
            })
            
            result.update({
                'method_used': 'original_scraping_with_agent',
                'content': agent_response.content if hasattr(agent_response, 'content') else str(agent_response),
                'status': 'success',
                'scraped_data': scrape_result
            })
            
        else:
            # Scraping failed or provided minimal content, use fallback search
            logger.info(f"Article scraping failed or minimal content. Using fallback search for: {url}")
            
            fallback_result = search_related_articles(url, agent_executor)
            
            result.update({
                'method_used': 'fallback_search',
                'content': fallback_result.get('content', ''),
                'urls': fallback_result.get('urls', []),
                'status': 'success',
                'scraped_data': scrape_result
            })
            
    except Exception as e:
        logger.error(f"Error processing article {url}: {e}")
        result.update({
            'status': 'failed',
            'error': str(e)
        })
    
    return result

def extract_real_article_url_from_google_news(soup):
    # Try to find the real article URL in <a> tags or meta tags
    # Google News RSS pages often have a canonical link or a main <a> tag
    canonical = soup.find('link', rel='canonical')
    if canonical and canonical.get('href'):
        return canonical['href']
    # Try to find the first large <a> tag
    main_link = soup.find('a', href=True)
    if main_link and main_link['href'].startswith('http'):
        return main_link['href']
    # Try meta refresh
    meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
    if meta_refresh and 'url=' in meta_refresh.get('content', ''):
        return meta_refresh['content'].split('url=')[-1]
    return None

def scrape_website(url: str) -> dict:
    extracted_data = {}
    status = "unknown"
    error_message = None
    parse_start_time = time.time()
    try:
        # 1. Download page
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page_content_raw = resp.text
        soup = BeautifulSoup(page_content_raw, 'html.parser')

        # --- Google News RSS Redirect Handling ---
        if 'news.google.com/rss/articles/' in url:
            real_url = extract_real_article_url_from_google_news(soup)
            if real_url and real_url != url:
                try:
                    logger.info(f"Following real article URL: {real_url}")
                    resp = requests.get(real_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                    resp.raise_for_status()
                    page_content_raw = resp.text
                    soup = BeautifulSoup(page_content_raw, 'html.parser')
                except Exception as e:
                    logger.warning(f"Failed to follow real article URL: {real_url}. Error: {e}")
                    # Fallback to original soup

        # Remove noisy elements (navigation, headers, footers, scripts, styles, ads, comments)
        for selector in ['nav', 'footer', 'aside', 'header', 'form', 'script', 'style',
                         '[class*="ad"]', '[id*="ad"]', '[class*="comment"]', '[id*="comment"]',
                         '.sidebar', '.ads', '.modal', '.popup']:
            for element in soup.select(selector):
                element.decompose()

        # --- Improved Headline and Main Story Extraction ---
        headline_tag = soup.find('h1')
        extracted_data['headline'] = headline_tag.get_text(strip=True) if headline_tag else None
        main_story_text = None

        if headline_tag:
            # Try to find the closest parent that is an <article> or a main content class
            main_container = None
            parent = headline_tag.parent
            while parent and parent != soup:
                if parent.name == 'article' or (
                    parent.has_attr('class') and any(
                        c in parent['class'] for c in [
                            'article-content', 'post-content', 'entry-content', 'main-content', 'content-body',
                            'story-content', 'single-post-content', 'news-body', 'item-page', 'blog-post',
                            'post-area', 'article-body', 'g-article', 's-content', 'body-content', 'page-content',
                            'td-post-content', 'richtext'
                        ]
                    )
                ):
                    main_container = parent
                    break
                parent = parent.parent
            if main_container:
                main_story_text = main_container.get_text(separator='\n', strip=True)
            else:
                # Fallback: use the headline's parent text
                main_story_text = headline_tag.parent.get_text(separator='\n', strip=True)
        else:
            # Fallback: use the best content block as before
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
                found_tags = soup.select(selector)
                for tag in found_tags:
                    text_content = tag.get_text(separator=' ', strip=True)
                    alphanum_chars = sum(c.isalnum() for c in text_content)
                    total_chars = len(text_content)
                    word_count = len(text_content.split())
                    if total_chars > 0 and word_count >= 50:
                        density = alphanum_chars / total_chars
                        score = density * word_count
                        if score > best_content_score:
                            best_content_score = score
                            main_content_tag = tag
            if main_content_tag:
                main_story_text = re.sub(r'\s{2,}', ' ', main_content_tag.get_text(separator='\n', strip=True))
            else:
                main_story_text = re.sub(r'\s{2,}', ' ', soup.get_text(separator='\n', strip=True))

        # Clean and organize the main story text
        MIN_WORDS = 50
        if not main_story_text or len(main_story_text.split()) < MIN_WORDS:
            logger.warning(f"Extracted main story for {url} is too short ({len(main_story_text.split()) if main_story_text else 0} words). Marking as parse_error.")
            status = "parse_error"
            error_message = "Main story content too short after parsing."
            main_story_text = None
        else:
            extracted_data['full_text'] = main_story_text
            words = main_story_text.split()
            meta_description = soup.find('meta', attrs={'name': 'description'})
            if meta_description and meta_description.get('content'):
                extracted_data['summary'] = meta_description['content'].strip()
            else:
                extracted_data['summary'] = " ".join(words[:100]) + ('...' if len(words) > 100 else '')
            time_tag = soup.find('time')
            extracted_data['date'] = time_tag.get('datetime') or time_tag.get_text(strip=True) if time_tag else None
            author_tag = soup.find(class_=re.compile(r'author', re.I)) or soup.find(rel='author')
            extracted_data['author'] = author_tag.get_text(strip=True) if author_tag else None
            extracted_data['content_hash'] = hashlib.sha256(main_story_text.encode('utf-8')).hexdigest()
            status = "success"
            logger.info(f"Successfully parsed {url}. Parse time: {time.time() - parse_start_time:.2f}s")

    except Exception as parse_e:
        logger.error(f"Parsing failed for {url}: {parse_e}")
        error_message = str(parse_e)
        status = "parse_error"

    extracted_data['status'] = status
    if error_message:
        extracted_data['error'] = error_message
    return extracted_data

from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder

# Define a Google Search tool using a placeholder implementation


def google_search(query: str) -> dict:
    """
    Perform Google search using the Serper API to get multiple relevant articles.
    Returns a comprehensive summary of various sources and perspectives.
    """
    print(f"Searching for: {query}")
    try:
        # Configure Serper API to return multiple results
        search = GoogleSerperAPIWrapper(
            k=10,  # Get up to 10 results
            gl="id",  # Geolocation for Indonesia
            hl="id",  # Language for Indonesia
            tbs="qdr:m"  # Time-based search: past month for recent articles
        )
        
        # Get multiple search results
        results = search.results(query)
        
        if not results or not results.get('organic'):
            return {
                "summary": f"Tidak dapat menemukan hasil pencarian untuk: {query}",
                "urls": [],
                "articles": []
            }
        
        # Extract and format multiple articles
        articles = results['organic'][:10]  # Get up to 10 articles
        formatted_results = []
        urls = []
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            snippet = article.get('snippet', 'No description')
            link = article.get('link', 'No link')
            
            formatted_article = f"{i}. {title}\n   {snippet}\n   Sumber: {link}\n"
            formatted_results.append(formatted_article)
            urls.append({
                "index": i,
                "title": title,
                "url": link
            })
        
        # Create a comprehensive summary
        summary = f"Hasil pencarian untuk '{query}':\n\n"
        summary += "Artikel-artikel terkait yang ditemukan:\n"
        summary += "=" * 50 + "\n\n"
        summary += "\n".join(formatted_results)
        
        # Add a note about the variety of sources
        summary += f"\n\nTotal {len(articles)} artikel ditemukan dari berbagai sumber. "
        summary += "Informasi ini memberikan perspektif yang beragam dan terkini tentang topik tersebut."
        
        print(f"Search completed successfully - found {len(articles)} articles")
        # Display the URLs of all found articles for reference and citation
        print("Daftar URL artikel yang ditemukan:")
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article.get('link', 'No link')}")
        
        return {
            "summary": summary,
            "urls": urls,
            "articles": articles
        }
        
    except Exception as e:
        logger.error(f"Error during Google Search: {str(e)}")
        return {
            "summary": f"Error during Google Search: {str(e)}",
            "urls": [],
            "articles": []
        }

def google_search_wrapper(query: str) -> str:
    """
    Wrapper function for google_search that returns only the summary string
    for compatibility with the agent system.
    """
    result = google_search(query)
    # Store URLs globally for later retrieval
    global last_search_urls
    last_search_urls = result.get("urls", [])
    return result["summary"]

# Global variable to store URLs from the last search
last_search_urls = []

# --- AGENT TOOL WRAPPERS ---
search = GoogleSerperAPIWrapper()

def google_search_more(urls: str) -> str:
    # Run the search multiple times for consistency
    print("qqeury", urls)
    try:
        result = search.run(urls)
        print("google_more",result)
        return result
    except Exception as e:
        return f"Error during Google Search: {str(e)}"

web_tools = [
    Tool(
        name="google_search_wrapper",
        description="Returns comprehensive information and up-to-date coverage of topics.",
        func=google_search_wrapper,
    ),
    Tool(
        name="google_search_more",
        description="Performs a Google search for the given query and returns more results. Input should be a search query.",
        func=google_search_more,
    ),
    Tool(
        name="google_search",
        description="Scrape the main content of a given article URL. Use this to extract the full text of the article directly from the website. Input should be a valid URL.",
        func=google_search,
    ),

]
print(web_tools)
from langchain_ollama import ChatOllama


# Set up the Gemini model with tool-calling capability
# model_Implementasi = ChatOllama(model="llama3-2.3b:latest")

model_implementasi = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
)





# Prompt for when we have successful scraping (no tools needed)
Implementasi_no_tools_prompt = ChatPromptTemplate.from_messages([
   
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# Prompt for fallback search (with tools)
Implementasi_tool_prompt = ChatPromptTemplate.from_messages([

    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Agent for successful scraping (no tools)
Implementasi_no_tools_agent = Implementasi_no_tools_prompt | model_implementasi

# Agent for fallback search (with tools)
Implementasi_tool_agent = create_tool_calling_agent(model_implementasi, web_tools, Implementasi_tool_prompt)
Implementasi_tool_executor = AgentExecutor(
    agent=Implementasi_tool_agent, 
    tools=web_tools, 
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate"
)

# Example usage:
if __name__ == "__main__":
    # Test with the example URL provided
    test_url = "https://www.nytimes.com/2025/07/19/us/politics/inside-trump-epstein-friendship.html"
    
    print("=" * 80)
    print("Testing article processing with automatic fallback")
    print("=" * 80)
    print(f"URL: {test_url}")
    print()
    
    # Use the new comprehensive function
    result = process_article_with_fallback(test_url)
    
    print(f"Processing method used: {result['method_used']}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print("\n" + "=" * 50)
        print("PROCESSED CONTENT:")
        print("=" * 50)
        print(result['content'])
    else:
        print(f"Error: {result['error']}")
    
    print("\n" + "=" * 80)
    print("Testing keyword extraction:")
    print("=" * 80)
    keywords = extract_keywords_from_url(test_url)
    print(f"Extracted keywords: {keywords}")
    
    # Test keyword extraction for the PBS URL specifically
    print("\n" + "=" * 80)
    print("Detailed keyword extraction test for PBS URL:")
    print("=" * 80)
    pbs_keywords = extract_keywords_from_url(test_url)
    print(f"Original URL: {test_url}")
    print(f"Extracted keywords: '{pbs_keywords}'")
    print(f"Expected search query: 'latest news {pbs_keywords} recent developments'")
