from flask import render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for
from app import app
from MultiAgent import app as multi_agent_app  # Import the Langchain graph app
from MultiAgent import model_rina, model_emilia, model_supervisor, rina_simple_prompt, emilia_simple_prompt, supervisor_routing_prompt, is_difficult_question, web_tools, rina_tool_executor, emilia_tool_executor
from langchain.schema import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import json
import os
from datetime import datetime
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup as soup

from langchain_community.utilities import GoogleSerperAPIWrapper
search = GoogleSerperAPIWrapper()

@app.route('/tts')
def tts():
    """Route for the TTS page"""
    return render_template('tts.html')


@app.route('/')
def index():
    """Main page route that renders the search interface"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Route for the Chat page"""
    session['conversation_messages'] = []
    session['conversation_title'] = ''
    return render_template('index.html')


@app.route('/clear-conversation')
def clear_conversation():
    """Clear conversation history and redirect to chat page"""
    session.pop('conversation_messages', None)
    session.pop('conversation_title', None)
    return redirect(url_for('chat'))



@app.route('/discover')
def discover():
    """Route for the Discover page using GoogleSerperAPIWrapper (type='news') for news search"""
    category = request.args.get('category', 'foryou')

    # Map categories to improved, high-quality Indonesian news search queries with explicit recency keywords
    from datetime import datetime
    today_str = datetime.now().strftime("%d %B %Y")
    category_queries = {
        'foryou': f"berita terbaru {today_str} Indonesia",
        'tech': f"berita teknologi {today_str} Indonesia",
        'finance': f"berita keuangan {today_str} Indonesia",
        'arts': f"berita seni {today_str} Indonesia",
        'sports': f"berita olahraga {today_str} Indonesia",
    }
    query = category_queries.get(category, category_queries['foryou'])

    # Use GoogleSerperAPIWrapper with type="news"
    from langchain_community.utilities import GoogleSerperAPIWrapper
    search = GoogleSerperAPIWrapper(type="news")

    try:
        results = search.results(query)
        # The structure of results['news'] is a list of dicts with keys: title, link, snippet, date, source, image_url (if available)
        news_items = results.get('news', [])
    except Exception as e:
        news_items = []

    articles = []
    for news in news_items:
        articles.append({
            "title": news.get('title', 'No Title'),
            "link": news.get('link', '#'),
            "pub_date": news.get('date', 'No Date'),
        })

    return render_template('discover.html', articles=articles, active_category=category)

@app.route('/fetch_image')
def fetch_image():
    """API endpoint to fetch the core image for a given article URL."""
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'URL parameter is missing'}), 400

    import requests
    from bs4 import BeautifulSoup

    def find_core_image(soup):
        # Try to find <meta property="og:image">
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return og_image["content"]
        # Try to find <meta name="twitter:image">
        twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
        if twitter_image and twitter_image.get("content"):
            return twitter_image["content"]
        # Try to find the first large <img> in the article
        article = soup.find("article")
        if article:
            imgs = article.find_all("img")
            if imgs:
                for img in imgs:
                    if img.get("src"):
                        return img["src"]
        # Fallback: first <img> in the page
        img = soup.find("img")
        if img and img.get("src"):
            return img["src"]
        return None

    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_url = find_core_image(soup)

        if img_url:
            # Handle relative URLs
            if not img_url.startswith(('http://', 'https://')):
                from urllib.parse import urljoin
                img_url = urljoin(url, img_url)
            return jsonify({'imageUrl': img_url})
        else:
            return jsonify({'imageUrl': "https://images.pexels.com/photos/1369476/pexels-photo-1369476.jpeg"})
    except Exception as e:
        return jsonify({'imageUrl': "https://images.pexels.com/photos/1369476/pexels-photo-1369476.jpeg"})


@app.route('/article-summary/<path:link>')
def article_summary(link):
    """
    Route for the Article Summary page.
    Synchronizes with backend scraping_website.py fallback logic:
    - If scraping fails or is minimal, fallback to agent-based related article search using keywords from the URL.
    """
    from scraping_website import process_article_with_fallback

    user_query = request.args.get('query', '')
    title = request.args.get('title', '')

    data = []

    # Only proceed if user_query is present (not empty)
    if not user_query.strip():
        data.append({
            "title": title,
            "link": link,
            "result": {},
            "agent_response": "Please enter a query to get a summary."
        })
        print(f"Article Summary for {link} with title: {title} - No query provided")
        return render_template(
            'article_summary.html',
            data=data,
            link=link,
            title=title
        )

    # If the preview button is pressed, just return the fallback result directly
    if user_query.strip() == 'proses':
        # Import scrape_website only when needed to avoid NameError
        from scraping_website import scrape_website
        result = scrape_website(link)
        print(f"[PREVIEW] scrape_website result: {result}")
        # Use the main content or summary as agent_response for preview
        return jsonify({
            "result": result
        })

    # Use the backend's fallback system for robust summary
    result = process_article_with_fallback(link)
    print(f"process_article_with_fallback result: {result}")

    # Compose the user question for the agent if scraping succeeded and content is sufficient
    article_text = ""
    agent_response = ""
    urls = result.get('urls', [])  # Get URLs from result

    if result.get('status') == 'success' and result.get('method_used') == 'original_scraping_with_agent':
        article_text = result.get('scraped_data', {}).get('full_text', '')
        user_question = f"{user_query}\n{article_text}\n"
        # Use the no-tools agent for successful scraping to avoid google_search
        from scraping_website import emilia_no_tools_agent
        agent_result = emilia_no_tools_agent.invoke({"input": user_question, "chat_history": []})
        agent_response = agent_result.content if hasattr(agent_result, 'content') else str(agent_result)
        # URLs are only available when fallback search is used (handled in backend)
        # No additional URL search needed here
    else:
        # If fallback_search was used, the content is already in result['content']
        agent_response = result.get('content', 'Tidak dapat menemukan ringkasan atau artikel terkait.')

    print(f"Article Summary for {link} with title: {title}")
    print(f"URLs found: {len(urls)}")

    return jsonify({
        "title": title,
        "link": link,
        "result": result,
        "agent_response": agent_response,
        "urls": urls
    })


# @app.route('/spaces')
# def spaces():
#     """Route for the Spaces page"""
#     return render_template('spaces.html')
    
@app.route('/account')
def account():
    """Route for the Account page"""
    return render_template('account.html')

@app.route('/upgrade')
def upgrade():
    """Route for the Upgrade page"""
    return render_template('upgrade.html')

@app.route('/install')
def install():
    """Route for the Install page"""
    return render_template('install.html')

@app.route('/multi-agent')
def multi_agent():
    """Route for the Multi-Agent page"""
    session['conversation_messages'] = []
    session['conversation_title'] = ''
    return render_template('multi-agent.html')


@app.route('/action/<action_type>', methods=['POST'])
def handle_action(action_type):
    """Handle different action button clicks"""
    query = request.form.get('query', '').strip()
    
    actions = {
        'health': 'Health analysis',
        'summarize': 'Content summarization',
        'analyze': 'Data analysis',
        'plan': 'Planning assistance',
        'local': 'Local search'
    }
    
    if action_type not in actions:
        return jsonify({'error': 'Invalid action type'}), 400
    
    if not query:
        return jsonify({'error': 'Please enter a query first'}), 400
    
    return jsonify({
        'action': actions[action_type],
        'query': query,
        'message': f'{actions[action_type]} would be performed for: "{query}"'
    })

@app.route('/refresh_web_page', methods=['POST', 'GET'])
def refresh_web_page():
    """
    Route to clear the query and messages from the session.
    Call this when the web page is refreshed and you want to clear conversation state.
    """
    session['conversation_messages'] = []
    session['conversation_title'] = ''
    return jsonify({'status': 'cleared'})

@app.route('/multi-agent-stream', methods=['POST'])
def multi_agent_stream():
    """
    Handle multi-agent chat requests and stream responses using the backend logic from MultiAgent_Working_Together.py.
    Now streams agent status and final result as JSON lines for real-time frontend updates.
    """
    from flask import stream_with_context, Response
    import MultiAgent_Working_Together as multi_agent_backend

    data = request.get_json()
    query = data.get('query', '').strip()
    messages = data.get('messages', [])  # <-- get full history

    if not query:
        return jsonify({'error': 'Please enter a query'}), 400

    # Save the conversation history in the session
    session['conversation_messages'] = messages

    def generate():
        # Pass the full conversation history to the agent system
        for update in multi_agent_backend.run_enhanced_agent_system_stream(query, messages=messages):
            yield update

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
    
    

@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    """Handle chat requests and stream responses from the Multi-Agent system using the full LangGraph workflow."""
    data = request.get_json()
    query = data.get('query', '').strip()
    is_new_conversation = data.get('is_new_conversation', True)

    if not query:
        return jsonify({'error': 'Please enter a query'}), 400

    def generate():
        # Get existing conversation history from session or start fresh
        if is_new_conversation:
            # Start new conversation
            conversation_messages = [HumanMessage(content=query)]
            session['conversation_messages'] = [HumanMessage(content=query)]
            session['conversation_title'] = query
        else:
            # Continue existing conversation
            conversation_messages = session.get('conversation_messages', [])
            # Add the new query to conversation history
            conversation_messages.append(HumanMessage(content=query))
        
        current_agent = "Supervisor"
        
        # Continue conversation until supervisor decides to finish
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if current_agent == "Supervisor":
                # Supervisor makes decision
                rina_workload = session.get('rina_workload', 0)
                emilia_workload = session.get('emilia_workload', 0)
                supervisor_input = {
                    "messages": conversation_messages,
                    "input": conversation_messages[-1].content,
                    "members": "Rina, Emilia",
                    "rina_workload": rina_workload,
                    "emilia_workload": emilia_workload,
                }
                
                # Stream supervisor's decision
                full_decision = ""
                for chunk in model_supervisor.stream(supervisor_routing_prompt.format_messages(**supervisor_input)):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_decision += chunk.content
                        # Do not yield chunk directly, as it might contain <think> tags
                        # yield f"data: {json.dumps({'sender': 'Supervisor', 'content': chunk.content, 'type': 'chunk'})}\n\n"
                
                # Clean the full_decision to remove <think>...</think> tags
                cleaned_content = re.sub(r"<think>.*?</think>\n?", "", full_decision, flags=re.DOTALL).strip()
                
                # Send complete message for supervisor with cleaned content
                yield f"data: {json.dumps({'sender': 'Supervisor', 'content': cleaned_content, 'type': 'complete'})}\n\n"
                
                # Add supervisor message to conversation (using cleaned content)
                supervisor_message = AIMessage(content=cleaned_content, name="Supervisor")
                conversation_messages.append(supervisor_message)
                # Update session with conversation history
                session['conversation_messages'] = conversation_messages
                
                # Parse supervisor's decision
                def parse_supervisor_decision(decision):
                    match = re.search(r"route_to:\s*(rina|emilia)", decision, re.IGNORECASE)
                    if match:
                        return match.group(1).capitalize()
                    if "finish" in decision.lower():
                        return "FINISH"
                    return "Rina"  # Default fallback

                next_agent_name = parse_supervisor_decision(full_decision)
                if next_agent_name in ["Rina", "Emilia"]:
                    current_agent = next_agent_name
                    # Update workload in session
                    if current_agent == "Rina":
                        session['rina_workload'] = rina_workload + 1
                    elif current_agent == "Emilia":
                        session['emilia_workload'] = emilia_workload + 1
                    # Send routing message
                    yield f"data: {json.dumps({'sender': 'Supervisor', 'content': f'ROUTE_TO: {next_agent_name} - Routing to {next_agent_name}', 'type': 'complete'})}\n\n"
                    continue
                elif next_agent_name == "FINISH":
                    yield f"data: {json.dumps({'sender': 'Supervisor', 'content': 'ROUTE_TO: FINISH - Conversation complete', 'type': 'complete'})}\n\n"
                    break
                else:
                    # Default to finish if no clear decision
                    yield f"data: {json.dumps({'sender': 'Supervisor', 'content': 'ROUTE_TO: FINISH - Default finish', 'type': 'complete'})}\n\n"
                    break
            
            elif current_agent in ["Rina", "Emilia"]:
                # Agent responds to the current question
                current_question = conversation_messages[-1].content
                chat_history = conversation_messages[:-1]
                difficulty_analysis = is_difficult_question(current_question)
                use_tools = difficulty_analysis["use_tools"]
                if current_agent in ["Rina", "Emilia"]:
                    use_tools = True
                full_content = ""
                try:
                    executor_to_use = rina_tool_executor if current_agent == "Rina" else emilia_tool_executor
                    simple_runnable_to_use = rina_simple_prompt | model_rina | StrOutputParser() if current_agent == "Rina" else emilia_simple_prompt | model_emilia | StrOutputParser()
                    # Use new agent_node streaming logic
                    from MultiAgent import agent_node
                    agent_stream = agent_node(
                        {"messages": conversation_messages, "metadata": {}},
                        simple_runnable_to_use,
                        executor_to_use,
                        current_agent
                    )
                    for item in agent_stream:
                        if item["type"] == "chunk":
                            full_content += item["content"]
                            yield f"data: {json.dumps({'sender': current_agent, 'content': item['content'], 'type': 'chunk'})}\n\n"
                        elif item["type"] == "system":
                            yield f"data: {json.dumps({'sender': item['sender'], 'content': item['content'], 'type': 'system'})}\n\n"
                        elif item["type"] == "complete":
                            yield f"data: {json.dumps({'sender': current_agent, 'content': full_content, 'type': 'complete'})}\n\n"
                    # Add agent response to conversation
                    agent_message = AIMessage(content=full_content, name=current_agent)
                    conversation_messages.append(agent_message)
                    session['conversation_messages'] = conversation_messages
                except Exception as e:
                    print(f"[DEBUG] {current_agent} tool execution failed: {str(e)}. Falling back to simple response.")
                    for chunk in simple_runnable_to_use.stream({
                        "input": current_question,
                        "chat_history": conversation_messages,
                    }):
                        if chunk:
                            full_content += chunk
                            yield f"data: {json.dumps({'sender': current_agent, 'content': chunk, 'type': 'chunk'})}\n\n"
                    yield f"data: {json.dumps({'sender': current_agent, 'content': full_content, 'type': 'complete'})}\n\n"
                    agent_message = AIMessage(content=full_content, name=current_agent)
                    conversation_messages.append(agent_message)
                    session['conversation_messages'] = conversation_messages
                current_agent = "Supervisor"
        
        # Send end stream signal
        yield "event: end_stream\n"
        yield "data: [END]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    data = request.get_json()
    conversation = data.get('conversation', [])

    if not conversation:
        return jsonify({'status': 'error', 'message': 'No conversation data provided.'}), 400

    # Create a directory to save conversations if it doesn't exist
    save_dir = 'conversations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f'conversation_{timestamp}.json')

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=4)
        return jsonify({'status': 'success', 'message': 'Conversation saved.', 'file_path': file_path}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/list_conversations')
def list_conversations():
    conversations_dir = 'conversations'
    if not os.path.exists(conversations_dir):
        return jsonify([])
    conversation_files = [f for f in os.listdir(conversations_dir) if f.endswith('.json')]
    return jsonify(conversation_files)


@app.route('/view_conversation/<filename>')
def view_conversation(filename):
    conversation_path = os.path.join('conversations', filename)
    if not os.path.exists(conversation_path):
        return jsonify({'status': 'error', 'message': 'Conversation not found.'}), 404
    with open(conversation_path, 'r') as f:
        conversation_data = json.load(f)

    # Transform the conversation data into the expected format
    # Extract title from filename or use a default
    title = filename.replace('.json', '').replace('conversation_', 'Conversation ')
    
    # Extract messages from the conversation data
    messages = conversation_data if isinstance(conversation_data, list) else []
    
    # Generate related questions based on the conversation content
    related_questions = []
    for message in messages:
        if message.get('type') == 'chat' and message.get('sender') == 'user':
            content = message.get('content', '')
            if content and len(content) > 10:  # Only add substantial questions
                related_questions.append(content)
    
    # Limit to 5 related questions
    related_questions = related_questions[:5]
    
    # Create the expected conversation structure
    conversation = {
        'title': title,
        'messages': messages,
        'related_questions': related_questions
    }

    return render_template('conversations.html', conversation=conversation)


@app.route ('/get_all_conversation/agent')
def get_all_conversation_agent():
    """Get all conversation from agent, return as list of dicts like conversation_*.json"""
    conversation_messages = session.get('conversation_messages', [])
    # Ensure the output is a list of dicts with keys: sender, content, type, timestamp
    formatted_messages = []
    for msg in conversation_messages:
        # If already a dict with the right keys, just use it
        if isinstance(msg, dict) and all(k in msg for k in ['sender', 'content', 'type', 'timestamp']):
            formatted_messages.append(msg)
        else:
            # Try to extract fields from possible LangChain or other message objects
            sender = getattr(msg, 'sender', None) or getattr(msg, 'role', None) or getattr(msg, 'name', None) or 'unknown'
            content = getattr(msg, 'content', None) or msg.get('content', '') if isinstance(msg, dict) else ''
            msg_type = getattr(msg, 'type', None) or getattr(msg, '__class__', type('X', (), {})).__name__ or msg.get('type', 'chat') if isinstance(msg, dict) else 'chat'
            timestamp = getattr(msg, 'timestamp', None) or msg.get('timestamp', None) if isinstance(msg, dict) else None
            if not timestamp:
                from datetime import datetime
                timestamp = datetime.utcnow().isoformat() + "Z"
            formatted_messages.append({
                "sender": sender,
                "content": content,
                "type": msg_type,
                "timestamp": timestamp
            })
    return jsonify(formatted_messages)


@app.route('/get_conversation_history')
def get_conversation_history():
    """Get conversation history from session"""
    conversation_messages = session.get('conversation_messages', [])
    conversation_title = session.get('conversation_title', '')
    
    # Convert LangChain messages to serializable format
    serialized_messages = []
    for msg in conversation_messages:
        if hasattr(msg, 'content') and hasattr(msg, '__class__'):
            message_data = {
                'type': msg.__class__.__name__,
                'content': msg.content
            }
            if hasattr(msg, 'name') and msg.name:
                message_data['name'] = msg.name
            serialized_messages.append(message_data)
    
    return jsonify({
        'conversation_messages': serialized_messages,
        'conversation_title': conversation_title
    })

@app.route('/grpo')
def grpo():
    """Route for the GRPO page"""
    return render_template('grpo.html')