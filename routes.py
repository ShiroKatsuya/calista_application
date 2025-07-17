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



@app.route('/')
def index():
    """Main page route that renders the search interface"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Route for the Chat page"""
    return render_template('index.html')


@app.route('/clear-conversation')
def clear_conversation():
    """Clear conversation history and redirect to chat page"""
    session.pop('conversation_messages', None)
    session.pop('conversation_title', None)
    return redirect(url_for('chat'))

@app.route('/discover')
def discover():
    """Route for the Discover page"""
    return render_template('discover.html')

@app.route('/spaces')
def spaces():
    """Route for the Spaces page"""
    return render_template('spaces.html')

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

    if not query:
        return jsonify({'error': 'Please enter a query'}), 400

    def generate():
        for update in multi_agent_backend.run_enhanced_agent_system_stream(query):
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