from liblary import *
from core_code import *
import json


agents = {
    "Alice": Agent("Alice", model_alice, "General Knowledge Agent", 
                   ["science", "technology", "humanities", "arts", "business", "research", "analysis"], web_tools),
    "Bob": Agent("Bob", model_bob, "Technical Implementation Agent", 
                 ["coding", "programming", "implementation", "technical", "architecture", "development"], web_tools),
    "Charlie": Agent("Charlie", model_charlie, "Creative Problem Solver", 
                     ["creativity", "innovation", "problem-solving", "design", "strategy", "brainstorming"], web_tools),
    "Synthesizer": Agent("Synthesizer", model_synthesizer, "Enhanced Response Synthesis Specialist", ["synthesis", "integration", "summary", "clarity", "coherence", "optimization"], []),
}

def create_enhanced_collaborative_prompt(agent: Agent, thread_safe_state: ThreadSafeState, task_description: str):
    other_outputs = thread_safe_state.get_other_agent_outputs(agent.name)
    collaboration_context = ""
    
    if other_outputs:
        collaboration_context = "\n\n🔗 REAL-TIME COLLABORATION CONTEXT:\n"
        for agent_name, output in other_outputs.items():
            collaboration_context += f"\n📝 {agent_name}'s contribution:\n{output[:500]}...\n"
        collaboration_context += "\n💡 Build upon these insights and provide complementary perspectives.\n"
    
    # Add task-specific collaboration instructions
    task_lower = task_description.lower()
    if any(word in task_lower for word in ["complex", "difficult", "challenging"]):
        collaboration_context += "\n⚠️  This is a complex task. Collaborate intensively with other agents.\n"
    
    return PromptTemplate.from_template(
        f"""You are {agent.name}, {agent.description}. Expertise: {', '.join(agent.expertise)}.

{collaboration_context}

You are working in a TRUE PARALLEL collaborative environment. Other agents are working simultaneously.
Consider their contributions in real-time and provide complementary insights.

IMPORTANT COLLABORATION RULES:
1. Build upon other agents' insights
2. Fill gaps they haven't addressed
3. Provide unique perspectives from your expertise
4. Avoid duplicating work already done
5. Enhance the overall solution quality

Conversation History: {{chat_history}}
User Request: {{input}}

Your response (focus on collaboration and your unique expertise):"""
    )

class EnhancedParallelTaskExecutor:
    def __init__(self):
        self.thread_safe_state = ThreadSafeState()
        self.execution_metrics = {"total_tasks": 0, "successful_tasks": 0, "avg_execution_time": 0.0}
        
    def execute_task_parallel(self, state: AgentState, task: AgentTask) -> AgentTask:
        agent = agents[task.agent_name]
        agent.is_busy = True
        self.thread_safe_state.increment_agent_load(agent.name)
        self.thread_safe_state.update_agent_status(agent.name, "working")
        
        start_time = time.time()
        
        try:
            self.thread_safe_state.update_task(task.id, status=TaskStatus.IN_PROGRESS)
            
            # Create enhanced collaborative prompt
            collaborative_prompt = create_enhanced_collaborative_prompt(agent, self.thread_safe_state, task.task_description)
            
            # Use tool-enabled agent with enhanced collaboration
            tool_prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are {agent.name}, {agent.description}. Use tools for comprehensive responses and collaborate with other agents in real-time."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            tool_agent = create_tool_calling_agent(agent.model, agent.tools, tool_prompt)
            tool_executor = AgentExecutor(agent=tool_agent, tools=agent.tools, verbose=False)
            
            # Execute with real-time collaboration monitoring
            result = tool_executor.invoke({
                "input": task.task_description,
                "chat_history": state["messages"]
            })
            
            task.result = result["output"]
            self.thread_safe_state.update_agent_output(agent.name, task.result)
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # Update performance metrics
            execution_time = task.completed_at - start_time
            agent.update_performance(execution_time)
            self.execution_metrics["successful_tasks"] += 1
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Task failed: {str(e)}"
            
        finally:
            agent.is_busy = False
            self.thread_safe_state.decrement_agent_load(agent.name)
            self.thread_safe_state.update_agent_status(agent.name, "available")
            self.thread_safe_state.update_task(task.id, status=task.status)
        
        self.execution_metrics["total_tasks"] += 1
        return task

class IntelligentTaskDistributor:
    def __init__(self):
        self.task_executor = EnhancedParallelTaskExecutor()
        
    def analyze_task_complexity(self, user_input: str) -> float:
        """Analyze task complexity for optimal distribution."""
        complexity_score = 0.0
        input_lower = user_input.lower()
        
        # Complexity indicators
        complexity_keywords = ["complex", "difficult", "challenging", "comprehensive", "detailed", "analysis"]
        complexity_score += sum(0.2 for keyword in complexity_keywords if keyword in input_lower)
        
        # Length and structure indicators
        if len(user_input.split()) > 20:
            complexity_score += 0.3
        if user_input.count('?') > 1:
            complexity_score += 0.2
            
        return min(complexity_score, 1.0)
    
    def select_optimal_agents(self, user_input: str, complexity_score: float) -> List[str]:
        """Select optimal agents based on task requirements and agent capabilities."""
        available_agents = list(agents.keys())
        
        # For demonstration and testing, always use all agents to show parallel execution
        # In production, you would use the complexity-based logic below
        print(f"🔍 Task complexity score: {complexity_score:.2f}")
        print(f"🎯 Using all agents for maximum parallel execution")
        return available_agents
  
    
    def distribute_tasks_parallel(self, state: AgentState, user_input: str) -> List[AgentTask]:
        complexity_score = self.analyze_task_complexity(user_input)
        optimal_agents = self.select_optimal_agents(user_input, complexity_score)
        
        tasks = []
        timestamp = int(time.time())
        
        for agent_name in optimal_agents:
            task = AgentTask(
                id=f"task_{len(state['tasks'])}_{agent_name}_{timestamp}",
                agent_name=agent_name,
                task_description=user_input,
                complexity_score=complexity_score,
                priority=1 if complexity_score > 0.5 else 2
            )
            tasks.append(task)
        
        return tasks

class EnhancedCoordinator:
    def __init__(self):
        self.task_distributor = IntelligentTaskDistributor()
        self.thread_safe_state = ThreadSafeState()
        self.coordination_metrics = {"total_coordinations": 0, "avg_synthesis_time": 0.0}
    
    def coordinate_parallel_execution(self, state: AgentState, user_input: str) -> AgentState:
        state["thread_safe_state"] = self.thread_safe_state
        tasks = self.task_distributor.distribute_tasks_parallel(state, user_input)
        
        print(f"\n🚀 PARALLEL EXECUTION STARTING")
        print(f"📋 Tasks to execute: {len(tasks)}")
        for task in tasks:
            print(f"   - {task.agent_name}: {task.task_description[:50]}...")
        
        # Add collaboration events tracking
        state["collaboration_events"] = []
        state["system_metrics"] = {"start_time": time.time()}
        
        for task in tasks:
            state["tasks"][task.id] = task
            self.thread_safe_state.add_task(task)
        
        # Execute tasks with enhanced parallel processing (excluding Synthesizer)
        parallel_agents = [name for name in agents.keys() if name != "Synthesizer"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_agents)) as executor:
            future_to_task = {
                executor.submit(self.task_distributor.task_executor.execute_task_parallel, state, task): task
                for task in tasks if task.agent_name != "Synthesizer"
            }
            
            print(f"⚡ Executing {len(future_to_task)} tasks in parallel...")
            
            # Monitor execution with real-time collaboration
            completed_tasks = []
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    completed_task = future.result()
                    completed_tasks.append(completed_task)
                    
                    if completed_task.status == TaskStatus.COMPLETED:
                        state["agent_outputs"][completed_task.agent_name] = completed_task.result
                        print(f"✅ {completed_task.agent_name} completed task: {completed_task.result}")
                        # Track collaboration events
                        state["collaboration_events"].append({
                            "agent": completed_task.agent_name,
                            "timestamp": time.time(),
                            "status": "completed"
                        })
                except Exception as e:
                    print(f"❌ Task {task.id} failed: {str(e)}")
        
        print(f"🎉 Parallel execution completed. {len(completed_tasks)} tasks finished.")
        
        # Update final state
        state["agent_outputs"].update(self.thread_safe_state.get_agent_outputs())
        state["system_metrics"]["end_time"] = time.time()
        state["system_metrics"]["execution_time"] = state["system_metrics"]["end_time"] - state["system_metrics"]["start_time"]

        return state

    def synthesize_results_enhanced(self, state: AgentState) -> str:
        if not state["agent_outputs"]:
            return "No agent outputs to synthesize."

        synthesis_start = time.time()
        # Prepare a single, unified synthesis prompt for the Synthesizer agent
        agent_outputs_text = "\n".join([
            output for name, output in state["agent_outputs"].items() if name != "Synthesizer"
        ])
        synthesis_prompt = PromptTemplate.from_template(
            """You are the SYNTHESIZER, an expert at creating a single, unified, high-quality answer from multiple agent contributions.\n\nAgent Contributions (raw, do not mention agent names):\n{agent_outputs}\n\nUser's Request: {user_input}\n\nInstructions:\n- Carefully analyze, compare, and integrate all agent contributions.\n- Resolve any contradictions, fill in missing details, and ensure the answer is complete and accurate.\n- Do NOT mention or identify any agent names or roles.\n- Do NOT list or enumerate the contributions.\n- Write a single, seamless, well-structured answer that is clear, deep, and valuable for the user.\n- The result should read as if written by one top expert, not a group.\n- Always provide the best possible answer, maximizing insight and usefulness.\n\nUnified Synthesized Response:"""
        )
        synthesizer = agents["Synthesizer"]
        synthesis_runnable = synthesis_prompt | synthesizer.model | StrOutputParser()
        synthesis = synthesis_runnable.invoke({
            "agent_outputs": agent_outputs_text,
            "user_input": state["messages"][-1].content
        })
        synthesis_time = time.time() - synthesis_start
        self.coordination_metrics["avg_synthesis_time"] = synthesis_time
        self.coordination_metrics["total_coordinations"] += 1
        return synthesis

def create_enhanced_parallel_agent_graph():
    coordinator = EnhancedCoordinator()
    
    def parallel_execution_node(state: AgentState):
        user_input = state["messages"][-1].content
        updated_state = coordinator.coordinate_parallel_execution(state, user_input)
        final_response = coordinator.synthesize_results_enhanced(updated_state)
        coordinator_message = AIMessage(content=final_response, name="Synthesizer")
        return {
            "messages": state["messages"] + [coordinator_message],
            "tasks": updated_state["tasks"],
            "agent_outputs": updated_state["agent_outputs"],
            "active_agents": [name for name in agents.keys() if name != "Synthesizer"],
            "collaboration_events": updated_state.get("collaboration_events", []),
            "system_metrics": updated_state.get("system_metrics", {})
        }
    
    builder = StateGraph(AgentState)
    builder.add_node("EnhancedParallelExecution", parallel_execution_node)
    builder.add_edge("EnhancedParallelExecution", END)
    builder.set_entry_point("EnhancedParallelExecution")
    
    return builder.compile()

def run_enhanced_agent_system(user_input: str, use_all_agents: bool = True, single_agent: str = None):
    initial_message = HumanMessage(content=user_input)
    initial_state = {
        "messages": [initial_message],
        "tasks": {},
        "agent_outputs": {},
        "active_agents": list(agents.keys()) if use_all_agents else [single_agent],
        "thread_safe_state": ThreadSafeState(),
        "collaboration_events": [],
        "system_metrics": {}
    }
    
    app = create_enhanced_parallel_agent_graph()
    start_time = time.time()
    
    final_state = None
    synthesized_response = None
    for state in app.stream(initial_state, stream_mode="values"):
        final_state = state
        if "messages" in state and len(state["messages"]) > 1:
            final_message = state["messages"][-1]
            if isinstance(final_message, AIMessage):
                synthesized_response = final_message.content
                print(f"\n🎯 ENHANCED SYNTHESIZED RESPONSE")
                print(f"{'='*80}")
                print(f"{synthesized_response}")
                print(f"{'='*80}")

    # # Enhanced fallback with system metrics
    # if final_state and "agent_outputs" in final_state and final_state["agent_outputs"]:
    #     print(f"\n🎯 ENHANCED SYNTHESIZED RESPONSE")
    #     print(f"{'='*80}")
    #     if "messages" in final_state and len(final_state["messages"]) > 1:
    #         final_message = final_state["messages"][-1]
    #         if isinstance(final_message, AIMessage):
    #             print(f"{final_message.content}")
    #     print(f"{'='*80}")
        
    # Enhanced execution summary with system metrics
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n📈 ENHANCED EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Tasks processed: {len(final_state.get('tasks', {})) if final_state else 0}")
    print(f"👥 Agents involved: {len(final_state.get('agent_outputs', {})) if final_state else 0}")
    
    if final_state and "system_metrics" in final_state:
        metrics = final_state["system_metrics"]
        if "execution_time" in metrics:
            print(f"🚀 Parallel execution time: {metrics['execution_time']:.2f} seconds")
        if "collaboration_events" in final_state:
            print(f"🤝 Collaboration events: {len(final_state['collaboration_events'])}")
    
    print(f"✅ TRUE PARALLEL MULTI-AGENT SYSTEM COMPLETE")
    print(f"{'='*80}")

    return synthesized_response

def run_enhanced_agent_system_stream(user_input: str, use_all_agents: bool = True, single_agent: str = None):
    initial_message = HumanMessage(content=user_input)
    initial_state = {
        "messages": [initial_message],
        "tasks": {},
        "agent_outputs": {},
        "active_agents": list(agents.keys()) if use_all_agents else [single_agent],
        "thread_safe_state": ThreadSafeState(),
        "collaboration_events": [],
        "system_metrics": {}
    }
    coordinator = EnhancedCoordinator()
    state = initial_state
    # Distribute tasks
    tasks = coordinator.task_distributor.distribute_tasks_parallel(state, user_input)
    # Mark all agents as pending initially
    agent_progress = {name: {"status": "PENDING", "progress": 0.0} for name in agents.keys()}
    for agent_name in agent_progress:
        yield json.dumps({
            "type": "status",
            "agent": agent_name,
            "status": "PENDING",
            "progress": 0.0
        }) + "\n"
    # Start parallel execution (excluding Synthesizer)
    parallel_agents = [name for name in agents.keys() if name != "Synthesizer"]
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_agents)) as executor:
        future_to_task = {}
        for task in tasks:
            if task.agent_name != "Synthesizer":
                # Yield PROCESSING status as soon as agent starts
                yield json.dumps({
                    "type": "status",
                    "agent": task.agent_name,
                    "status": "PROCESSING",
                    "progress": 0.01
                }) + "\n"
                future = executor.submit(coordinator.task_distributor.task_executor.execute_task_parallel, state, task)
                future_to_task[future] = task
        # Monitor execution
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                completed_task = future.result()
                # Yield COMPLETED status
                yield json.dumps({
                    "type": "status",
                    "agent": completed_task.agent_name,
                    "status": "COMPLETED",
                    "progress": 1.0
                }) + "\n"
                # Update state for synthesis
                state["agent_outputs"][completed_task.agent_name] = completed_task.result
            except Exception as e:
                yield json.dumps({
                    "type": "status",
                    "agent": task.agent_name,
                    "status": "FAILED",
                    "progress": 1.0,
                    "error": str(e)
                }) + "\n"
    # Synthesize results
    yield json.dumps({
        "type": "status",
        "agent": "Synthesizer",
        "status": "PROCESSING",
        "progress": 0.01
    }) + "\n"
    import re
    # --- Begin streaming Synthesizer output ---
    synthesis_prompt = PromptTemplate.from_template(
        """You are the SYNTHESIZER, an expert at creating a single, unified, high-quality answer from multiple agent contributions.\n\nAgent Contributions (raw, do not mention agent names):\n{agent_outputs}\n\nUser's Request: {user_input}\n\nInstructions:\n- Carefully analyze, compare, and integrate all agent contributions.\n- Resolve any contradictions, fill in missing details, and ensure the answer is complete and accurate.\n- Do NOT mention or identify any agent names or roles.\n- Do NOT list or enumerate the contributions.\n- Write a single, seamless, well-structured answer that is clear, deep, and valuable for the user.\n- The result should read as if written by one top expert, not a group.\n- Always provide the best possible answer, maximizing insight and usefulness.\n\nUnified Synthesized Response:"""
    )
    agent_outputs_text = "\n".join([
        output for name, output in state["agent_outputs"].items() if name != "Synthesizer"
    ])
    user_input_val = state["messages"][-1].content
    synthesizer = agents["Synthesizer"]
    synthesis_runnable = synthesis_prompt | synthesizer.model | StrOutputParser()
    # Use streaming if available
    try:
        for chunk in synthesis_runnable.stream({
            "agent_outputs": agent_outputs_text,
            "user_input": user_input_val
        }):
            if chunk:
                yield json.dumps({
                    "type": "final_chunk",
                    "chunk": chunk
                }) + "\n"
    except Exception as e:
        # Fallback: yield error as a chunk
        yield json.dumps({
            "type": "final_chunk",
            "chunk": f"[SYNTHESIZER ERROR: {str(e)}]"
        }) + "\n"
    yield json.dumps({
        "type": "status",
        "agent": "Synthesizer",
        "status": "COMPLETED",
        "progress": 1.0
    }) + "\n"
    yield json.dumps({
        "type": "final_done"
    }) + "\n"

def run_single_agent_system(user_input: str, agent_name: str):
    if agent_name not in agents:
        print(f"❌ Error: Agent '{agent_name}' not found. Available agents: {', '.join(agents.keys())}")
        return
    run_enhanced_agent_system(user_input, use_all_agents=False, single_agent=agent_name)

def run_all_agents_system(user_input: str):
    run_enhanced_agent_system(user_input, use_all_agents=True)

# if __name__ == "__main__":
#     question = "Ironheart kapan rilis"
#     print("🚀 ENHANCED PARALLEL MULTI-AGENT SYSTEM")
#     print("=" * 80)
#     print("Features:")
#     print("✅ True parallel execution")
#     print("✅ Real-time collaboration")
#     print("✅ Intelligent task distribution")
#     print("✅ Load balancing")
#     print("✅ Performance optimization")
#     print("✅ Enhanced synthesis")
#     print("=" * 80)
#     run_all_agents_system(question)