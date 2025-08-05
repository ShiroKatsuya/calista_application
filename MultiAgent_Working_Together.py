from liblary import *
from core_code import *
import json


agents = {
    "Analisis Penyebab": Agent("Analisis Penyebab", model_Analisis_Penyebab, "General Knowledge Agent", 
                   ["ilmu pengetahuan", "teknologi", "humaniora", "seni", "bisnis", "riset", "analisis"], web_tools),
    "Analisis Dampak": Agent("Analisis Dampak", model_Analisis_Dampak, "Technical Implementation Agent", 
                 ["pengkodean", "pemrograman", "implementasi", "teknis", "arsitektur", "pengembangan"], web_tools),
    "Mengusulkan Solusi": Agent("Mengusulkan Solusi", model_Mengusulkan_Solusi, "Creative Problem Solver", 
                     ["kreativitas", "inovasi", "pemecahan masalah", "desain", "strategi", "brainstorming"], web_tools),
    "Synthesizer": Agent("Synthesizer", model_synthesizer, "Enhanced Response Synthesis Specialist", 
                        ["sintesis", "integrasi", "ringkasan", "kejelasan", "koherensi", "optimasi"], []),
}

def create_enhanced_collaborative_prompt(agent: Agent, thread_safe_state: ThreadSafeState, task_description: str, agent_role: str = None):
    other_outputs = thread_safe_state.get_other_agent_outputs(agent.name)
    collaboration_context = ""
    
    if other_outputs:
        collaboration_context = "\n\n🔗 KONTEKS KOLABORASI REAL-TIME:\n"
        for agent_name, output in other_outputs.items():
            collaboration_context += f"\n📝 Kontribusi {agent_name}:\n{output[:500]}...\n"
        collaboration_context += "\n💡 Kembangkan wawasan ini dan berikan perspektif yang saling melengkapi.\n"
    
    # Tambahkan instruksi kolaborasi khusus tugas
    task_lower = task_description.lower()
    if any(word in task_lower for word in ["kompleks", "sulit", "menantang", "complex", "difficult", "challenging"]):
        collaboration_context += "\n⚠️  Ini adalah tugas yang kompleks. Kolaborasikan secara intensif dengan agen lain.\n"
    
    # Tambahkan instruksi peran khusus jika ada
    role_instruction = f"\n\nFokus utama Anda: {agent_role}." if agent_role else ""
    
    return PromptTemplate.from_template(
        f"""Anda adalah {agent.name}, {agent.description}. Keahlian: {', '.join(agent.expertise)}.{role_instruction}

{collaboration_context}

Anda bekerja dalam lingkungan kolaboratif PARALEL SEBENARNYA. Agen lain bekerja secara bersamaan.
Pertimbangkan kontribusi mereka secara real-time dan berikan wawasan yang saling melengkapi.

ATURAN KOLABORASI PENTING:
1. Kembangkan wawasan dari agen lain
2. Isi kekosongan yang belum dibahas
3. Berikan perspektif unik sesuai keahlian Anda
4. Hindari duplikasi pekerjaan yang sudah dilakukan
5. Tingkatkan kualitas solusi secara keseluruhan

Riwayat Percakapan: {{chat_history}}
Permintaan Pengguna: {{input}}

Tanggapan Anda (fokus pada kolaborasi dan keahlian unik Anda, gunakan Bahasa Indonesia):"""
    )

class EnhancedParallelTaskExecutor:
    def __init__(self):
        self.thread_safe_state = ThreadSafeState()
        self.execution_metrics = {"total_tasks": 0, "successful_tasks": 0, "avg_execution_time": 0.0}
        # Define agent roles for parallel execution
        self.agent_roles = {
            "Analisis_Penyebab": "Menganalisis Penyebab (Analyzing Causes)",
            "Analisis_Dampak": "Menganalisis Dampak (Analyzing Impacts)",
            "Mengusulkan_Solusi": "Mengusulkan Solusi (Proposing Solutions)",
            "Synthesizer": "Menyintesis Laporan Komprehensif (Synthesizing Comprehensive Report)"
        }
    
    def execute_task_parallel(self, state: AgentState, task: AgentTask) -> AgentTask:
        agent = agents[task.agent_name]
        agent.is_busy = True
        self.thread_safe_state.increment_agent_load(agent.name)
        self.thread_safe_state.update_agent_status(agent.name, "working")
        
        start_time = time.time()
        
        try:
            self.thread_safe_state.update_task(task.id, status=TaskStatus.IN_PROGRESS)
            
            # Get agent role for this execution
            agent_role = self.agent_roles.get(agent.name, None)
            # Create enhanced collaborative prompt with role
            collaborative_prompt = create_enhanced_collaborative_prompt(agent, self.thread_safe_state, task.task_description, agent_role)
            
            # Use tool-enabled agent with enhanced collaboration
            # SYSTEM PROMPT: Bahasa Indonesia
            tool_prompt = ChatPromptTemplate.from_messages([
                ("system", f"Anda adalah {agent.name}, {agent.description}. Gunakan alat (tools) untuk memberikan jawaban yang komprehensif dan kolaborasi dengan agen lain secara real-time. Selalu gunakan Bahasa Indonesia dalam seluruh respons Anda. Fokus utama Anda: {agent_role if agent_role else ''}"),
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
            
            # Hidden <think> tags from supervisor respond for Mengusulkan_Solusi
            if agent.name == "Mengusulkan_Solusi":
                import re
                task.result = re.sub(r"<think>.*?</think>\n?", "", task.result, flags=re.DOTALL).strip()
                
            self.thread_safe_state.update_agent_output(agent.name, task.result)
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            # Update performance metrics
            execution_time = task.completed_at - start_time
            agent.update_performance(execution_time)
            self.execution_metrics["successful_tasks"] += 1
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Tugas gagal: {str(e)}"
            
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
        # Define agent roles for parallel execution
        self.agent_roles = {
            "Analisis_Penyebab": "Menganalisis Penyebab (Analyzing Causes)",
            "Analisis_Dampak": "Menganalisis Dampak (Analyzing Impacts)",
            "Mengusulkan_Solusi": "Mengusulkan Solusi (Proposing Solutions)",
            "Synthesizer": "Menyintesis Laporan Komprehensif (Synthesizing Comprehensive Report)"
        }
    
    def analyze_task_complexity(self, user_input: str) -> float:
        """Analisis kompleksitas tugas untuk distribusi optimal."""
        complexity_score = 0.0
        input_lower = user_input.lower()
        
        # Indikator kompleksitas (Bahasa Indonesia dan Inggris)
        complexity_keywords = [
            "kompleks", "sulit", "menantang", "komprehensif", "terperinci", "analisis",
            "complex", "difficult", "challenging", "comprehensive", "detailed", "analysis"
        ]
        complexity_score += sum(0.2 for keyword in complexity_keywords if keyword in input_lower)
        
        # Indikator panjang dan struktur
        if len(user_input.split()) > 20:
            complexity_score += 0.3
        if user_input.count('?') > 1:
            complexity_score += 0.2
            
        return min(complexity_score, 1.0)
    
    def select_optimal_agents(self, user_input: str, complexity_score: float) -> list:
        """Pilih agen optimal berdasarkan kebutuhan tugas dan kapabilitas agen."""
        available_agents = list(agents.keys())
        
        # Untuk demonstrasi, gunakan semua agen agar eksekusi paralel terlihat
        print(f"🔍 Skor kompleksitas tugas: {complexity_score:.2f}")
        print(f"🎯 Menggunakan semua agen untuk eksekusi paralel maksimal")
        return available_agents
  
    
    def distribute_tasks_parallel(self, state: AgentState, user_input: str) -> list:
        complexity_score = self.analyze_task_complexity(user_input)
        optimal_agents = self.select_optimal_agents(user_input, complexity_score)
        
        tasks = []
        timestamp = int(time.time())
        
        for agent_name in optimal_agents:
            # Assign specific role to each agent
            agent_role = self.agent_roles.get(agent_name, None)
            task = AgentTask(
                id=f"task_{len(state['tasks'])}_{agent_name}_{timestamp}",
                agent_name=agent_name,
                task_description=f"{user_input}\n\nFokus utama Anda: {agent_role}.",
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
        
        print(f"\n🚀 EKSEKUSI PARALEL DIMULAI")
        print(f"📋 Tugas yang akan dieksekusi: {len(tasks)}")
        for task in tasks:
            print(f"   - {task.agent_name}: {task.task_description[:50]}...")
        
        # Tambahkan pelacakan event kolaborasi
        state["collaboration_events"] = []
        state["system_metrics"] = {"start_time": time.time()}
        
        for task in tasks:
            state["tasks"][task.id] = task
            self.thread_safe_state.add_task(task)
        
        # Eksekusi paralel (kecuali Synthesizer)
        parallel_agents = [name for name in agents.keys() if name != "Synthesizer"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_agents)) as executor:
            future_to_task = {
                executor.submit(self.task_distributor.task_executor.execute_task_parallel, state, task): task
                for task in tasks if task.agent_name != "Synthesizer"
            }
            
            print(f"⚡ Mengeksekusi {len(future_to_task)} tugas secara paralel...")
            
            # Monitoring eksekusi dengan kolaborasi real-time
            completed_tasks = []
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    completed_task = future.result()
                    completed_tasks.append(completed_task)
                    
                    if completed_task.status == TaskStatus.COMPLETED:
                        state["agent_outputs"][completed_task.agent_name] = completed_task.result
                        print(f"✅ {completed_task.agent_name} selesai: {completed_task.result}")
                        # Catat event kolaborasi
                        state["collaboration_events"].append({
                            "agent": completed_task.agent_name,
                            "timestamp": time.time(),
                            "status": "completed"
                        })
                except Exception as e:
                    print(f"❌ Tugas {task.id} gagal: {str(e)}")
        
        print(f"🎉 Eksekusi paralel selesai. {len(completed_tasks)} tugas selesai.")
        
        # Update state akhir
        state["agent_outputs"].update(self.thread_safe_state.get_agent_outputs())
        state["system_metrics"]["end_time"] = time.time()
        state["system_metrics"]["execution_time"] = state["system_metrics"]["end_time"] - state["system_metrics"]["start_time"]

        return state

    def synthesize_results_enhanced(self, state: AgentState) -> str:
        if not state["agent_outputs"]:
            return "Tidak ada output agen untuk disintesis."

        synthesis_start = time.time()
        # Satu prompt sintesis  untuk agen Synthesizer (Bahasa Indonesia)
        agent_outputs_text = "\n".join([
            output for name, output in state["agent_outputs"].items() if name != "Synthesizer"
        ])
        synthesis_prompt = PromptTemplate.from_template(
            """Anda adalah SYNTHESIZER, seorang ahli dalam membuat satu jawaban , berkualitas tinggi dari berbagai kontribusi agen.

Kontribusi Agen (mentah, jangan sebut nama agen):
{agent_outputs}

Permintaan Pengguna: {user_input}

Instruksi:
- Analisis, bandingkan, dan integrasikan semua kontribusi agen secara cermat.
- JANGAN sebut atau identifikasi nama/role agen manapun.
- JANGAN membuat daftar kontribusi.
- Tulis satu jawaban, terstruktur baik, jelas, mendalam, dan bernilai bagi pengguna.
- Hasil akhir harus terasa seperti ditulis satu pakar terbaik, bukan tim.
- Selalu berikan jawaban terbaik, maksimalkan wawasan dan manfaat.

Jawaban Sintesis :"""
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
                print(f"\n🎯 JAWABAN SINTESIS ")
                print(f"{'='*80}")
                print(f"{synthesized_response}")
                print(f"{'='*80}")

    # Ringkasan eksekusi dengan metrik sistem
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n📈 RINGKASAN EKSEKUSI TERENHANCE")
    print(f"{'='*80}")
    print(f"⏱️  Total waktu eksekusi: {execution_time:.2f} detik")
    print(f"📊 Tugas yang diproses: {len(final_state.get('tasks', {})) if final_state else 0}")
    print(f"👥 Agen yang terlibat: {len(final_state.get('agent_outputs', {})) if final_state else 0}")
    
    if final_state and "system_metrics" in final_state:
        metrics = final_state["system_metrics"]
        if "execution_time" in metrics:
            print(f"🚀 Waktu eksekusi paralel: {metrics['execution_time']:.2f} detik")
        if "collaboration_events" in final_state:
            print(f"🤝 Event kolaborasi: {len(final_state['collaboration_events'])}")
    
    print(f"✅ SISTEM MULTI-AGENT PARALEL SEBENARNYA SELESAI")
    print(f"{'='*80}")

    return synthesized_response

def run_enhanced_agent_system_stream(user_input: str, messages=None, use_all_agents: bool = True, single_agent: str = None):
    # If messages are provided, use them; otherwise, start with the new message
    if messages is not None and len(messages) > 0:
        initial_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Try to reconstruct HumanMessage or AIMessage
                if msg.get('type') == 'HumanMessage' or msg.get('sender') == 'user':
                    initial_messages.append(HumanMessage(content=msg['content']))
                elif msg.get('type') == 'AIMessage' or msg.get('sender') == 'ai' or msg.get('type') == 'final_result':
                    initial_messages.append(AIMessage(content=msg['content'], name=msg.get('sender', 'ai')))
                else:
                    # Fallback to HumanMessage
                    initial_messages.append(HumanMessage(content=msg['content']))
            else:
                initial_messages.append(msg)
        # Ensure the last message is the new user input
        if initial_messages[-1].content != user_input:
            initial_messages.append(HumanMessage(content=user_input))
    else:
        initial_messages = [HumanMessage(content=user_input)]

    initial_state = {
        "messages": initial_messages,
        "tasks": {},
        "agent_outputs": {},
        "active_agents": list(agents.keys()) if use_all_agents else [single_agent],
        "thread_safe_state": ThreadSafeState(),
        "collaboration_events": [],
        "system_metrics": {}
    }
    coordinator = EnhancedCoordinator()
    state = initial_state
    # Distribusi tugas
    tasks = coordinator.task_distributor.distribute_tasks_parallel(state, user_input)
    # Tandai semua agen sebagai pending di awal
    agent_progress = {name: {"status": "PENDING", "progress": 0.0} for name in agents.keys()}
    for agent_name in agent_progress:
        yield json.dumps({
            "type": "status",
            "agent": agent_name,
            "status": "PENDING",
            "progress": 0.0
        }) + "\n"
    # Eksekusi paralel (kecuali Synthesizer)
    parallel_agents = [name for name in agents.keys() if name != "Synthesizer"]
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_agents)) as executor:
        future_to_task = {}
        for task in tasks:
            if task.agent_name != "Synthesizer":
                # Status PROCESSING saat agen mulai
                yield json.dumps({
                    "type": "status",
                    "agent": task.agent_name,
                    "status": "PROCESSING",
                    "progress": 0.01
                }) + "\n"
                future = executor.submit(coordinator.task_distributor.task_executor.execute_task_parallel, state, task)
                future_to_task[future] = task
        # Monitoring eksekusi
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                completed_task = future.result()
                # Status COMPLETED
                yield json.dumps({
                    "type": "status",
                    "agent": completed_task.agent_name,
                    "status": "COMPLETED",
                    "progress": 1.0
                }) + "\n"
                # Update state untuk sintesis
                state["agent_outputs"][completed_task.agent_name] = completed_task.result
            except Exception as e:
                yield json.dumps({
                    "type": "status",
                    "agent": task.agent_name,
                    "status": "FAILED",
                    "progress": 1.0,
                    "error": str(e)
                }) + "\n"
    # Sintesis hasil
    yield json.dumps({
        "type": "status",
        "agent": "Synthesizer",
        "status": "PROCESSING",
        "progress": 0.01
    }) + "\n"
    import re
    # --- Streaming output Synthesizer (Bahasa Indonesia) ---
    synthesis_prompt = PromptTemplate.from_template(
       
        """Anda adalah SYNTHESIZER, seorang ahli dalam membuat satu jawaban , berkualitas tinggi dari berbagai kontribusi agen.

            Kontribusi Agen (mentah, jangan sebut nama agen):
            {agent_outputs}

            Permintaan Pengguna: {user_input}

            Instruksi:
            - Analisis, bandingkan, dan integrasikan semua kontribusi agen secara cermat.
            - JANGAN sebut atau identifikasi nama/role agen manapun.
            - JANGAN membuat daftar kontribusi.
            - Tulis satu jawaban , terstruktur baik, jelas, mendalam, dan bernilai bagi pengguna.
            - Hasil akhir harus terasa seperti ditulis satu pakar terbaik, bukan tim.
            - Selalu berikan jawaban terbaik, maksimalkan wawasan dan manfaat.

            Jawaban Sintesis :"""

    )
    agent_outputs_text = "\n".join([
        output for name, output in state["agent_outputs"].items() if name != "Synthesizer"
    ])
    user_input_val = state["messages"][-1].content
    synthesizer = agents["Synthesizer"]
    synthesis_runnable = synthesis_prompt | synthesizer.model | StrOutputParser()
    # Streaming jika tersedia
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
        # Fallback: error sebagai chunk
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
        print(f"❌ Error: Agen '{agent_name}' tidak ditemukan. Agen yang tersedia: {', '.join(agents.keys())}")
        return
    run_enhanced_agent_system(user_input, use_all_agents=False, single_agent=agent_name)

def run_all_agents_system(user_input: str):
    run_enhanced_agent_system(user_input, use_all_agents=True)

if __name__ == "__main__":
    question = "Ironheart kapan rilis"
    print("🚀 SISTEM MULTI-AGENT PARALEL TERENHANCE")
    print("=" * 80)
    print("Fitur:")
    print("✅ Eksekusi paralel sesungguhnya")
    print("✅ Kolaborasi real-time")
    print("✅ Distribusi tugas cerdas")
    print("✅ Load balancing")
    print("✅ Optimasi performa")
    print("✅ Sintesis jawaban ")
    print("=" * 80)
    run_all_agents_system(question)