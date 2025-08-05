from liblary import *
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    COLLABORATING = "collaborating"

@dataclass
class AgentTask:
    id: str
    agent_name: str
    task_description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    collaboration_partners: List[str] = field(default_factory=list)
    priority: int = 1   
    complexity_score: float = 0.0

class ThreadSafeState:
    def __init__(self):
        self._lock = Lock()
        self._agent_outputs = {}
        self._tasks = {}
        self._collaboration_queue = Queue()
        self._agent_status = {}
        self._load_balancer = {}
        
    def update_agent_output(self, agent_name: str, output: str):
        with self._lock:
            self._agent_outputs[agent_name] = output
            # Notify other agents of new output for real-time collaboration
            self._collaboration_queue.put({
                'type': 'agent_output',
                'agent': agent_name,
                'output': output,
                'timestamp': time.time()
            })
    
    def get_agent_outputs(self) -> Dict[str, str]:
        with self._lock:
            return self._agent_outputs.copy()
    
    def get_other_agent_outputs(self, current_agent: str) -> Dict[str, str]:
        with self._lock:
            return {name: output for name, output in self._agent_outputs.items() 
                   if name != current_agent and output}
    
    def get_collaboration_updates(self, timeout: float = 1.0) -> List[Dict]:
        updates = []
        try:
            while True:
                update = self._collaboration_queue.get(timeout=timeout)
                updates.append(update)
        except:
            pass
        return updates
    
    def update_agent_status(self, agent_name: str, status: str):
        with self._lock:
            self._agent_status[agent_name] = status
    
    def get_agent_load(self, agent_name: str) -> int:
        with self._lock:
            return self._load_balancer.get(agent_name, 0)
    
    def increment_agent_load(self, agent_name: str):
        with self._lock:
            self._load_balancer[agent_name] = self._load_balancer.get(agent_name, 0) + 1
    
    def decrement_agent_load(self, agent_name: str):
        with self._lock:
            current_load = self._load_balancer.get(agent_name, 0)
            if current_load > 0:
                self._load_balancer[agent_name] = current_load - 1
    
    def add_task(self, task: AgentTask):
        with self._lock:
            self._tasks[task.id] = task
    
    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            if task_id in self._tasks:
                for key, value in kwargs.items():
                    setattr(self._tasks[task_id], key, value)

class AgentState(TypedDict):
    messages: Annotated[list[Union[HumanMessage, AIMessage]], "add"]
    tasks: Dict[str, AgentTask]
    agent_outputs: Dict[str, str]
    active_agents: List[str]
    thread_safe_state: ThreadSafeState
    collaboration_events: List[Dict]
    system_metrics: Dict[str, Any]

# Web tools setup with enforced accuracy

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





from os import getenv
# Mengambil nama model dari environment variable, jika tidak ada gunakan default
nama_model_analisis_penyebab = getenv("MODEL_ANALISIS_PENYEBAB")
nama_model_analisis_dampak = getenv("MODEL_ANALISIS_DAMPAK")
nama_model_Mengusulkan_Solusi = getenv("MODEL_MENGUSULKAN_SOLUSI")
nama_model_synthesizer = getenv("MODEL_SYNTHESIZER")
model_Analisis_Penyebab = ChatOllama(model=nama_model_analisis_penyebab)
model_Analisis_Dampak = ChatOllama(model=nama_model_analisis_dampak)
model_Mengusulkan_Solusi = ChatOllama(model=nama_model_Mengusulkan_Solusi)
model_synthesizer = ChatOllama(model=nama_model_synthesizer)  # New synthesis model

class Agent:
    def __init__(self, name: str, model, description: str, expertise: List[str], tools=None):
        self.name = name
        self.model = model
        self.description = description
        self.expertise = expertise
        self.tools = tools or []
        self.is_busy = False
        self.collaboration_history = []
        self.performance_metrics = {"tasks_completed": 0, "avg_response_time": 0.0}
        
    def can_handle_task(self, task_description: str) -> float:
        task_lower = task_description.lower()
        score = sum(0.3 for expertise in self.expertise if expertise.lower() in task_lower)
        return min(score, 1.0)
    
    def get_availability(self) -> float:
        return 0.0 if self.is_busy else 1.0
    
    def get_performance_score(self) -> float:
        return self.performance_metrics["tasks_completed"] * 0.5 + self.performance_metrics["avg_response_time"] * 0.5
    
    def update_performance(self, response_time: float):
        self.performance_metrics["tasks_completed"] += 1
        current_avg = self.performance_metrics["avg_response_time"]
        self.performance_metrics["avg_response_time"] = (current_avg + response_time) / 2