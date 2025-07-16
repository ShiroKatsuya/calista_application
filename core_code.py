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

# Web tools setup
search = GoogleSerperAPIWrapper()

def google_search(query: str) -> str:
    try:
        return search.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

def web_browse(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text[:2000] + "..." if len(text) > 2000 else text
    except Exception as e:
        return f"Failed to browse {url}: {str(e)}"

web_tools = [
    Tool(name="google_search", description="Search Google for information", func=google_search),
    Tool(name="web_browse", description="Browse a web page", func=web_browse)
]

# # Models
# model_alice = ChatOllama(model="llama3.2:3b")
# model_bob = ChatOllama(model="llama3-2.3b:latest")
# model_charlie = ChatOllama(model="llama3.2:3b")
# model_coordinator = ChatOllama(model="llama3.2:3b")



# model_alice = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# model_bob = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# model_charlie = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# model_coordinator = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

model_alice = ChatOllama(model="llama3-2.3b:latest", temperature=0.7)
model_bob = ChatOllama(model="llama3-2.3b:latest", temperature=0.3)
model_charlie = ChatOllama(model="llama3.2:3b", temperature=0.9)
model_coordinator = ChatOllama(model="llama3.2:3b", temperature=0.5)
model_synthesizer = ChatOllama(model="llama3-2.3b:latest", temperature=0.4)  # New synthesis model

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

