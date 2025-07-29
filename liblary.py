#! /usr/bin/env python
import concurrent.futures
import time
import threading
from typing import Annotated, TypedDict, Union, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Event
from queue import Queue
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

