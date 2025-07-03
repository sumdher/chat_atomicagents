import os
import subprocess
import sys
import uuid
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import instructor
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema, SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from typing import Dict, List, Set, AsyncGenerator, Optional
import aiofiles
import time
import json
import dotenv
import traceback
import asyncpg
from datetime import datetime
from contextlib import asynccontextmanager

dotenv.load_dotenv()
def setup_client(provider):
    if provider == "1" or provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=api_key))
        model = "gpt-4o-mini"
    elif provider == "2" or provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = instructor.from_anthropic(Anthropic(api_key=api_key))
        model = "claude-3-5-haiku-20241022"
    elif provider == "3" or provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        model = "mixtral-8x7b-32768"
    elif provider == "4" or provider == "ollama":
        from openai import OpenAI as OllamaClient

        client = instructor.from_openai(
            OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama"), mode=instructor.Mode.JSON
        )
        model = "llama3"
    elif provider == "5" or provider == "gemini":
        from openai import OpenAI

        api_key = os.getenv("GEMINI_API_KEY")
        client = instructor.from_openai(
            OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            mode=instructor.Mode.JSON,
        )
        model = "gemini-2.0-flash-exp"
    elif provider == "6" or provider == "openrouter":
        from openai import OpenAI as OpenRouterClient

        api_key = os.getenv("OPENROUTER_API_KEY")
        client = instructor.from_openai(OpenRouterClient(base_url="https://openrouter.ai/api/v1", api_key=api_key))
        model = "mistral/ministral-8b"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client, model