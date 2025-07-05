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

ENVS: Dict[str, str | None] = {}
DB = False

if dotenv.find_dotenv():
    dotenv.load_dotenv()
    DOTENV_PATH = dotenv.find_dotenv()
    ENVS = dotenv.dotenv_values()
    # add-feature: send the list of env keys to front end client. (ENVS dictionary's keys)
else:
    # add-feature: tell frontend client that no env vars are set. type: WARN
    pass

DOTENV_PATH = dotenv.find_dotenv()
PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

async def set_api_key(provider: str, key: str):
    """Set API key in environment and .env file"""
    # env_var = PROVIDER_ENV_MAP.get(provider.lower())
    # if not env_var:
    #     return False
    
    # Set in current environment
    os.environ[provider] = key
    
    # Update .env file if exists
    if DOTENV_PATH:
        dotenv.set_key(DOTENV_PATH, provider, key)
    
    return True

# Add this function to get current API keys
def get_api_keys():
    """Get current API keys from environment"""
    dotenv.load_dotenv()
    ENVS = dotenv.dotenv_values()
    return {k: v for k, v in dict(ENVS).items()}


if os.getenv("DATABASE_URL") is not None:
    DATABASE_URL = os.getenv("DATABASE_URL")
    DB_CONTAINER_NAME = "chat-app-db"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "chat_db")
    DB = True
    os.makedirs(os.path.join(DATA_DIR, "user_files"), exist_ok=True)
else:
    # add-feature: tell frontend client that DB env var is not set and chats will not be persistent. type: WARN
    pass

def start_database():
    """Start the PostgreSQL Docker container if not running"""
    try:
        subprocess.run(["docker", "--version"], check=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if sys.platform == "win32":
            script_path = os.path.join(DATA_DIR, "docker-db.bat")
            if os.path.exists(script_path):
                subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", script_path], check=True, shell=True)
                print("✅ PostgreSQL container started successfully")
            else:
                print(f"⚠ docker-db.ps1 not found at {script_path}. Using existing DB connection.")
        else:
            script_path = os.path.join(DATA_DIR, "docker-db.sh")
            if os.path.exists(script_path):
                subprocess.run(["bash", script_path], check=True)
                print("✅ PostgreSQL container started successfully")
            else:
                print(f"⚠ docker-db.sh not found at {script_path}. Using existing DB connection.")
                
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Docker error: {str(e)}. Using existing database connection.")

async def stop_database():
    """Stop the PostgreSQL Docker container if running"""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "stop", DB_CONTAINER_NAME,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            print(f"✅ Stopped container {DB_CONTAINER_NAME}")
        else:
            error_msg = stderr.decode().strip()
            if "No such container" in error_msg:
                print(f"⚠️ Container {DB_CONTAINER_NAME} not found")
            else:
                print(f"⚠️ Could not stop container: {error_msg}")
    except FileNotFoundError:
        print("❌ Docker not installed, skipping container stop")

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_database()
    await asyncio.sleep(2)
    
    app.state.db_pool = await asyncpg.create_pool(
        DATABASE_URL, 
        min_size=5, 
        max_size=20,
        command_timeout=60
    )
    
    await init_db(app.state.db_pool)
    print("✅ Database connected and initialized")
    
    yield
    
    await app.state.db_pool.close()
    print("❌ Database connection closed")
    await stop_database()

if DB:
    app = FastAPI(lifespan=lifespan)
else:
    app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, 'Session'] = {}
# UPLOAD_DIR = os.path.join(DATA_DIR, "user_files")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent = None
        self.loaded_files: Set[str] = set()
        self.active_generation = None
        self.provider = None
        self.model = None
        self.memory = AgentMemory()
        self.history_loaded: bool = False
        
    async def initialize_agent(self, provider: str, model: str):
        """Initialize agent after provider is known"""
        
        if self.agent and self.provider == provider:
            return  
        
        self.provider = provider
        self.model = model  # Store model
        client = self.setup_client(self.provider)
        
        if DB:
            async with app.state.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO chat_sessions (id, provider, model, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE
                    SET provider = EXCLUDED.provider,
                        model = EXCLUDED.model,
                        updated_at = EXCLUDED.updated_at
                ''', self.session_id, self.provider, self.model, datetime.now(), datetime.now())
                
        
        if DB and not self.history_loaded:
            await self.load_history()
            self.history_loaded = True
        
        system_prompt_geospatial = SystemPromptGenerator(
            background=[
                "You are an expert in ONLY geospatial reasoning and you are a geospatial DB manager.",
                "Your tone is highly professional and technical."
            ],
            steps=[
                "Make sure the prompt is strictly related to geospatial concepts.",
                "Understand the user's intent and and goal step-by-step.",
            ],
            output_instructions=[
                "You will only respond to users' prompts if they are related to geospatial tasks; otherwise you throw a small cute tantrum and respond with you only deal woth geospatial tasks.",
            ],
        )

        if self.memory.get_message_count() == 0:
            initial_message = "Hello! How can I assist you today?"
            initial_schema = BaseAgentOutputSchema(chat_message=initial_message)
            self.memory.add_message("assistant", content=initial_schema)
            if DB:
                await self.save_message("assistant", initial_message)
        
        self.agent = BaseAgent(
            config=BaseAgentConfig(
                client=client,
                model=self.model,
                memory=self.memory,
                # system_prompt_generator=system_prompt_geospatial,
                model_api_parameters={"max_tokens": 2048},
            )
        )
        print(f"Agent initialized with provider: {provider}, model: {model}")
    
    async def load_history(self):
        """Load chat history from database"""
        async with app.state.db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT role, content 
                FROM chat_messages 
                WHERE session_id = $1 
                ORDER BY created_at
            ''', self.session_id)
            
            for row in rows:
                content = row['content']
                if row['role'] == "user":
                    input_schema = BaseAgentInputSchema(chat_message=content)
                    self.memory.add_message("user", content=input_schema)
                elif row['role'] == "assistant":
                    output_schema = BaseAgentOutputSchema(chat_message=content)
                    self.memory.add_message("assistant", content=output_schema)
        
        print(f"Loaded {len(rows)} messages for session {self.session_id}")
    
    async def save_message(self, role: str, content: str):
        """Save message to database"""
        async with app.state.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES ($1, $2, $3, $4)
            ''', self.session_id, role, content, datetime.now())
            
    async def truncate_history(self, reset_index: int):
        """Truncate history to the specified index"""
        # First, get the ordered message IDs
        async with app.state.db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT id 
                FROM chat_messages 
                WHERE session_id = $1 
                ORDER BY created_at
            ''', self.session_id)
            
            if len(rows) <= reset_index + 1:
                print(f"No messages to truncate for session {self.session_id}")
                return
            
            # Get the ID to delete from
            delete_from_id = rows[reset_index + 1]['id']
            
            # Delete messages after the specified index
            await conn.execute('''
                DELETE FROM chat_messages
                WHERE session_id = $1 AND id >= $2
            ''', self.session_id, delete_from_id)
            print(f"Truncated history for session {self.session_id} at index {reset_index}")
        
        # Reload history to update agent memory
        self.memory = AgentMemory()
        await self.load_history()
        
        # Reinitialize agent with updated history
        if self.provider and self.model:
            client = self.setup_client(self.provider, self.model)
            self.agent = BaseAgent(
                config=BaseAgentConfig(
                    client=client,
                    model=self.model,
                    memory=self.memory,
                    model_api_parameters={"max_tokens": 2048},
                )
            )
    
    def setup_client(self, provider: str):
        provider = provider.lower()
        
        if PROVIDER_ENV_MAP.get(provider.lower()):
            if provider == "openai":
                import openai
                from openai import AsyncOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                    
                client = instructor.from_openai(AsyncOpenAI(api_key=api_key))
                return client

            elif provider == "anthropic":
                import anthropic
                from anthropic import AsyncAnthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                    
                client = instructor.from_anthropic(AsyncAnthropic(api_key=api_key))
                return client

            elif provider == "groq":
                import groq
                from groq import AsyncGroq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY environment variable not set")
                    
                client = instructor.from_groq(
                    AsyncGroq(api_key=api_key),
                    mode=instructor.Mode.JSON
                )
                return client

            elif provider == "gemini":
                from openai import OpenAI, AsyncOpenAI
                # import google.generativeai as genai
                # genai.configure(api_key=api_key)
                
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")

                client = instructor.from_openai(
                    AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
                    mode=instructor.Mode.JSON,
                )
                return client
        else:
            # add-feature: tell frontend client that no env var set for the provider: WARN
            raise ValueError(f"Provider env var not set: {provider}")
    
    async def stream_response(self, input_text: str) -> AsyncGenerator[str, None]:
        """Stream response only if agent is initialized"""
        if not self.agent:
            yield "[ERROR] Agent not initialized. Please reconnect."
            return

        input_schema = BaseAgentInputSchema(chat_message=input_text)
        # self.memory.add_message("user", content=input_schema)
        
        try:
            full_response: str = ""
            async for partial_response in self.agent.run_async(input_schema):

                if not hasattr(partial_response, "chat_message") or not partial_response.chat_message:
                    continue

                new_text = partial_response.chat_message[len(full_response):]
                
                if new_text:
                    full_response = partial_response.chat_message
                    yield new_text

            if full_response:
                output_schema = BaseAgentOutputSchema(chat_message=full_response)
                self.memory.add_message("assistant", content=output_schema)
                if DB:
                    await self.save_message("assistant", full_response)
                
        except Exception as e:
            traceback.print_exc()
            yield f"[ERROR] {str(e)}"

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    active_sessions: Dict[str, Session] = {}

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "sessionId": None,
                    "token": "[ERROR] Invalid JSON message"
                }))
                continue

            msg_type = message.get("type")
            session_id = message.get("sessionId")

            # Add API key management handlers
            if msg_type == "get_api_keys":
                api_keys = get_api_keys()
                await websocket.send_text(json.dumps({
                    "type": "api_keys",
                    "keys": api_keys
                }))
                continue
                
            elif msg_type == "set_api_key":
                provider = message.get("provider")
                key = message.get("key")
                
                if not provider or not key:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Missing provider or key"
                    }))
                    continue
                    
                success = await set_api_key(provider, key)
                if success:
                    # Send updated keys
                    api_keys = get_api_keys()
                    await websocket.send_text(json.dumps({
                        "type": "api_keys",
                        "keys": api_keys
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Invalid provider: {provider}"
                    }))
                continue

            if msg_type == "init":
                provider = message.get("provider")
                model = message.get("model")
                # print(f"Received init for session {session_id} with provider {provider} and model {model}") 
                if not session_id or not provider or not model:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] 'sessionId' and 'provider' required in init"
                    }))
                    continue
                # Create or reuse Session object
                if session_id not in sessions:
                    sessions[session_id] = Session(session_id)
                    print(f"Created new session: {session_id}")

                session = sessions[session_id]
                
                try:
                    await session.initialize_agent(provider, model)
                    active_sessions[session_id] = session
                    print(f"Initialized agent for session {session_id} with provider {provider} and model {model}")

                    if  session.memory.get_message_count() > 0 and session.memory.history[-1].role == "assistant":
                        await websocket.send_text(json.dumps({
                            "sessionId": session_id,
                            "token": "[[END]]"
                        }))

                except Exception as e:
                    error_msg = f"Agent initialization failed: {str(e)}"
                    print(error_msg)
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": f"[ERROR] {error_msg}"
                    }))
                    continue

            elif msg_type == "message":
                if not session_id or session_id not in active_sessions:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] Session not initialized or invalid"
                    }))
                    continue

                text = message.get("text")
                if not text:
                    continue

                session = active_sessions[session_id]
                if DB:
                    await session.save_message("user", text)

                async def stream():
                    try:
                        async for token in session.stream_response(text):
                            await websocket.send_text(json.dumps({
                                "sessionId": session_id,
                                "token": token
                            }))
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            "sessionId": session_id,
                            "token": f"[ERROR] {str(e)}"
                        }))
                    finally:
                        await websocket.send_text(json.dumps({
                            "sessionId": session_id,
                            "token": "[[END]]"
                        }))

                await stream()
                
            elif msg_type == "reset":
                if not session_id or session_id not in active_sessions:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] Session not initialized or invalid"
                    }))
                    continue
                
                reset_index = message.get("resetToIndex")
                if reset_index is None:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] Missing reset index"
                    }))
                    continue
                
                try:
                    session = active_sessions[session_id]
                    if DB:
                        await session.truncate_history(reset_index)
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "type": "reset_ack"
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": f"[ERROR] Reset failed: {str(e)}"
                    }))

            elif msg_type == "stop":
                if not session_id or session_id not in active_sessions:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] Session not initialized or invalid"
                    }))
                    continue

                await websocket.send_text(json.dumps({
                    "sessionId": session_id,
                    "token": "[[END]]"
                }))

            else:
                await websocket.send_text(json.dumps({
                    "sessionId": session_id,
                    "token": "[ERROR] Unknown message type"
                }))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        traceback.print_exc()
        await websocket.close(code=1011)

async def stream_to_websocket(session: Session, websocket: WebSocket, input_text: str):
    try:
        async for token in session.stream_response(input_text):
            await websocket.send_text(token)
    except asyncio.CancelledError:
        print("Generation cancelled")
        pass
    except Exception as e:
        error_msg = f"Stream error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        await websocket.send_text(f"[ERROR] {error_msg}")
    finally:
        await websocket.send_text("[[END]]")

async def init_db(pool):
    try:
        async with pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New Chat',
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session 
                ON chat_messages(session_id)
            ''')
            print("✅ Database tables verified")
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reload_dirs = ["."]
    
    if DB:
        chat_db_relative = os.path.relpath(DATA_DIR, current_dir)
        reload_excludes: List[str] = [os.path.join(chat_db_relative, '*')]
    else:
        reload_excludes = []
        
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=4580,
        ws="websockets",
        reload=True,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes
    )