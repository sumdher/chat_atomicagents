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

DATABASE_URL=os.getenv("DATABASE_URL")
DB_CONTAINER_NAME = "chat-app-db"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "chat_db")

# os.makedirs(os.path.join(DATA_DIR, "postgres-data"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "user_files"), exist_ok=True)

def start_database():
    """Start the PostgreSQL Docker container if not running"""
    try:
        # Check if Docker is installed
        subprocess.run(["docker", "--version"], check=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        script_path = os.path.join(DATA_DIR, "docker-db.sh")
        if os.path.exists(script_path):
            subprocess.run(["bash", script_path], check=True)
            print("✅ PostgreSQL container started successfully")
        else:
            print(f"⚠️ docker-db.sh script not found at {script_path}. Using existing DB connection.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Docker error: {str(e)}. Using existing database connection.")

async def stop_database():
    """Stop the PostgreSQL Docker container if running"""
    try:
        # Check if container exists and is running
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
    
    # Create database connection pool
    app.state.db_pool = await asyncpg.create_pool(
        DATABASE_URL, 
        min_size=5, 
        max_size=20,
        command_timeout=60
    )
    
    # Initialize database tables
    await init_db(app.state.db_pool)
    print("✅ Database connected and initialized")
    
    yield
    
    # Shutdown
    await app.state.db_pool.close()
    print("❌ Database connection closed")
    await stop_database()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, 'Session'] = {}
UPLOAD_DIR = os.path.join(DATA_DIR, "user_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)
file_map: Dict[str, str] = {}  # {file_key: original_name}

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
        
    async def initialize_agent(self, provider: str):
        """Initialize agent after provider is known"""
        
        if self.agent and self.provider == provider:
            return  
        
        self.provider = provider
        client, model = self.setup_client(provider)
        self.model = model
        
        # Create or update session in database
        async with app.state.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO chat_sessions (id, provider, model, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE
                SET updated_at = EXCLUDED.updated_at
            ''', self.session_id, provider, model, datetime.utcnow(), datetime.utcnow())
        
        # Load chat history from database
        if not self.history_loaded:
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

        # Add initial message if no history exists
        if self.memory.get_message_count() == 0:
            initial_message = "Hello! How can I assist you today?"
            initial_schema = BaseAgentOutputSchema(chat_message=initial_message)
            self.memory.add_message("assistant", content=initial_schema)
            await self.save_message("assistant", initial_message)
        
        self.agent = BaseAgent(
            config=BaseAgentConfig(
                client=client,
                model=model,
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
            ''', self.session_id, role, content, datetime.utcnow())
    
    def setup_client(self, provider):
        provider = provider.lower()
        
        if provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            client = instructor.from_openai(openai.AsyncOpenAI(api_key=api_key))
            model = "gpt-4o-mini"
            return client, model

        elif provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                
            client = instructor.from_anthropic(anthropic.AsyncAnthropic(api_key=api_key))
            model = "claude-3-haiku-20240307"
            return client, model

        elif provider == "groq":
            import groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
                
            client = instructor.from_groq(
                groq.AsyncGroq(api_key=api_key),
                mode=instructor.Mode.JSON
            )
            model = "mixtral-8x7b-32768"
            return client, model

        elif provider == "ollama":
            import openai
            client = instructor.from_openai(
                openai.AsyncOpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama"
                ),
                mode=instructor.Mode.JSON
            )
            model = "llama3"
            return client, model

        elif provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
                
            genai.configure(api_key=api_key)
            client = genai.AsyncClient()
            model = "gemini-1.5-flash-latest"
            return client, model
            
        elif provider == "openrouter":
            import openai
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
                
            client = instructor.from_openai(
                openai.AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key
                )
            )
            model = "mistralai/mistral-7b-instruct:free"
            return client, model

        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
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
                await self.save_message("assistant", full_response)
                
        except Exception as e:
            traceback.print_exc()
            yield f"[ERROR] {str(e)}"

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    # Keep track of sessionId -> Session instances active on this WS connection
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

            if msg_type == "init":
                provider = message.get("provider")
                if not session_id or not provider:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] 'sessionId' and 'provider' required in init"
                    }))
                    continue
                # Create or reuse Session object
                if session_id not in sessions:
                    sessions[session_id] = Session(session_id)
                    print(f"Created new session: {session_id}")

                is_new_session = session_id not in sessions
                session = sessions[session_id]
                
                try:
                    await session.initialize_agent(provider)
                    active_sessions[session_id] = session
                    print(f"Initialized agent for session {session_id} with provider {provider}")

                    if  session.memory.get_message_count() > 0 and session.memory.history[-1].role == "assistant":
                        last_message = session.memory.history[-1].content.chat_message
                        # await websocket.send_text(json.dumps({
                        #     "sessionId": session_id,
                        #     "token": last_message
                        # }))
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

                # Run the stream and wait for it to complete
                await stream()

            elif msg_type == "stop":
                if not session_id or session_id not in active_sessions:
                    await websocket.send_text(json.dumps({
                        "sessionId": session_id,
                        "token": "[ERROR] Session not initialized or invalid"
                    }))
                    continue

                # Optional: implement cancellation logic per session here
                # For now just send [[END]]
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
            # Create tables if they don't exist
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
    
    chat_db_relative = os.path.relpath(DATA_DIR, current_dir)
    reload_excludes = [os.path.join(chat_db_relative, '*')]
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=4580,
        ws="websockets",
        reload=True,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes
    )