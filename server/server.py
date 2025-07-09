from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
from subprocess import CalledProcessError
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, AsyncGenerator, Dict, Optional, Set
from google.oauth2 import id_token
from google.auth.transport import requests

import asyncpg
import dotenv
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# === 3rd‑party LLM helper libs (import inside functions if heavy) ===
import instructor  # type: ignore

# ---------------------------------------------------------------------------
# Environment / configuration ------------------------------------------------
# ---------------------------------------------------------------------------
LOCAL_MODE = (
    "local" in sys.argv
    or "--local" in sys.argv
    or os.getenv("LOCAL", "0") == "1"
)

DB_CONTAINER_NAME = "chat-app-db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "chat_db")
DOTENV_PATH: Optional[str] = None

if LOCAL_MODE:
    os.makedirs(os.path.join(DATA_DIR, "user_files"), exist_ok=True)
    if dotenv.find_dotenv():
        DOTENV_PATH = dotenv.find_dotenv()
        dotenv.load_dotenv(DOTENV_PATH)
else:
    # In production we expect envs already set (Render / Docker secrets etc.)
    pass

ENVS: Dict[str, str | None] = dict(os.environ)
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

DB = bool(DATABASE_URL)
if not DB:
    print("⚠️ DATABASE_URL not set - chat history will NOT persist.")

# LLM provider → env var mapping ---------------------------------------------
PROVIDER_ENV_MAP: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# ---------------------------------------------------------------------------
# Pydantic models -------------------------------------------------------------
# ---------------------------------------------------------------------------
class User(BaseModel):
    id: str
    email: str
    name: str
    picture: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    user_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Password hashing & auth helpers --------------------------------------------
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(pwd: str) -> str:
    return pwd_context.hash(pwd)


# ---------------------------------------------------------------------------
# FastAPI app instantiation (with lifespan) ----------------------------------
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if LOCAL_MODE:
        _start_local_postgres()
        await asyncio.sleep(2)
    
    if DB:    
        app.state.db_pool = await asyncpg.create_pool(
            DATABASE_URL, 
            min_size=5, 
            max_size=20,
            command_timeout=60
        )
        await _init_db(app.state.db_pool)
    
    if not LOCAL_MODE:
        await _load_keys_from_db(app.state.db_pool)

    print("✅ Database connected and initialized")
    
    yield
    
    if DB and hasattr(app.state, "db_pool"):
        await app.state.db_pool.close()
        print("❌ Database connection closed")
    
    if LOCAL_MODE:
        await _stop_local_postgres()

if DB:
    app = FastAPI(lifespan=lifespan)
else:
    app = FastAPI()

sessions: Dict[str, Session] = {}

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://172.23.32.1:5173",
    "http://192.168.8.172:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------------------------------------------------------------------
# Utility functions -----------------------------------------------------------
# ---------------------------------------------------------------------------
def _start_local_postgres() -> None:
    """Launch the docker compose / shell script that starts Postgres locally."""

    try:
        subprocess.run(["docker", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    

        script = "docker-db.ps1" if sys.platform == "win32" else "docker-db.sh"
        script_path = os.path.join(DATA_DIR, script)
        if not os.path.exists(script_path):
            print(f"⚠️ Local DB script not found: {script_path}")
            return
        try:
            subprocess.run(
                ["bash", script_path] if sys.platform != "win32" else ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", script_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅ Local PostgreSQL container started")
            
        except subprocess.CalledProcessError as exc:
            print(f"❌ Failed to start local Postgres: {exc}")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Docker error: {str(e)}. Using existing database connection.")


async def _stop_local_postgres() -> None:
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["docker", "stop", DB_CONTAINER_NAME],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Local PostgreSQL container stopped")
            else:
                error_msg = result.stderr.strip()
                if "No such container" in error_msg:
                    print(f"⚠️ Container {DB_CONTAINER_NAME} not found")
                else:
                    print(f"⚠️ Could not stop container: {error_msg}")
        except CalledProcessError as e:
            print(e)
        
    else:
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "stop",
                DB_CONTAINER_NAME,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                print("✅ Local PostgreSQL container stopped")
            else:
                error_msg = stderr.decode().strip()
                if "No such container" in error_msg:
                    print(f"⚠️ Container {DB_CONTAINER_NAME} not found")
                else:
                    print(f"⚠️ Could not stop container: {error_msg}")
        except FileNotFoundError:
            print("❌ Docker not installed, skipping container stop")


async def _init_db(pool: asyncpg.pool.Pool) -> None:
    """Create tables / indices if they don't exist."""

    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS chat_messages CASCADE;")
        await conn.execute("DROP TABLE IF EXISTS chat_sessions CASCADE;")
        await conn.execute("DROP TABLE IF EXISTS llm_api_keys CASCADE;")
        await conn.execute("DROP TABLE IF EXISTS users CASCADE;")
        
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                picture TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Chat',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                loaded_keys TEXT[] DEFAULT '{}'::TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_api_keys (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                env_var TEXT NOT NULL,
                api_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, provider)
            )
            """
        )

        # indices -----------------------------------------------------------
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_api_keys_user ON llm_api_keys(user_id)"
        )

        print("✅ Database schema ensured")

async def _load_keys_from_db(pool: asyncpg.pool.Pool) -> None:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT env_var, api_key FROM llm_api_keys")
        for row in rows:
            os.environ[row["env_var"]] = row["api_key"]


# ---------------------------------------------------------------------------
# JWT helpers ----------------------------------------------------------------
# ---------------------------------------------------------------------------
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Token extraction – first from header, then from cookie ---------------------
async def _extract_token(request: Request) -> str:
    header = request.headers.get("Authorization")
    if header and header.lower().startswith("bearer "):
        return header.split(" ", 1)[1]

    cookie = request.cookies.get("access_token")
    if cookie:
        return cookie

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
    )


async def get_current_user(request: Request) -> User:
    token = await _extract_token(request)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    async with app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, name, picture FROM users WHERE id = $1", user_id
        )
        if not row:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return User(**row)


CurrentUser = Annotated[User, Depends(get_current_user)]

# ---------------------------------------------------------------------------
# Auth endpoints -------------------------------------------------------------
# ---------------------------------------------------------------------------
COOKIE_PARAMS = dict(
    httponly=True,
    max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    secure=not LOCAL_MODE,
    samesite="lax",
)


@app.post("/auth/register")
async def register(form_data: OAuth2PasswordRequestForm = Depends()) -> JSONResponse:
    async with app.state.db_pool.acquire() as conn:
        if await conn.fetchval("SELECT 1 FROM users WHERE email = $1", form_data.username):
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO users (id, email, hashed_password, name)
            VALUES ($1, $2, $3, $4)
            """,
            user_id,
            form_data.username,
            hash_password(form_data.password),
            form_data.username.split("@")[0],
        )

    token = create_access_token({"sub": user_id})
    resp = JSONResponse(Token(access_token=token, token_type="bearer").model_dump())
    resp.set_cookie("access_token", token, **COOKIE_PARAMS)
    return resp


@app.get("/auth/token")
async def refresh_token(request: Request) -> JSONResponse:
    try:
        current_user = await get_current_user(request)
    except HTTPException:
        # Return empty token instead of 401
        return JSONResponse({"token": ""})

    token = create_access_token({"sub": current_user.id})
    resp = JSONResponse({"token": token})
    resp.set_cookie("access_token", token, **COOKIE_PARAMS)
    return resp


@app.get("/auth/me")
async def read_me(current_user: CurrentUser) -> User:
    return current_user


# Google OAuth (mock?) --------------------------------------------------------
@app.post("/auth/google")
async def google_oauth(request: Request):
    try:
        data = await request.json()
        token = data.get("token")

        # Verify Google token
        id_info = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            "851038444090-ihuvm3cdm6dl74sq1078peptj37r6mlr.apps.googleusercontent.com",  # Your client ID
        )

        if id_info["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
            raise ValueError("Invalid issuer")

        email = id_info["email"]
        name = id_info.get("name", email.split("@")[0])

        print(email)
        print(name)

        async with app.state.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM users WHERE email = $1", email)
            if not row:
                user_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO users (id, email, name)
                    VALUES ($1, $2, $3)
                    """,
                    user_id,
                    email,
                    name,
                )
            else:
                user_id = row["id"]

        user = {"id": user_id, "email": email, "name": name}
        token_jwt = create_access_token({"sub": user_id})
        resp = JSONResponse(
            {
                "access_token": token_jwt,
                "token_type": "bearer",
                "user": user,
            }
        )
        resp.set_cookie("access_token", token_jwt, **COOKIE_PARAMS)
        return resp

    except ValueError as e:
        print("OAuth error:", e)
        return JSONResponse({"detail": str(e)}, status_code=400)
    except Exception as e:
        print("Unexpected error during Google OAuth:", e)
        return JSONResponse({"detail": "Server error"}, status_code=500)


# ---------------------------------------------------------------------------
# API key management ---------------------------------------------------------
# ---------------------------------------------------------------------------


def _get_env_api_keys() -> Dict[str, str]:
    if LOCAL_MODE and DOTENV_PATH:
        dotenv.load_dotenv(DOTENV_PATH)
        envs = dotenv.dotenv_values(DOTENV_PATH)
    else:
        envs = os.environ
    return {k: v for k, v in envs.items() if k in PROVIDER_ENV_MAP.values()}


@app.get("/api-keys")
async def list_api_keys(current_user: CurrentUser) -> Dict[str, str]:
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT provider, api_key FROM llm_api_keys WHERE user_id = $1", current_user.id
        )
    return {row["provider"]: row["api_key"] for row in rows}


@app.post("/api-keys")
async def upsert_api_key(provider: str, key: str, current_user: CurrentUser) -> dict:
    env_var = PROVIDER_ENV_MAP.get(provider.lower(), provider)
    os.environ[env_var] = key

    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO llm_api_keys (user_id, provider, env_var, api_key)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, provider) DO UPDATE
            SET api_key = EXCLUDED.api_key
            """,
            current_user.id,
            provider.lower(),
            env_var,
            key,
        )

    return {"status": "success"}


@app.delete("/api-keys/{provider}")
async def remove_api_key(provider: str, current_user: CurrentUser) -> dict:
    env_var = PROVIDER_ENV_MAP.get(provider.lower(), provider)
    os.environ.pop(env_var, None)

    async with app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM llm_api_keys WHERE user_id = $1 AND provider = $2",
            current_user.id,
            provider.lower(),
        )
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="API key not found")

    return {"status": "success"}


# ---------------------------------------------------------------------------
# Chat sessions CRUD ---------------------------------------------------------
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"


@app.get("/sessions")
async def list_sessions(current_user: CurrentUser):
    async with app.state.db_pool.acquire() as conn:
        sessions_db = await conn.fetch(
            """
            SELECT cs.*, COUNT(cm.id) AS message_count
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cs.id = cm.session_id
            WHERE cs.user_id = $1
            GROUP BY cs.id
            ORDER BY cs.updated_at DESC
            """,
            current_user.id,
        )

        result = []
        for s in sessions_db:
            messages = await conn.fetch(
                "SELECT role, content FROM chat_messages WHERE session_id = $1 ORDER BY created_at",
                s["id"],
            )
            result.append(
                {
                    "id": s["id"],
                    "title": s["title"],
                    "provider": s["provider"],
                    "model": s["model"],
                    "loadedKeys": s["loaded_keys"] or [],
                    "messages": [
                        {"from": "user" if m["role"] == "user" else "bot", "text": m["content"]}
                        for m in messages
                    ],
                }
            )
    return result


@app.post("/sessions")
async def create_session(current_user: CurrentUser):
    session_id = str(uuid.uuid4())
    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chat_sessions (id, user_id, provider, model)
            VALUES ($1, $2, $3, $4)
            """,
            session_id,
            current_user.id,
            DEFAULT_PROVIDER,
            DEFAULT_MODEL,
        )

    return {
        "id": session_id,
        "title": "New Chat",
        "provider": DEFAULT_PROVIDER,
        "model": DEFAULT_MODEL,
        "loadedKeys": [],
        "messages": [],
    }


@app.put("/sessions/{session_id}")
async def update_session(session_id: str, payload: dict, current_user: CurrentUser):
    async with app.state.db_pool.acquire() as conn:
        owner = await conn.fetchval("SELECT user_id FROM chat_sessions WHERE id = $1", session_id)
        if owner != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden")

        await conn.execute(
            """
            UPDATE chat_sessions SET
                title        = COALESCE($2, title),
                provider     = COALESCE($3, provider),
                model        = COALESCE($4, model),
                loaded_keys  = COALESCE($5, loaded_keys),
                updated_at   = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            session_id,
            payload.get("title"),
            payload.get("provider"),
            payload.get("model"),
            payload.get("loadedKeys"),
        )

        if "messages" in payload:
            await conn.execute("DELETE FROM chat_messages WHERE session_id = $1", session_id)
            for m in payload["messages"]:
                await conn.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES ($1, $2, $3)
                    """,
                    session_id,
                    "user" if m["from"] == "user" else "assistant",
                    m["text"],
                )
    return {"status": "success"}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: CurrentUser):
    async with app.state.db_pool.acquire() as conn:
        owner = await conn.fetchval("SELECT user_id FROM chat_sessions WHERE id = $1", session_id)
        if owner != current_user.id:
            raise HTTPException(status_code=403, detail="Forbidden")

        await conn.execute("DELETE FROM chat_sessions WHERE id = $1", session_id)
    return {"status": "success"}


from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema, AgentMemory, SystemPromptGenerator

class Session:
    """One per chat session id - reused across websocket reconnects."""
    
    # sessions: Dict[str, Session] = {}
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent: Optional[BaseAgent] = None
        self.loaded_files: Set[str] = set()
        self.provider: Optional[str] = None
        self.model: Optional[str] = None
        self.memory = AgentMemory()
        self.history_loaded = False
        
    async def initialize_agent(self, provider: str, model: str):
        """Initialize agent after provider is known"""
        
        if self.agent and self.provider == provider and self.model == model:
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
            client = self.setup_client(self.provider)
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
async def ws_chat(websocket: WebSocket):
    token = websocket.query_params.get("token")
    
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await websocket.accept()
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

    except WebSocketDisconnect as wse:
        print(f"Client disconnected: {str(wse)}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        traceback.print_exc()
        await websocket.close(code=1011, reason=str(e))

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

if __name__ == "__main__":
    import uvicorn
    import signal
    from typing import List
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reload_dirs = ["."]
    
    if DB:
        chat_db_relative = os.path.relpath(DATA_DIR, current_dir)
        reload_excludes: List[str] = [os.path.join(chat_db_relative, '*')]
    else:
        reload_excludes = []
        
    def handle_exit_signal():
        print("⚠️ Received shutdown signal, cleaning up...")
        loop = asyncio.get_event_loop()
        if DB and hasattr(app.state, "db_pool"):
            loop.run_until_complete(app.state.db_pool.close())
        if LOCAL_MODE:
            loop.run_until_complete(_stop_local_postgres())
        print("✅ Clean shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda sig, frame: handle_exit_signal())
    signal.signal(signal.SIGTERM, lambda sig, frame: handle_exit_signal())

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=4580,
        ws="websockets",
        log_level="debug",
        reload=LOCAL_MODE,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes
    )
