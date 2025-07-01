import os
import uuid
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dotenv.load_dotenv()

sessions: Dict[str, 'Session'] = {}
UPLOAD_DIR = "user_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
file_map: Dict[str, str] = {}  # {file_key: original_name}

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent = None
        self.loaded_files: Set[str] = set()
        self.active_generation = None
        
    def initialize_agent(self, provider: str):
        """Initialize agent after provider is known"""
        client, model = self.setup_client(provider)
        self.memory = AgentMemory()
        
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

        initial_message = "Hello! How can I assist you today?"
        initial_schema = BaseAgentOutputSchema(chat_message=initial_message)
        self.memory.add_message("assistant", content=initial_schema)
        
        self.agent = BaseAgent(
            config=BaseAgentConfig(
                client=client,
                model=model,
                memory=self.memory,
                system_prompt_generator=system_prompt_geospatial,
                model_api_parameters={"max_tokens": 2048}
            )
        )
        print(f"✅ Agent initialized with provider: {provider}, model: {model}")
    
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
            
        print(f"Processing input: '{input_text}'")
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
                self.memory.add_message("assistant", {"chat_message": output_schema})
                
        except Exception as e:
            traceback.print_exc()
            yield f"[ERROR] {str(e)}"

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        # Step 1: Get provider selection from client
        provider_data = await websocket.receive_text()
        print(f"Received provider data: {provider_data}")
        
        try:
            provider_info = json.loads(provider_data)
            provider = provider_info.get("provider")
            session_id = provider_info.get("sessionId")
            
            if not provider or not session_id:
                error_msg = "Missing provider or session ID"
                print(error_msg)
                await websocket.send_text(f"[ERROR] {error_msg}")
                await websocket.close(code=1008)
                return
        except json.JSONDecodeError as e:
            error_msg = f"Invalid provider format: {str(e)}"
            print(error_msg)
            await websocket.send_text(f"[ERROR] {error_msg}")
            await websocket.close(code=1008)
            return
        
        # Step 2: Create or retrieve session
        if session_id not in sessions:
            sessions[session_id] = Session(session_id)
            print(f"Created new session: {session_id}")
        
        session = sessions[session_id]

        # Initialize agent if not already done
        if not session.agent:
            try:
                session.initialize_agent(provider)
                print(f"✅ Initialized agent for session {session_id} with provider {provider}")
            except Exception as e:
                error_msg = f"Agent initialization failed: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                await websocket.send_text(f"[ERROR] {error_msg}")
                await websocket.close(code=1011)
                return
        
        await websocket.send_text("Hello! How can I assist you today?")
        await websocket.send_text("[[END]]") 
        print("Sent welcome message")
        
        # Step 3: Handle messages
        while True:
            data = await websocket.receive_text()
            print(f"Received message: '{data}'")
            
            if data == "__STOP__":
                print("Received STOP command")
                # session.cancel_generation()
                await websocket.send_text("[[END]]")
                continue
                
            elif data == "__CONTEXT__":
                print("Loading files to context")
                # Load files to context
                new_files = []
                for file_key, filename in file_map.items():
                    if file_key not in session.loaded_files:
                        file_path = os.path.join(UPLOAD_DIR, file_key)
                        try:
                            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = await f.read()
                                # Add as system message
                                session.agent.memory.add_message(
                                    "system", 
                                    {"chat_message": f"File: {filename}\nContent:\n{content}"}
                                )
                                session.loaded_files.add(file_key)
                                new_files.append(filename)
                        except Exception as e:
                            error_msg = f"Error reading file {file_key}: {e}"
                            print(error_msg)
                            await websocket.send_text(f"[ERROR] {error_msg}")
                
                if new_files:
                    await websocket.send_text(f"[[LOADED::{','.join(new_files)}]]")
                await websocket.send_text("[[END]]")
                print("Finished loading context")
                continue
            
            # Normal message processing
            session.active_generation = asyncio.create_task(
                stream_to_websocket(session, websocket, data)
            )
            await session.active_generation
            
    except WebSocketDisconnect:
        print("Client disconnected")
        pass
    except Exception as e:
        error_msg = f"WebSocket error: {str(e)}"
        print(error_msg)
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

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    key_map = {}
    for file in files:
        key = f"{int(time.time() * 1000)}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, key)
        
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        file_map[key] = file.filename
        key_map[key] = file.filename
        saved_files.append(file.filename)
    
    return {"status": "success", "uploaded": saved_files, "file_map": key_map}

@app.post("/delete-file")
async def delete_file(file_key: str = Form(...)):
    if file_key not in file_map:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(UPLOAD_DIR, file_key)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    del file_map[file_key]
    
    # Remove from all sessions that loaded this file
    for session in sessions.values():
        if file_key in session.loaded_files:
            session.loaded_files.remove(file_key)
    
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4580, ws="websockets")