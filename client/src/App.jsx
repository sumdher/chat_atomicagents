import { useRef, useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import Sidebar from "./components/Sidebar/Sidebar";
import Header from "./components/Header/Header";
import ChatArea from "./components/ChatArea/ChatArea";
import InputArea from "./components/InputArea/InputArea";
import './App.css';

// const PROVIDER_MODELS = {
//   openai: [
//     { id: 'gpt-4.1', name: 'GPT-4.1', description: 'Flagship GPT model for complex tasks' },
//     { id: 'gpt-4.1-mini', name: 'GPT-4.1 Mini', description: 'Balanced for intelligence, speed and cost' },
//     { id: 'gpt-4.1-nano', name: 'GPT-4.1 Nano', description: 'Fastest, most cost-effecttive GPT-4.1 model' },
//     { id: 'gpt-4o', name: 'GPT-4o', description: 'Fast, intellinent, flexible GPT model' },
//     { id: 'gpt-4o-mini', name: 'GPT-4 Mini', description: 'GPT-4o Mini' },
//     { id: 'o3', name: 'o3', description: 'Most poweful reasoning model' },
//     { id: 'o3-mini', name: 'o3-mini', description: 'A small model alternative to o3' },
//     { id: 'o3-deep-research', name: 'o3-deep-research', description: 'Most powerful deep research model' },
//     { id: 'o3-mini-deep-research', name: 'o3-mini-deep-research', description: 'Faster, more affordable research model' },
//     { id: 'o4-mini', name: 'o4-mini', description: 'Faster, more affordable reasoning model' },
//     // old
//     { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', description: 'Latest GPT-4 model with improved capabilities and knowledge' },
//     { id: 'gpt-4', name: 'GPT-4', description: 'Previous generation of GPT-4' },
//     { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', description: 'Faster and cheaper than GPT-4' },
//   ],
//   anthropic: [
//     { id: 'claude-opus-4-20250514', name: 'Claude Opus 4', description: 'Highest level of intelligence and capability' },
//     { id: 'claude-sonnnet-4-20250514', name: 'Claude Sonnet 4', description: 'High intelligence and balanced performance' },
//     { id: 'claude-3-7-sonnnet-latest', name: 'Claude Sonnet 3.7', description: 'High intelligence with toggleable extended thinking' },
//     { id: 'claude-3-5-haiku-latest', name: 'Claude Haiku 3.5', description: 'Intelligence at blazing speeds' },
//     { id: 'claude-3-5-sonnet-latest', name: 'Claude Sonnet 3.5 v2', description: '...' },
//     { id: 'claude-3-5-sonnet-20240620', name: 'Claude Sonnet 3.5', description: 'High level of intelligence and capability' },
//     { id: 'claude-3-haiku-20240307', name: 'Claude Haiku 3', description: 'Quick and accurate targeted performance' },
//     { id: 'claude-3-opus-20240229', name: 'Claude Opus 3', description: 'Top-level intelligence, fluency, and understanding' },
//   ],
//   gemini: [
//     { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro', description: 'Enhanced thinking and reasoning, multimodal understanding, advanced coding, and more' },
//     { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', description: 'Adaptive thinking, cost efficiency' },
//     { id: 'gemini-2.5-flash-lite-preview-06-17', name: 'Gemini 2.5 Flash-Lite Preview', description: 'Most cost-efficient model supporting high throughput' },
//     { id: 'gemini-2.0-flash', name: 'Gemini 2.5 Pro', description: 'Next generation features, speed, and realtime streaming' },
//     { id: 'gemini-2.0-flash-lite', name: 'Gemini 2.0 Flash-Lite', description: 'Cost efficiency and low latency' },
//     { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', description: 'Fast and versatile performance across a diverse variety of tasks' },
//     { id: 'gemini-1.5-flash-8b', name: 'Gemini 1.5 Flash 8B', description: 'High volume and lower intelligence tasks' },
//     { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro', description: 'Complex reasoning tasks requiring more intelligence' },
//   ]
// };

const PROVIDER_MODELS = {
  openai: [
    {
      id: 'gpt-4.1',
      name: 'GPT-4.1',
      description:
        'Top-tier GPT model for complex reasoning, coding, and instruction-following. Supports multimodal input (text + image). Pricing: $2.00 / 1M input, $8.00 / 1M output tokens.'
    },
    {
      id: 'gpt-4.1-mini',
      name: 'GPT-4.1 Mini',
      description:
        'Balanced GPT-4.1 variant optimized for cost and speed. Supports multimodal input (text + image). Pricing: $0.40 / 1M input, $1.60 / 1M output tokens.'
    },
    {
      id: 'gpt-4.1-nano',
      name: 'GPT-4.1 Nano',
      description:
        'Most affordable GPT-4.1 model with solid reasoning performance. Multimodal support (text + image). Pricing: $0.10 / 1M input, $0.40 / 1M output tokens.'
    },
    {
      id: 'o3',
      name: 'OpenAI o3',
      description:
        'Highly capable reasoning model optimized for math, science, and coding. Full multimodal support including vision. Pricing: $2.00 / 1M input, $8.00 / 1M output tokens.'
    },
    {
      id: 'o4-mini',
      name: 'OpenAI o4 Mini',
      description:
        'Affordable reasoning model ideal for math, code, and visual tasks. Multimodal (text + image). Pricing: $1.10 / 1M input, $4.40 / 1M output tokens.'
    },
    {
      id: 'gpt-4-turbo',
      name: 'GPT-4 Turbo',
      description:
        'Earlier GPT-4 model with good reasoning performance and multimodal support. Pricing: $5.00 / 1M input, $20.00 / 1M output tokens.'
    },
    {
      id: 'gpt-4o-mini',
      name: 'GPT-4o Mini',
      description:
        'Lean version of GPT-4 Turbo for low-latency use. Multimodal capable. Pricing: $0.60 / 1M input, $2.40 / 1M output tokens.'
    },
    {
      id: 'gpt-3.5-turbo',
      name: 'GPT-3.5 Turbo',
      description:
        'Fast and affordable model for standard chat tasks. Text-only. Pricing: $0.50 / 1M input, $1.50 / 1M output tokens.'
    }
  ],

  anthropic: [
    {
      id: 'claude-opus-4',
      name: 'Claude Opus 4',
      description:
        'Most advanced Claude model for agentic workflows, reasoning, and coding. Multimodal support (vision + text). Pricing: $15.00 / 1M input, $75.00 / 1M output tokens.'
    },
    {
      id: 'claude-sonnet-4',
      name: 'Claude Sonnet 4',
      description:
        'Versatile model for planning, coding, and general reasoning. Multimodal (vision + text). Pricing: $3.00 / 1M input, $15.00 / 1M output tokens.'
    }
  ],

  gemini: [
    {
      id: 'gemini-2.5-pro',
      name: 'Gemini 2.5 Pro',
      description:
        'High-end multimodal model for reasoning, coding, and long-context tasks. Supports 1M context, text + image. Pricing: $1.25 / 1M input, $10.00 / 1M output tokens.'
    },
    {
      id: 'gemini-2.5-flash',
      name: 'Gemini 2.5 Flash',
      description:
        'Balanced model for fast, multimodal tasks. Supports 1M context, text + image. Pricing: $0.30 / 1M input, $2.50 / 1M output tokens.'
    },
    {
      id: 'gemini-2.5-flash-lite-preview-06-17',
      name: 'Gemini 2.5 Flash-Lite Preview',
      description:
        'Lightest Gemini 2.5 model for high-volume workloads. Multimodal. Pricing: $0.10 / 1M input, $0.40 / 1M output tokens.'
    },
    {
      id: 'gemini-2.0-flash',
      name: 'Gemini 2.0 Flash',
      description:
        'Early multimodal model for general reasoning. Supports text + image. Pricing: $0.10 / 1M input, $0.40 / 1M output tokens.'
    },
    {
      id: 'gemini-2.0-flash-lite',
      name: 'Gemini 2.0 Flash-Lite',
      description:
        'Lightweight, cost-efficient multimodal model. Text + image support. Pricing: $0.075 / 1M input, $0.30 / 1M output tokens.'
    },
    {
      id: 'gemini-1.5-pro',
      name: 'Gemini 1.5 Pro',
      description:
        'Strong long-context reasoning model (up to 2M tokens). Multimodal. Pricing: $1.25 / 1M input, $5.00 / 1M output tokens.'
    },
    {
      id: 'gemini-1.5-flash',
      name: 'Gemini 1.5 Flash',
      description:
        'Efficient model for common tasks with fast responses. Multimodal. Pricing: $0.075 / 1M input, $0.30 / 1M output tokens.'
    },
    {
      id: 'gemini-1.5-flash-8b',
      name: 'Gemini 1.5 Flash 8B',
      description:
        'Smaller Gemini model tuned for high-throughput, low-intelligence workloads. Multimodal. Pricing: $0.0375 / 1M input, $0.15 / 1M output tokens.'
    }
  ]
};

const DEFAULT_MODELS = {
  openai: 'gpt-4.1',
  anthropic: 'claude-3-sonnet-20240229',
  gemini: 'gemini-1.5-pro'
};

function createDefaultSession() {
  const provider = "openai";
  return {
    id: uuidv4(),
    title: "New Chat",
    messages: [],
    loadedKeys: new Set(),
    isConnected: false,
    isConnecting: false,
    isTyping: false,
    isLoadingContext: false,
    input: "",
    provider: provider,
    model: DEFAULT_MODELS[provider],
    hasBeenConnected: false
  };
}

export default function App() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const socketRef = useRef(null);

  const selectModelAndConnect = (modelId) => {
    // Update the model in state
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, model: modelId, isConnecting: true }
        : session
    ));

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        type: "init",
        sessionId: activeSessionId,
        provider: activeSession.provider,
        model: modelId,
      }));
    }
  };

  const getProviderDisplayName = (provider) => {
    const providerNames = {
      openai: "OpenAI",
      anthropic: "Anthropic",
      gemini: "Gemini"
    };
    return providerNames[provider] || provider;
  };

  const getModelDisplayName = (session) => {
    if (!session || !session.provider || !session.model) return null;
    const providerModels = PROVIDER_MODELS[session.provider] || [];
    const model = providerModels.find(m => m.id === session.model);
    return model ? model.name : null;
  };



  useEffect(() => {
    const savedSessions = localStorage.getItem('chat_sessions');
    const savedActiveId = localStorage.getItem('active_session_id');

    let initialSessions = [];
    let initialActiveId = null;

    if (savedSessions) {
      try {
        const parsed = JSON.parse(savedSessions);
        initialSessions = parsed.map(s => ({
          ...s,
          loadedKeys: new Set(s.loadedKeys || []),
          isConnected: false,
          isConnecting: false,
          isTyping: false,
          isLoadingContext: false,
          input: "",
          hasBeenConnected: s.hasBeenConnected ?? true,
        }));
        initialActiveId = savedActiveId || (parsed[0]?.id || null);
      } catch (e) {
        console.error("Error loading sessions:", e);
        initialSessions = [createDefaultSession()];
        initialActiveId = initialSessions[0].id;
      }
    } else {
      initialSessions = [createDefaultSession()];
      initialActiveId = initialSessions[0].id;
    }

    setSessions(initialSessions);
    setActiveSessionId(initialActiveId);
    setIsInitialized(true);
  }, []);

  // Save sessions & activeSessionId to localStorage on change
  useEffect(() => {
    if (!isInitialized) return;

    const sessionsToSave = sessions.map(s => ({
      id: s.id,
      title: s.title,
      messages: s.messages,
      loadedKeys: Array.from(s.loadedKeys),
      provider: s.provider,
      model: s.model,
      hasBeenConnected: s.hasBeenConnected,
    }));

    localStorage.setItem('chat_sessions', JSON.stringify(sessionsToSave));
    localStorage.setItem('active_session_id', activeSessionId);
  }, [sessions, activeSessionId, isInitialized]);

  // Update session title from first user message
  useEffect(() => {
    setSessions(prev => prev.map(session => {
      if (session.id === activeSessionId && session.messages.length > 0) {
        const firstMessage = session.messages.find(m => m.from === "user")?.text || "";
        if (firstMessage && session.title === "New Chat") {
          const newTitle = firstMessage.length > 20
            ? `${firstMessage.substring(0, 20)}...`
            : firstMessage;
          return { ...session, title: newTitle };
        }
      }
      return session;
    }));
  }, [sessions.find(s => s.id === activeSessionId)?.messages]);

  // Open one persistent WebSocket on app mount
  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:4580/ws/chat");
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("✅ WebSocket connected");
      // Initialize currently active session on connect
      if (activeSessionId) {
        const activeSession = sessions.find(s => s.id === activeSessionId);
        if (activeSession) {
          socket.send(JSON.stringify({
            type: "init",
            sessionId: activeSessionId,
            provider: activeSession.provider,
            model: activeSession.model,
          }));
          setSessions(prev => prev.map(session =>
            session.id === activeSessionId
              ? { ...session, isConnecting: true }
              : session
          ));
        }
      }
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const { sessionId, token } = data;
        if (!sessionId) return;

        setSessions(prev => prev.map(session => {
          if (session.id !== sessionId) return session;

          if (token === "[[END]]") {
            return {
              ...session,
              isTyping: false,
              isLoadingContext: false,
              isConnected: true,
              isConnecting: false,
              hasBeenConnected: true,
            };
          }

          if (token.startsWith("[[LOADED::")) {
            const loadedList = token.replace("[[LOADED::", "").replace("]]", "").split(",");
            const newLoadedKeys = new Set([...session.loadedKeys, ...loadedList]);
            return { ...session, loadedKeys: newLoadedKeys, isLoadingContext: true };
          }

          if (token.startsWith("[ERROR]")) {
            console.error(`Session ${sessionId} error:`, token);
            return { ...session, isTyping: false, isConnected: false, isConnecting: false };
          }

          // Append or create bot message
          const lastMsg = session.messages[session.messages.length - 1];
          let newMessages = [...session.messages];
          if (lastMsg?.from === "bot") {
            newMessages[newMessages.length - 1] = { ...lastMsg, text: lastMsg.text + token };
          } else {
            newMessages.push({ from: "bot", text: token });
          }

          return {
            ...session,
            messages: newMessages,
            isTyping: true,
            isConnected: true,
            isConnecting: false,
            hasBeenConnected: true,
          };
        }));

      } catch (e) {
        console.error("Error parsing WS message", e);
      }
    };

    socket.onclose = () => {
      console.log("❌ WebSocket closed");
      setSessions(prev => prev.map(session => ({
        ...session,
        isConnected: false,
        isConnecting: false,
        isTyping: false,
      })));
    };

    socket.onerror = (error) => {
      console.error("WebSocket error", error);
    };

    return () => {
      socket.close();
    };
  }, []);

  const activeSession = sessions.find(session => session.id === activeSessionId) || sessions[0];
  const modelDisplayName = getModelDisplayName(activeSession);

  const createNewSession = () => {
    const currentProvider = activeSession.provider;
    const newSession = {
      ...createDefaultSession(),
      provider: currentProvider,
      model: activeSession.model
    };

    setSessions(prev => [...prev, newSession]);
    setActiveSessionId(newSession.id);

  };

  const deleteSession = (id) => {
    if (sessions.length <= 1) return;

    setSessions(prev => prev.filter(session => session.id !== id));

    if (id === activeSessionId) {
      const newActive = sessions.find(s => s.id !== id);
      setActiveSessionId(newActive.id);
    }
  };

  const switchSession = (id) => {
    setActiveSessionId(id);

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      const session = sessions.find(s => s.id === id);
      if (session) {
        socketRef.current.send(JSON.stringify({
          type: "init",
          sessionId: id,
          provider: session.provider,
        }));
        setSessions(prev => prev.map(session =>
          session.id === id
            ? { ...session, isConnecting: true }
            : session
        ));
      }
    }
  };

  const setSessionProvider = (provider) => {
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? {
          ...session,
          provider,
          model: DEFAULT_MODELS[provider]
        }
        : session
    ));
  };


  const sendMessage = () => {
    if (!activeSession.input.trim() ||
      activeSession.isTyping ||
      socketRef.current?.readyState !== WebSocket.OPEN) return;

    const userMessage = { from: "user", text: activeSession.input };

    setSessions(prev => prev.map(session => {
      if (session.id === activeSessionId) {
        return {
          ...session,
          messages: [...session.messages, userMessage],
          input: "",
          isTyping: true
        };
      }
      return session;
    }));

    socketRef.current.send(JSON.stringify({
      type: "message",
      sessionId: activeSessionId,
      text: activeSession.input,
    }));
  };

  // Stop AI generation for active session
  const stopAI = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        type: "stop",
        sessionId: activeSessionId,
      }));

      setSessions(prev => prev.map(session =>
        session.id === activeSessionId
          ? { ...session, isTyping: false }
          : session
      ));
    }
  };

  // Update input text for active session
  const setInput = (value) => {
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, input: value }
        : session
    ));
  };

  if (!isInitialized) {
    return <div className="app-loading">Loading chats...</div>;
  }
  
  const headerTitle = activeSession.hasBeenConnected
    ? (modelDisplayName || activeSession.model)
    : getProviderDisplayName(activeSession.provider);
  
  return (
    <div className="app-container">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        setActiveSessionId={switchSession}
        createNewSession={createNewSession}
        deleteSession={deleteSession}
        isLoadingContext={activeSession.isLoadingContext}
        setIsLoadingContext={(value) => {
          setSessions(prev => prev.map(session =>
            session.id === activeSessionId
              ? { ...session, isLoadingContext: value }
              : session
          ));
        }}
      />
      <div className="main">
        <Header title={headerTitle} />
        {!activeSession.isConnected && !activeSession.hasBeenConnected ? (
          <div className="connect-screen">
            <div className="provider-selection">
              <label>Select AI Provider:</label>
              <select
                value={activeSession.provider}
                onChange={(e) => setSessionProvider(e.target.value)}
                disabled={activeSession.isConnecting}
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>

            <div className="model-grid">
              {PROVIDER_MODELS[activeSession.provider]?.map((model) => (
                <button
                  key={model.id}
                  className={`model-card ${activeSession.model === model.id ? 'selected' : ''}`}
                  onClick={() => selectModelAndConnect(model.id)}
                  disabled={activeSession.isConnecting}
                >
                  <h4>{model.name}</h4>
                  <p>
                    {model.description.split('Pricing:')[0].trim()}
                    <br />
                    <strong>Pricing:</strong> {model.description.split('Pricing:')[1]?.trim()}
                  </p>
                  {activeSession.model === model.id && activeSession.isConnecting && (
                    <div className="model-card-spinner"></div>
                  )}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            <ChatArea
              messages={activeSession.messages}
              isTyping={activeSession.isTyping}
              isLoadingContext={activeSession.isLoadingContext}
            />
            <InputArea
              input={activeSession.input}
              setInput={setInput}
              sendMessage={sendMessage}
              stopAI={stopAI}
              isTyping={activeSession.isTyping}
            />
          </>
        )}
      </div>
    </div>
  );
}