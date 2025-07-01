import { useRef, useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import Sidebar from "./components/Sidebar/Sidebar";
import Header from "./components/Header/Header";
import ChatArea from "./components/ChatArea/ChatArea";
import InputArea from "./components/InputArea/InputArea";
import './App.css';
// import './index.css'

function createDefaultSession() {
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
    provider: "openai"
  };
}

export default function App() {
  const [isInitialized, setIsInitialized] = useState(false); // New state for initialization
  const [sessions, setSessions] = useState([]); // Start with empty array
  const [activeSessionId, setActiveSessionId] = useState(null); // Start with null
  const socketRef = useRef(null);

  // Initialize from localStorage
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
          input: ""
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

  // Save to localStorage whenever sessions change
  useEffect(() => {
    if (!isInitialized) return;

    const sessionsToSave = sessions.map(s => ({
      id: s.id,
      title: s.title,
      messages: s.messages,
      loadedKeys: Array.from(s.loadedKeys),
      provider: s.provider
    }));

    localStorage.setItem('chat_sessions', JSON.stringify(sessionsToSave));
    localStorage.setItem('active_session_id', activeSessionId);
  }, [sessions, activeSessionId, isInitialized]);

  // Update session title based on first message
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

  if (!isInitialized) {
    return <div className="app-loading">Loading chats...</div>;
  }

  const activeSession = sessions.find(session => session.id === activeSessionId) || sessions[0];

  // Create new session
  const createNewSession = () => {
    const currentProvider = activeSession.provider;
    const newSession = {
      ...createDefaultSession(),
      title: `Chat ${sessions.length + 1}`,
      provider: currentProvider
    };

    setSessions(prev => [...prev, newSession]);
    setActiveSessionId(newSession.id);

    // Close existing connection if any
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }
  };

  // Delete a session
  const deleteSession = (id) => {
    if (sessions.length <= 1) return; // Don't delete the last session

    setSessions(prev => prev.filter(session => session.id !== id));

    // If deleting active session, switch to first available
    if (id === activeSessionId) {
      const newActive = sessions.find(s => s.id !== id);
      setActiveSessionId(newActive.id);
    }
  };

  // Update session provider
  const setSessionProvider = (provider) => {
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, provider }
        : session
    ));
  };

  // Connect to WebSocket
  const connectToLLM = () => {
    // Close existing connection if any
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }

    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, isConnecting: true }
        : session
    ));

    const socket = new WebSocket("ws://127.0.0.1:4580/ws/chat");
    socketRef.current = socket;

    socket.onmessage = (event) => {
      const chunk = event.data;

      if (chunk === "[[END]]") {
        setSessions(prev => prev.map(session =>
          session.id === activeSessionId
            ? { ...session, isTyping: false, isLoadingContext: false }
            : session
        ));
        return;
      }

      if (chunk.startsWith("[[LOADED::")) {
        const loadedList = chunk.replace("[[LOADED::", "").replace("]]", "").split(",");
        setSessions(prev => prev.map(session => {
          if (session.id === activeSessionId) {
            const newLoadedKeys = new Set([...session.loadedKeys, ...loadedList]);
            return { ...session, loadedKeys: newLoadedKeys, isLoadingContext: true };
          }
          return session;
        }));
        return;
      }

      // Normal bot message logic
      setSessions(prev => prev.map(session => {
        if (session.id === activeSessionId) {
          const last = session.messages[session.messages.length - 1];
          let newMessages = [...session.messages];

          if (last?.from === "bot") {
            newMessages[newMessages.length - 1] = { ...last, text: last.text + chunk };
          } else {
            newMessages = [...newMessages, { from: "bot", text: chunk }];
          }

          return { ...session, messages: newMessages, isTyping: true };
        }
        return session;
      }));
    };

    socket.onopen = () => {
      // Send provider information first
      socket.send(JSON.stringify({
        provider: activeSession.provider,
        sessionId: activeSessionId
      }));

      console.log("✅ WebSocket connected");

      setSessions(prev => prev.map(session =>
        session.id === activeSessionId
          ? { ...session, isConnected: true, isConnecting: false }
          : session
      ));
    };

    socket.onclose = () => {
      console.log("❌ WebSocket closed");
      setSessions(prev => prev.map(session =>
        session.id === activeSessionId
          ? { ...session, isConnected: false, isConnecting: false }
          : session
      ));
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setSessions(prev => prev.map(session =>
        session.id === activeSessionId
          ? { ...session, isConnecting: false }
          : session
      ));
    };
  };

  const stopAI = () => {
    if (socketRef.current?.readyState === 1) {
      socketRef.current.send("__STOP__");
      setSessions(prev => prev.map(session =>
        session.id === activeSessionId
          ? { ...session, isTyping: false }
          : session
      ));
    }
  };

  const sendMessage = () => {
    if (!activeSession.input.trim() ||
      activeSession.isTyping ||
      socketRef.current?.readyState !== 1) return;

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

    socketRef.current.send(activeSession.input);
  };

  const setInput = (value) => {
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, input: value }
        : session
    ));
  };

  return (
    <div className="app-container">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        setActiveSessionId={setActiveSessionId}
        createNewSession={createNewSession}
        deleteSession={deleteSession} // Pass delete function
        socketRef={socketRef}
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
        <Header />
        {!activeSession.isConnected ? (
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
                {/* <option value="groq">Groq</option> */}
                {/* <option value="ollama">Ollama</option> */}
                <option value="gemini">Gemini</option>
                {/* <option value="openrouter">OpenRouter</option> */}
              </select>
            </div>

            <button
              className={`connect-button ${activeSession.isConnecting ? 'connecting' : ''}`}
              onClick={connectToLLM}
              disabled={activeSession.isConnecting}
            >
              {activeSession.isConnecting ? 'Connecting...' : `Connect to ${activeSession.provider.toUpperCase()}`}
            </button>
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