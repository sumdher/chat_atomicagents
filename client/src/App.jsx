import { useRef, useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import Sidebar from "./components/Sidebar/Sidebar";
import Header from "./components/Header/Header";
import ChatArea from "./components/ChatArea/ChatArea";
import InputArea from "./components/InputArea/InputArea";
import './App.css';

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
  const [isInitialized, setIsInitialized] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const socketRef = useRef(null);

  // Initialize sessions from localStorage
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

  // Save sessions & activeSessionId to localStorage on change
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

  // Create new session and init on server
  const createNewSession = () => {
    const currentProvider = activeSession.provider;
    const newSession = {
      ...createDefaultSession(),
      provider: currentProvider,
    };

    setSessions(prev => [...prev, newSession]);
    setActiveSessionId(newSession.id);

    // Send init message for new session when activeSessionId changes (handled by useEffect)
  };

  // Delete a session
  const deleteSession = (id) => {
    if (sessions.length <= 1) return; // Prevent deleting last session

    setSessions(prev => prev.filter(session => session.id !== id));

    if (id === activeSessionId) {
      const newActive = sessions.find(s => s.id !== id);
      setActiveSessionId(newActive.id);
    }
  };

  // When user switches active session, send init for that session
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

  // Change provider of active session and re-init connection for that session
  const setSessionProvider = (provider) => {
    setSessions(prev => prev.map(session =>
      session.id === activeSessionId
        ? { ...session, provider }
        : session
    ));

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        type: "init",
        sessionId: activeSessionId,
        provider,
      }));
      setSessions(prev => prev.map(session =>
        session.id === activeSessionId
          ? { ...session, isConnecting: true }
          : session
      ));
    }
  };

  // Send user message for active session
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
                <option value="gemini">Gemini</option>
                {/* Add other providers as needed */}
              </select>
            </div>

            <button
              className={`connect-button ${activeSession.isConnecting ? 'connecting' : ''}`}
              onClick={() => {
                // On connect button click, send init message again
                if (socketRef.current?.readyState === WebSocket.OPEN) {
                  socketRef.current.send(JSON.stringify({
                    type: "init",
                    sessionId: activeSessionId,
                    provider: activeSession.provider,
                  }));
                  setSessions(prev => prev.map(session =>
                    session.id === activeSessionId
                      ? { ...session, isConnecting: true }
                      : session
                  ));
                }
              }}
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
