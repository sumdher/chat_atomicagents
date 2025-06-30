import React, { useState } from "react";
import './Sidebar.css';

export default function Sidebar({ 
  sessions, 
  activeSessionId, 
  setActiveSessionId, 
  createNewSession,
  socketRef, 
  isLoadingContext, 
  setIsLoadingContext 
}) {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [fileMap, setFileMap] = useState({});
  
  const activeSession = sessions.find(session => session.id === activeSessionId) || sessions[0];
  const loadedKeys = activeSession.loadedKeys;

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach(file => formData.append("files", file));

    try {
      const response = await fetch("http://localhost:4580/upload", {
        method: "POST",
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        const keyMap = data.file_map;

        setUploadedFiles(prev => [...prev, ...files]);

        const updatedMap = { ...fileMap };
        for (const [key, name] of Object.entries(keyMap)) {
          updatedMap[key] = name;
        }

        setFileMap(updatedMap);
      } else {
        alert("Failed to upload files");
      }
    } catch (error) {
      console.error("Upload error:", error);
    }
  };

  const handleAddToContext = () => {
    if (socketRef?.current && activeSession.isConnected) {
      setIsLoadingContext(true);
      socketRef.current.send(`__CONTEXT__::${activeSessionId}`);
    }
  };

  const handleDeleteFile = async (key, index) => {
    const formData = new FormData();
    formData.append("file_key", key);

    const response = await fetch("http://localhost:4580/delete-file", {
      method: "POST",
      body: formData
    });

    if (response.ok) {
      const updatedFiles = [...uploadedFiles];
      updatedFiles.splice(index, 1);
      setUploadedFiles(updatedFiles);

      const updatedMap = { ...fileMap };
      delete updatedMap[key];
      setFileMap(updatedMap);
    }
  };

  return (
    <aside className="sidebar">
      <h2 className="sidebar-title">Any Chat</h2>

      {/* Session Management */}
      <div className="session-management">
        <h4>Sessions</h4>
        <div className="session-list">
          {sessions.map(session => (
            <div 
              key={session.id}
              className={`session-item ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(session.id)}
            >
              {session.title}
              {session.id === activeSessionId && <span className="active-indicator">●</span>}
            </div>
          ))}
          <button className="new-session-btn" onClick={createNewSession}>
            + New Session
          </button>
        </div>
      </div>

      <div className="file-upload-box">
        <h4>Files</h4>

        <div className="file-list">
          {uploadedFiles.map((file, idx) => {
            const key = Object.keys(fileMap).find(k => fileMap[k] === file.name);
            const isLoaded = key && loadedKeys.has(key);

            return (
              <div key={idx} className="file-entry" style={{ opacity: isLoaded ? 0.5 : 1 }}>
                <span role="img" aria-label="file">📄</span>
                <div className="file-name" title={file.name}>
                  {file.name.length > 15
                    ? `${file.name.slice(0, 10)}...${file.name.slice(-5)}`
                    : file.name}
                </div>
                {!isLoaded && key && (
                  <button
                    className="delete-btn"
                    onClick={() => handleDeleteFile(key, idx)}
                  >✖</button>
                )}
              </div>
            );
          })}
        </div>

        <input
          id="file-upload"
          type="file"
          multiple
          accept=".txt,.pdf,.doc,.docx"
          style={{ display: 'none' }}
          onChange={handleFileUpload}
        />
        <div className="button-row">
          <label htmlFor="file-upload" className="file-button">
            Add Files
          </label>
          <button
            className="load-button"
            disabled={
              isLoadingContext || 
              uploadedFiles.length === 0 || 
              uploadedFiles.every(file => {
                const key = Object.keys(fileMap).find(k => fileMap[k] === file.name);
                return key && loadedKeys.has(key);
              })
            }
            onClick={handleAddToContext}
          >
            {isLoadingContext ? (
              <>
                <span className="spinner" /> Loading
              </>
            ) : (
              "Load to Context"
            )}
          </button>
        </div>
      </div>
    </aside>
  );
}