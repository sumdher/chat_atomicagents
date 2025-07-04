// Sidebar.jsx
import React, { useState } from "react";
import './Sidebar.css';

export default function Sidebar({
  sessions,
  activeSessionId,
  setActiveSessionId,
  createNewSession,
  deleteSession,
  isCollapsed,
  toggleCollapse,
}) {
 

  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <h2 className="sidebar-title">Any Chat</h2>
        <button className="collapse-btn" onClick={toggleCollapse}>
          ←
        </button>
      </div>

      <div className="session-management">
        <div className="session-header">
          <h4>Sessions</h4>
          <button className="new-session-btn" onClick={createNewSession}>
            + New Session
          </button>
        </div>
        <div className="session-list">
          {sessions.map(session => (
            <div
              key={session.id}
              className={`session-item ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(session.id)}
            >
              <span className="session-title">{session.title}</span>
              {sessions.length > 1 && ( // Only show delete if multiple sessions
                <button
                  className="delete-session-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(session.id);
                  }}
                >
                  X
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
