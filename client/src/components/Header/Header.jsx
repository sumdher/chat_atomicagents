// Header.jsx
import React from "react";
import './Header.css';

export default function Header({
    title,
    isSidebarCollapsed,
    toggleSidebar
}) {
    return (
        <header className="header">
            {isSidebarCollapsed && (
                <button className="expand-btn" onClick={toggleSidebar}>
                    →
                </button>
            )}
            <h1 className="header-title">{title || ''}</h1>
        </header>
    );
}