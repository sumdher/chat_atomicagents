// Header.jsx
import React from 'react';
import expandIcon from '../../assets/expand.svg'
import './Header.css';

export default function Header({
    title,
    isSidebarCollapsed,
    toggleSidebar,
    onApiKeysClick // Add this prop
}) {
    return (
        <header className="header">
            {isSidebarCollapsed && (
                <button className="expand-btn" onClick={toggleSidebar}>
                    <img src={expandIcon} alt="expand" className="expand-icon" />
                </button>
            )}
            <h1 className="header-title">{title || ''}</h1>
            <button
                className="api-keys-btn"
                onClick={onApiKeysClick}
            >
                API Keys
            </button>
        </header>
    );
}