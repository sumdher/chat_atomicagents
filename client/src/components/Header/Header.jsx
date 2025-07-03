// components/Header.jsx
import React from "react";
import './Header.css';

export default function Header({ modelName }) {
    return (
        <header className="header">
            <h1 className="header-title">{modelName || ''}</h1>
        </header>
    );
}