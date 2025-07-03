// components/Header.jsx
import React from "react";
import './Header.css';

export default function Header({ title }) {
    return (
        <header className="header">
            <h1 className="header-title">{title || ''}</h1>
        </header>
    );
}