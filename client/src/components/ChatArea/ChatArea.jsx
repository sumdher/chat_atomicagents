// components/ChatArea.jsx
import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from './CodeBlock';
import './ChatArea.css';

export default function ChatArea({ messages, isTyping, isLoadingContext, onEditMessage }) {
    const [editingIndex, setEditingIndex] = useState(-1);
    const [editText, setEditText] = useState('');
    const [longMessages, setLongMessages] = useState({});

    const startEditing = (index, text) => {
        setEditingIndex(index);
        setEditText(text);
    };

    const cancelEditing = () => {
        setEditingIndex(-1);
        setEditText('');
    };

    const saveEdit = () => {
        if (editText.trim() && editingIndex >= 0) {
            onEditMessage(editingIndex, editText);
            cancelEditing();
        }
    };
    const messageRefs = useRef([]);

    useEffect(() => {
        const newLongMessages = {};
        messageRefs.current.forEach((ref, index) => {
            if (ref && messages[index]?.from === 'user') {
                const lineHeight = parseInt(getComputedStyle(ref).lineHeight);
                const height = ref.clientHeight;
                newLongMessages[index] = height > lineHeight * 10;
            }
        });
        setLongMessages(newLongMessages);
    }, [messages]);

    return (
        <div className="chat-area">
            <div className="messages-wrapper">
                <div className="messages">
                    {messages.length === 0 && (
                        <div className="message-row bot">
                            <div className="message-bubble bot welcome-message">
                                <ReactMarkdown>
                                    {"**Connected**\n\nChat started"}
                                </ReactMarkdown>
                            </div>
                        </div>
                    )}
                    {messages.map((msg, i) => (
                        <div key={i} className={`message-row ${msg.from}`}>
                            {msg.from === "user" && editingIndex === i ? (
                                <div className="message-edit-container">
                                    <textarea
                                        value={editText}
                                        onChange={(e) => setEditText(e.target.value)}
                                        autoFocus
                                        className="message-edit-input"
                                    />
                                    <div className="message-edit-buttons">
                                        <button onClick={saveEdit} className="save-edit">Save</button>
                                        <button onClick={cancelEditing} className="cancel-edit">Cancel</button>
                                    </div>
                                </div>
                            ) : (
                                <div className={`message-bubble ${msg.from}`}>
                                    {msg.from === "user" && (
                                        <div className="edit-button-container">
                                            {longMessages[i] && (
                                                <button
                                                    className="edit-message-btn top"
                                                    onClick={() => startEditing(i, msg.text)}
                                                    title="Edit message"
                                                >
                                                    ✏️
                                                </button>
                                            )}
                                            <button
                                                className="edit-message-btn bottom"
                                                onClick={() => startEditing(i, msg.text)}
                                                title="Edit message"
                                            >
                                                ✏️
                                            </button>
                                        </div>
                                    )}
                                    {msg.from === "bot" ? (
                                        <ReactMarkdown
                                            components={{
                                                code({ inline, className, children }) {
                                                    const match = /language-(\w+)/.exec(className || "");
                                                    const codeString = String(children).trim();
                                                    return !inline && match ? (
                                                        <CodeBlock language={match[1]} value={codeString} />
                                                    ) : (
                                                        <code className={className}>{codeString}</code>
                                                    );
                                                }
                                            }}
                                        >
                                            {msg.text}
                                        </ReactMarkdown>
                                    ) : (
                                        <div
                                            ref={el => messageRefs.current[i] = el}
                                            className="user-message-content"
                                        >
                                            {msg.text}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoadingContext && (
                        <div className="message-row bot">
                            <div className="message-bubble bot">
                                loading files to context...
                            </div>
                        </div>
                    )}
                    {isTyping && <div className="typing">thinking...</div>}
                </div>
            </div>
        </div>
    );
}