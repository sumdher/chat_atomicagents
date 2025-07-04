// components/InputArea.jsx
import React, { useRef, useEffect } from "react";
import "./InputArea.css";

export default function InputArea({
    input,
    setInput,
    sendMessage,
    stopAI,
    isTyping,
}) {
    const textareaRef = useRef(null);

    useEffect(() => {
        const t = textareaRef.current;
        if (!t) return;
        t.style.height = "auto";
        t.style.height = `${t.scrollHeight + 2}px`;
    }, [input]);

    return (
        <div className="input-area">
            <div className="input-inner">
                <textarea
                    ref={textareaRef}
                    className="input-field"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => {
                        if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    }}
                    placeholder="Type your message…"
                />
                <button
                    className="send-button"
                    onClick={isTyping ? stopAI : sendMessage}
                    disabled={!input.trim() && !isTyping}
                >
                    {isTyping ? "Stop" : "Send"}
                </button>
            </div>
        </div>
    );
}
