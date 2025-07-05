// ApiKeysModal.jsx
import React, { useState } from 'react';
import './ApiKeysModal.css';

export default function ApiKeysModal({
    apiKeys,
    onClose,
    onAddKey,
    onDeleteKey
}) {
    const [newProvider, setNewProvider] = useState('');
    const [newKey, setNewKey] = useState('');
    const [isAdding, setIsAdding] = useState(false);
    const [error, setError] = useState('');

    const handleAddClick = () => {
        if (!newProvider.trim() || !newKey.trim()) {
            setError('Both provider and key are required');
            return;
        }

        onAddKey(newProvider.trim(), newKey.trim());
        setNewProvider('');
        setNewKey('');
        setIsAdding(false);
        setError('');
    };

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h2>Manage API Keys</h2>
                    <button className="close-btn" onClick={onClose}>
                        &times;
                    </button>
                </div>

                <div className="keys-list">
                    {Object.entries(apiKeys).map(([provider, key]) => (
                        <div key={provider} className="key-item">
                            <div className="key-info">
                                <span className="provider">{provider}</span>
                                <span className="key-value">{key ? '••••••••' : 'Not set'}</span>
                            </div>
                            <button
                                className="delete-btn"
                                onClick={() => onDeleteKey(provider)}
                            >
                                Delete
                            </button>
                        </div>
                    ))}
                </div>

                {isAdding ? (
                    <div className="add-key-form">
                        <div className="input-group">
                            <input
                                type="text"
                                value={newProvider}
                                onChange={(e) => setNewProvider(e.target.value)}
                                placeholder="Provider (e.g., openai)"
                                autoFocus
                            />
                            <input
                                type="password"
                                value={newKey}
                                onChange={(e) => setNewKey(e.target.value)}
                                placeholder="API Key"
                            />
                        </div>
                        {error && <p className="error">{error}</p>}
                        <div className="form-actions">
                            <button onClick={handleAddClick}>Save</button>
                            <button onClick={() => setIsAdding(false)}>Cancel</button>
                        </div>
                    </div>
                ) : (
                    <button
                        className="add-btn"
                        onClick={() => setIsAdding(true)}
                    >
                        + Add New Key
                    </button>
                )}

                <div className="modal-footer">
                    <p>API keys are securely stored in your local environment</p>
                </div>
            </div>
        </div>
    );
}