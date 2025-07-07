// src/components/AuthContext/AuthContext.jsx
import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();
const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:4580';

export default function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const checkAuth = async () => {
            try {
                const response = await fetch(`${backendUrl}/auth/me`, {
                    credentials: 'include'
                });
                
                if (response.ok) {
                    setUser(await response.json());
                }
            } catch (error) {
                console.error('Auth check failed:', error);
            } finally {
                setLoading(false);
            }
        };
        checkAuth();
    }, []);

    const login = () => {
        window.location.href = `${backendUrl}/auth/google`;
    };

    const logout = async () => {
        try {
            await fetch(`${backendUrl}/auth/logout`, {
                method: 'POST',
                credentials: 'include'
            });
            setUser(null);
        } catch (error) {
            console.error('Logout failed:', error);
        }
      };

    const value = { user, login, logout, loading };

    return (
        <AuthContext.Provider value={value}>
            {loading ? <div className="app-loading">Loading...</div> : children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}