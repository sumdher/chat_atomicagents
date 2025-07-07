const [user, setUser] = useState(null);
const [isAuthenticating, setIsAuthenticating] = useState(true);

// Wrap your App with this
function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if user is logged in on app load
        const checkAuth = async () => {
            try {
                const response = await fetch('/auth/me');
                if (response.ok) {
                    setUser(await response.json());
                }
            } finally {
                setLoading(false);
            }
        };
        checkAuth();
    }, []);

    const login = () => {
        window.location.href = '/auth/google';
    };

    const logout = async () => {
        await fetch('/auth/logout', { method: 'POST' });
        setUser(null);
    };

    if (loading) return <div>Loading...</div>;

    return (
        <AuthContext.Provider value={{ user, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
}