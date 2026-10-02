import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import {
  getMe,
  loadStoredUser,
  login as apiLogin,
  logout as apiLogout,
  saveAuth,
  type User,
} from "../api/client";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (
    email: string,
    password: string,
  ) => Promise<User>;
  logout: () => void;
}

const AuthContext =
  createContext<AuthContextValue | null>(null);

export function AuthProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [user, setUser] =
    useState<User | null>(loadStoredUser());

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function validateSession() {
      const token =
        localStorage.getItem("vitalis_token");

      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const currentUser = await getMe();

        setUser(currentUser);

        localStorage.setItem(
          "vitalis_user",
          JSON.stringify(currentUser),
        );
      } catch {
        apiLogout();
        setUser(null);
      } finally {
        setLoading(false);
      }
    }

    validateSession();

    const handleLogout = () => {
      setUser(null);
    };

    window.addEventListener(
      "vitalis:logout",
      handleLogout,
    );

    return () => {
      window.removeEventListener(
        "vitalis:logout",
        handleLogout,
      );
    };
  }, []);

  async function login(
    email: string,
    password: string,
  ) {
    const result = await apiLogin(
      email,
      password,
    );

    saveAuth(
      result.access_token,
      result.user,
    );

    setUser(result.user);

    return result.user;
  }

  function logout() {
    apiLogout();
    setUser(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error(
      "useAuth must be used inside AuthProvider",
    );
  }

  return context;
}
