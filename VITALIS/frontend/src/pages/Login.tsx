import {
  Activity,
  ArrowRight,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  ShieldCheck,
} from "lucide-react";

import {
  useState,
  type FormEvent,
} from "react";

import {
  Link,
  useNavigate,
} from "react-router-dom";

import { useAuth } from "../auth/AuthContext";

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [showPassword, setShowPassword] =
    useState(false);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  async function handleSubmit(
    event: FormEvent,
  ) {
    event.preventDefault();

    setError("");
    setLoading(true);

    try {
      await login(
        email.trim(),
        password,
      );

      navigate(
        "/dashboard",
        { replace: true },
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to sign in.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-background" />

      <div className="auth-container">
        <div className="auth-brand">
          <div className="auth-brand-mark">
            <Activity size={24} />
          </div>

          <div>
            <strong>VITALIS</strong>
            <span>
              Clinical Intelligence
            </span>
          </div>
        </div>

        <div className="auth-card">
          <div className="auth-card-heading">
            <span className="eyebrow">
              ICU COMMAND CENTER
            </span>

            <h1>
              Welcome back
            </h1>

            <p>
              Sign in to access your
              clinical risk dashboard.
            </p>
          </div>

          {error && (
            <div className="auth-error">
              {error}
            </div>
          )}

          <form
            className="auth-form"
            onSubmit={handleSubmit}
          >
            <label>
              Email address

              <div className="input-wrapper">
                <Mail size={17} />

                <input
                  type="email"
                  value={email}
                  onChange={(event) =>
                    setEmail(
                      event.target.value,
                    )
                  }
                  placeholder="name@doctor.vitalis.com"
                  autoComplete="email"
                  required
                />
              </div>
            </label>

            <label>
              Password

              <div className="input-wrapper">
                <LockKeyhole size={17} />

                <input
                  type={
                    showPassword
                      ? "text"
                      : "password"
                  }
                  value={password}
                  onChange={(event) =>
                    setPassword(
                      event.target.value,
                    )
                  }
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  required
                />

                <button
                  type="button"
                  className="password-toggle"
                  onClick={() =>
                    setShowPassword(
                      (value) => !value,
                    )
                  }
                >
                  {showPassword ? (
                    <EyeOff size={16} />
                  ) : (
                    <Eye size={16} />
                  )}
                </button>
              </div>
            </label>

            <button
              className="auth-submit"
              disabled={loading}
            >
              {loading
                ? "Signing in..."
                : "Sign in"}

              {!loading && (
                <ArrowRight size={17} />
              )}
            </button>
          </form>

          <div className="auth-security">
            <ShieldCheck size={16} />

            <span>
              Secure clinical access
            </span>
          </div>

          <div className="auth-footer">
            Don't have an account?

            <Link to="/signup">
              Create account
            </Link>
          </div>
        </div>

        <div className="auth-note">
          VITALIS • ICU Clinical Risk
          Intelligence
        </div>
      </div>
    </div>
  );
}
