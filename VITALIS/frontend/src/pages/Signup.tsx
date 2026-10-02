import {
  Activity,
  ArrowRight,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  UserRound,
} from "lucide-react";

import {
  useState,
  type FormEvent,
} from "react";

import {
  Link,
  useNavigate,
} from "react-router-dom";

import { signup } from "../api/client";

function getRoleFromEmail(
  email: string,
) {
  const normalized =
    email.trim().toLowerCase();

  if (
    normalized.endsWith(
      "@doctor.vitalis.com",
    )
  ) {
    return "DOCTOR";
  }

  if (
    normalized.endsWith(
      "@nurse.vitalis.com",
    )
  ) {
    return "NURSE";
  }

  return null;
}

export default function Signup() {
  const navigate = useNavigate();

  const [name, setName] =
    useState("");

  const [email, setEmail] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [confirmPassword, setConfirmPassword] =
    useState("");

  const [showPassword, setShowPassword] =
    useState(false);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const role =
    getRoleFromEmail(email);

  async function handleSubmit(
    event: FormEvent,
  ) {
    event.preventDefault();

    setError("");

    if (!role) {
      setError(
        "Use a @doctor.vitalis.com or @nurse.vitalis.com email address.",
      );
      return;
    }

    if (password.length < 8) {
      setError(
        "Password must contain at least 8 characters.",
      );
      return;
    }

    if (
      password !== confirmPassword
    ) {
      setError(
        "Passwords do not match.",
      );
      return;
    }

    setLoading(true);

    try {
      await signup({
        name: name.trim(),
        email: email.trim(),
        password,
      });

      navigate("/login", {
        replace: true,
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to create account.",
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
              CLINICAL ACCESS
            </span>

            <h1>
              Create your account
            </h1>

            <p>
              Your email domain determines
              your VITALIS role.
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
              Full name

              <div className="input-wrapper">
                <UserRound size={17} />

                <input
                  value={name}
                  onChange={(event) =>
                    setName(
                      event.target.value,
                    )
                  }
                  placeholder="Your full name"
                  required
                />
              </div>
            </label>

            <label>
              Clinical email

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
                  placeholder="you@doctor.vitalis.com"
                  required
                />
              </div>

              <span className="field-help">
                Doctor:
                @doctor.vitalis.com
                {" • "}
                Nurse:
                @nurse.vitalis.com
              </span>
            </label>

            {role && (
              <div className="detected-role">
                <span>
                  Detected role
                </span>

                <strong>
                  {role}
                </strong>
              </div>
            )}

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
                  placeholder="Minimum 8 characters"
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

            <label>
              Confirm password

              <div className="input-wrapper">
                <LockKeyhole size={17} />

                <input
                  type={
                    showPassword
                      ? "text"
                      : "password"
                  }
                  value={
                    confirmPassword
                  }
                  onChange={(event) =>
                    setConfirmPassword(
                      event.target.value,
                    )
                  }
                  placeholder="Repeat your password"
                  required
                />
              </div>
            </label>

            <button
              className="auth-submit"
              disabled={loading}
            >
              {loading
                ? "Creating..."
                : "Create account"}

              {!loading && (
                <ArrowRight size={17} />
              )}
            </button>
          </form>

          <div className="auth-footer">
            Already have an account?

            <Link to="/login">
              Sign in
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
