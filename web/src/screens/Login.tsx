import { useEffect, useRef, useState } from "react";
import { ApiError, formatApiError } from "../api/client";
import { useLogin } from "../hooks/useAuth";

/** Token-entry screen.
 *
 * Single password input; submits to /api/auth/login, surfaces a 401
 * inline. On success the cookie is set server-side and the parent
 * App's useMe query refetches and unmounts this screen.
 */
export function Login() {
  const login = useLogin();
  const [token, setToken] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!token.trim()) return;
    login.mutate(token.trim());
  }

  const error =
    login.error instanceof ApiError && login.error.status === 401
      ? "Token not recognized — check the value with whoever invited you."
      : login.error
      ? formatApiError(login.error)
      : null;

  return (
    <div
      className="app"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
      }}
    >
      <form
        onSubmit={onSubmit}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          minWidth: 360,
          padding: "2rem",
          border: "1px solid var(--border, #d8d8d8)",
          borderRadius: 8,
          background: "var(--surface, #fff)",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "1.25rem" }}>Sign in</h1>
        <p style={{ margin: 0, color: "var(--muted, #666)", fontSize: 14 }}>
          Paste the access token your operator sent you.
        </p>
        <input
          ref={inputRef}
          type="password"
          autoComplete="current-password"
          placeholder="Access token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          disabled={login.isPending}
          style={{
            padding: "0.5rem 0.75rem",
            fontSize: "1rem",
            fontFamily: "var(--font-mono, monospace)",
            border: "1px solid var(--border, #d8d8d8)",
            borderRadius: 4,
          }}
        />
        {error && (
          <div
            role="alert"
            style={{
              fontSize: 13,
              color: "var(--danger, #c33)",
            }}
          >
            {error}
          </div>
        )}
        <button
          type="submit"
          disabled={!token.trim() || login.isPending}
          style={{
            padding: "0.5rem 0.75rem",
            fontSize: "1rem",
            cursor:
              !token.trim() || login.isPending ? "not-allowed" : "pointer",
          }}
        >
          {login.isPending ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
