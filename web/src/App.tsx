import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { setUnauthorizedHandler } from "./api/client";
import { useUi } from "./state/ui";
import { useMe, useLogout } from "./hooks/useAuth";
import { Login } from "./screens/Login";
import { SessionList } from "./screens/SessionList";
import { RatingScreen } from "./screens/Rating";

export function App() {
  const sid = useUi((s) => s.sessionId);
  const setSession = useUi((s) => s.setSession);
  const me = useMe();
  const logout = useLogout();
  const qc = useQueryClient();

  // Wire the API client's 401 handler exactly once. On unauthorized
  // we drop every cached query (so a re-login as a different user
  // doesn't see the previous user's data) and route back to "/".
  // useMe's failed refetch then renders <Login>.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      qc.clear();
      setSession(null);
      if (window.location.pathname !== "/") {
        window.history.pushState(null, "", "/");
      }
    });
    return () => setUnauthorizedHandler(null);
  }, [qc, setSession]);

  if (me.isLoading) {
    return (
      <div className="theme-warm" style={{ height: "100%" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            color: "var(--muted, #666)",
          }}
        >
          Loading…
        </div>
      </div>
    );
  }

  if (me.isError || !me.data) {
    return (
      <div className="theme-warm" style={{ height: "100%" }}>
        <Login />
      </div>
    );
  }

  return (
    <div className="theme-warm" style={{ height: "100%" }}>
      {sid ? <RatingScreen sid={sid} /> : <SessionList />}
      <SignedInBadge
        userId={me.data.user_id}
        onLogout={() => {
          logout.mutate(undefined, {
            onSuccess: () => {
              setSession(null);
              window.history.pushState(null, "", "/");
              // Force the auth probe to re-evaluate and render <Login>.
              qc.invalidateQueries({ queryKey: ["auth", "me"] });
            },
          });
        }}
        busy={logout.isPending}
      />
    </div>
  );
}

function SignedInBadge({
  userId,
  onLogout,
  busy,
}: {
  userId: string;
  onLogout: () => void;
  busy: boolean;
}) {
  return (
    <div
      style={{
        position: "fixed",
        bottom: 8,
        right: 8,
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 8px",
        background: "var(--surface, rgba(255,255,255,0.9))",
        border: "1px solid var(--border, #d8d8d8)",
        borderRadius: 4,
        fontSize: 12,
        color: "var(--muted, #666)",
        zIndex: 1000,
      }}
    >
      <span>signed in as {userId}</span>
      <button
        type="button"
        onClick={onLogout}
        disabled={busy}
        style={{
          fontSize: 11,
          padding: "1px 6px",
          cursor: busy ? "not-allowed" : "pointer",
        }}
      >
        sign out
      </button>
    </div>
  );
}
