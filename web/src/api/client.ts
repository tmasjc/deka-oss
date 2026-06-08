export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function formatApiError(err: unknown): string {
  if (err instanceof ApiError) {
    switch (err.status) {
      case 401: return "Sign in required — redirecting to login.";
      case 403: return "This session belongs to another user.";
      case 422: return "Query is not covered by the corpus — try rephrasing.";
      case 503: return "Search service unavailable — is Milvus running?";
      case 409: return "Session state conflict — please refresh.";
      case 404: {
        // Prefer the server's detail (e.g. "No reflection recorded for
        // the last turn", "No turns completed yet"). Only fall back to
        // the "session gone" copy when the server gave nothing beyond
        // the bare status line — otherwise we mask precise endpoint
        // messages with a misleading "start a new session" prompt.
        const bareStatus = `${err.status} Not Found`;
        if (err.message && err.message !== bareStatus) return err.message;
        return "Session no longer exists — please start a new one.";
      }
      default:  return err.message;
    }
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

// Module-level callback that the App registers at boot to react to a
// 401 (typically: clear the session-id state, route to /login). Kept
// out of the request closure so we don't have to thread it through
// every call site.
let onUnauthorized: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
    // Required for the SessionMiddleware cookie to flow on every
    // /api/* call (including cross-origin in dev where the Vite
    // proxy rewrites the host).
    credentials: "include",
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data && typeof data.detail === "string") detail = data.detail;
    } catch {
      /* ignore parse errors */
    }
    if (res.status === 401 && onUnauthorized !== null && !path.startsWith("/api/auth/")) {
      // Fire-and-forget — the handler routes to /login and clears
      // React Query caches. We still throw so caller-level error
      // handling stays correct. Skip for /api/auth/* so the auth
      // probe's expected 401 does not invalidate its own cache key
      // and refetch in a loop.
      onUnauthorized();
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const http = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body),
  del: <T>(path: string) => request<T>("DELETE", path),
};

export async function downloadBlob(path: string, filename: string): Promise<void> {
  const res = await fetch(path, { credentials: "include" });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data && typeof data.detail === "string") detail = data.detail;
    } catch {
      /* ignore parse errors */
    }
    if (res.status === 401 && onUnauthorized !== null && !path.startsWith("/api/auth/")) onUnauthorized();
    throw new ApiError(res.status, detail);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
