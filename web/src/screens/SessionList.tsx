import { useEffect, useRef, useState } from "react";
import { ApiError, downloadBlob, formatApiError } from "../api/client";
import { ElapsedTimer } from "../components/ElapsedTimer";
import { LoadingHelix } from "../components/LoadingHelix";
import { QueryEntry } from "./QueryEntry";
import {
  useSessions,
  useResume,
  useDiscardSession,
  useStartReplay,
} from "../hooks/useSessions";
import type { ResumeTarget, SessionListItem } from "../types";

/** Post-login landing page.
 *
 * Two states:
 * - "Start new" — collapses into the existing :class:`QueryEntry`
 *   screen. The button is the default action when the listing is
 *   empty (first-time user).
 * - "Resume" — a list of the user's prior sessions. Clicking a row
 *   POSTs ``/api/session/<sid>/resume`` and routes to ``/s/<sid>``.
 *
 * Sessions classified as ``DONE_VIEW`` / ``POST_HARVEST`` /
 * ``POST_RUBRIC`` still appear in the list (they exist on disk!),
 * but their resume call currently returns 501 — full hydration is
 * a follow-up. The UI surfaces the 501 message inline so the user
 * understands why the click didn't take them anywhere.
 */
export function SessionList() {
  const sessions = useSessions();
  const resume = useResume();
  const discard = useDiscardSession();
  const replay = useStartReplay();
  const [showStartNew, setShowStartNew] = useState(false);

  if (showStartNew) {
    return <QueryEntry />;
  }

  if (sessions.isLoading) {
    return (
      <div className="app">
        <div style={{ padding: "2rem", color: "var(--muted, #666)" }}>
          Loading your sessions…
        </div>
      </div>
    );
  }

  if (sessions.isError) {
    return (
      <div className="app">
        <div
          style={{
            padding: "2rem",
            color: "var(--danger, #c33)",
          }}
        >
          Could not load sessions: {formatApiError(sessions.error)}
        </div>
      </div>
    );
  }

  const items = sessions.data ?? [];
  const resumeError =
    resume.error instanceof ApiError
      ? formatApiError(resume.error)
      : resume.error
      ? formatApiError(resume.error)
      : null;
  const discardError = discard.error ? formatApiError(discard.error) : null;
  const replayError = replay.error ? formatApiError(replay.error) : null;
  const rowError = resumeError ?? discardError ?? replayError;

  if (items.length === 0) {
    return <QueryEntry />;
  }

  return (
    <div
      className="app"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1rem",
        padding: "2rem",
        maxWidth: 800,
        margin: "0 auto",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "1.4rem" }}>Your sessions</h1>
        <button
          type="button"
          onClick={() => setShowStartNew(true)}
          style={{ padding: "0.4rem 0.9rem" }}
        >
          + Start new
        </button>
      </header>

      {rowError && (
        <div
          role="alert"
          style={{
            padding: "0.5rem 0.75rem",
            background: "var(--surface-warn, #fff5f0)",
            border: "1px solid var(--border, #d8d8d8)",
            borderRadius: 4,
            fontSize: 13,
            color: "var(--danger, #a33)",
          }}
        >
          {rowError}
        </div>
      )}

      <ul
        style={{
          listStyle: "none",
          padding: 0,
          margin: 0,
          display: "flex",
          flexDirection: "column",
          gap: "0.5rem",
        }}
      >
        {items.map((row) => (
          <SessionRow
            key={row.session_id}
            row={row}
            onClick={() => resume.mutate(row.session_id)}
            onDiscard={() => {
              const sid8 = row.session_id.slice(0, 8);
              const ok = window.confirm(
                `Discard session ${sid8}? This permanently deletes all logs and cannot be undone.`,
              );
              if (ok) discard.mutate(row.session_id);
            }}
            onReplay={() => replay.mutate(row.session_id)}
            disabled={resume.isPending || discard.isPending || replay.isPending}
          />
        ))}
      </ul>
    </div>
  );
}

function SessionRow({
  row,
  onClick,
  onDiscard,
  onReplay,
  disabled,
}: {
  row: SessionListItem;
  onClick: () => void;
  onDiscard: () => void;
  onReplay: () => void;
  disabled: boolean;
}) {
  return (
    <li
      style={{
        display: "flex",
        alignItems: "stretch",
        background: "var(--surface, #fff)",
        border: "1px solid var(--border, #d8d8d8)",
        borderRadius: 6,
        position: "relative",
      }}
    >
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          flex: 1,
          minWidth: 0,
          padding: "0.75rem",
          textAlign: "left",
          background: "transparent",
          border: "none",
          cursor: disabled ? "wait" : "pointer",
          font: "inherit",
        }}
      >
        <ResumeBadge target={row.resume_target} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              fontWeight: 500,
            }}
          >
            {row.query || <em style={{ color: "var(--muted, #888)" }}>no query</em>}
          </div>
          <div
            style={{
              fontSize: 12,
              color: "var(--muted, #888)",
              marginTop: 2,
            }}
          >
            {row.scope ?? "Unknown scope"} · {stageDetail(row)} ·{" "}
            <RelTime iso={row.last_modified} />
          </div>
        </div>
        <code
          style={{ fontSize: 11, color: "var(--muted, #888)" }}
          title={row.session_id}
        >
          {row.session_id.slice(0, 12)}
        </code>
      </button>
      <RowMenu
        sessionId={row.session_id}
        hasRubric={row.has_rubric}
        hasArtifacts={row.has_artifacts}
        nTurns={row.n_turns}
        onDiscard={onDiscard}
        onReplay={onReplay}
        disabled={disabled}
      />
    </li>
  );
}

function RowMenu({
  sessionId,
  hasRubric,
  hasArtifacts,
  nTurns,
  onDiscard,
  onReplay,
  disabled,
}: {
  sessionId: string;
  hasRubric: boolean;
  hasArtifacts: boolean;
  nTurns: number;
  onDiscard: () => void;
  onReplay: () => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [artifactsBusy, setArtifactsBusy] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) {
      setCopied(false);
      return;
    }
    const onMouseDown = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const sid8 = sessionId.slice(0, 8);

  const handleDownloadArtifacts = async () => {
    if (!hasArtifacts || artifactsBusy) return;
    setOpen(false);
    setArtifactsBusy(true);
    try {
      await downloadBlob(
        `/api/session/${sessionId}/artifacts/download`,
        `session-${sid8}-artifacts.zip`,
      );
    } catch (err) {
      // No toast infra yet — log so the user at least gets something
      // when the build fails (e.g. 422 missing scope, 5xx Postgres).
      console.error("artifacts download failed", err);
    } finally {
      setArtifactsBusy(false);
    }
  };

  const handleDownloadRubric = () => {
    if (!hasRubric) return;
    setOpen(false);
    window.location.href = `/api/session/${sessionId}/rubric/download`;
  };

  const handleCopyId = () => {
    void navigator.clipboard?.writeText(sessionId);
    setCopied(true);
    window.setTimeout(() => setOpen(false), 900);
  };

  const handleDiscard = () => {
    setOpen(false);
    onDiscard();
  };

  const canReplay = nTurns > 0;
  const handleReplay = () => {
    if (!canReplay) return;
    setOpen(false);
    onReplay();
  };

  return (
    <div ref={containerRef} style={{ position: "relative", display: "flex" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Actions for session ${sid8}`}
        title="More actions"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "0 0.75rem",
          background: "transparent",
          border: "none",
          color: "var(--muted, #888)",
          cursor: disabled ? "wait" : "pointer",
          font: "inherit",
        }}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <circle cx="12" cy="5" r="1.7" />
          <circle cx="12" cy="12" r="1.7" />
          <circle cx="12" cy="19" r="1.7" />
        </svg>
      </button>
      {open && (
        <div
          role="menu"
          style={{
            position: "absolute",
            top: "100%",
            right: 0,
            marginTop: 4,
            minWidth: 220,
            background: "var(--surface, #fff)",
            border: "1px solid var(--border, #d8d8d8)",
            borderRadius: 6,
            boxShadow: "0 8px 24px rgba(0, 0, 0, 0.12)",
            padding: "4px 0",
            zIndex: 10,
            fontSize: 13,
          }}
        >
          <MenuItem
            onClick={handleDownloadArtifacts}
            icon={<DownloadIcon />}
            disabled={!hasArtifacts}
            title={hasArtifacts ? undefined : "No artifacts yet"}
          >
            Download artifacts
          </MenuItem>
          <MenuItem
            onClick={handleDownloadRubric}
            icon={<DocumentIcon />}
            disabled={!hasRubric}
            title={hasRubric ? undefined : "Rubric not yet generated"}
          >
            Download rubric
          </MenuItem>
          <MenuItem
            onClick={handleCopyId}
            icon={copied ? <CheckIcon /> : <ClipboardIcon />}
            flash={copied}
          >
            {copied ? "Copied!" : "Copy session id"}
          </MenuItem>
          <MenuItem
            onClick={handleReplay}
            icon={<ReplayIcon />}
            disabled={!canReplay}
            title={canReplay ? undefined : "Session has no Phase 1 turns yet"}
          >
            Replay session
          </MenuItem>
          <div
            style={{
              height: 1,
              background: "var(--border, #d8d8d8)",
              margin: "4px 0",
            }}
          />
          <MenuItem onClick={handleDiscard} icon={<TrashIcon />} danger>
            Discard
          </MenuItem>
        </div>
      )}
      {artifactsBusy && (
        <div className="turn-overlay" role="status" aria-live="polite">
          <LoadingHelix />
          <p className="turn-overlay__text">Preparing artifacts…</p>
          <ElapsedTimer />
          <p className="turn-overlay__detail">
            Binding chunks from Postgres. The browser will download the
            zip when the bundle is ready.
          </p>
        </div>
      )}
    </div>
  );
}

function MenuItem({
  onClick,
  icon,
  children,
  disabled = false,
  danger = false,
  flash = false,
  title,
}: {
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
  disabled?: boolean;
  danger?: boolean;
  flash?: boolean;
  title?: string;
}) {
  const color = disabled
    ? "var(--muted, #aaa)"
    : flash
    ? "var(--pos, #2c6a1f)"
    : danger
    ? "var(--danger, #c33)"
    : "var(--bg, #111)";
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      aria-disabled={disabled}
      title={title}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        width: "100%",
        padding: "8px 12px",
        background: flash ? "rgba(143, 196, 154, 0.14)" : "transparent",
        border: "none",
        textAlign: "left",
        color,
        cursor: disabled ? "not-allowed" : "pointer",
        font: "inherit",
        transition: "color 160ms ease, background 160ms ease",
      }}
    >
      <span
        aria-hidden="true"
        style={{ display: "inline-flex", color, opacity: disabled ? 0.6 : 1 }}
      >
        {icon}
      </span>
      <span>{children}</span>
    </button>
  );
}

function DownloadIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function DocumentIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function ClipboardIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="9" y="2" width="6" height="4" rx="1" />
      <path d="M9 4H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-2" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function ReplayIcon() {
  // Counter-clockwise arc with an arrow — a "rewind / time-travel"
  // affordance that pairs visually with the other monochrome menu icons.
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="1 4 1 10 7 10" />
      <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  );
}

// Secondary line under the query. The "turn N" wording only applies
// to POST_TUNING (phase-1 turns); for later stages the count is
// misleading because the session has progressed past tuning, so we
// substitute a stage-aware phrase instead.
function stageDetail(row: SessionListItem): string {
  switch (row.resume_target) {
    case "POST_TUNING":
      return `turn ${row.n_turns}`;
    case "POST_HARVEST":
      return "harvest ready";
    case "POST_RUBRIC":
      return "rubric pending review";
    case "APPLY_PENDING":
      return "ready to apply";
    case "DONE_VIEW":
      return "shipped";
    default:
      return `turn ${row.n_turns}`;
  }
}

const BADGE_STYLE: Record<ResumeTarget, { bg: string; fg: string; label: string }> = {
  POST_TUNING: {
    bg: "#e7f0ff",
    fg: "#1a4ea0",
    label: "Tuning",
  },
  POST_HARVEST: {
    bg: "#e9f5e1",
    fg: "#2c6a1f",
    label: "Harvest",
  },
  POST_RUBRIC: {
    bg: "#fbf3d4",
    fg: "#7a5b00",
    label: "Review",
  },
  APPLY_PENDING: {
    bg: "#d8efe7",
    fg: "#1a6452",
    label: "Ship",
  },
  DONE_VIEW: {
    bg: "#ece9f9",
    fg: "#4d3aa1",
    label: "Done",
  },
};

function ResumeBadge({ target }: { target: ResumeTarget }) {
  const style = BADGE_STYLE[target] ?? {
    bg: "#eee",
    fg: "#444",
    label: target,
  };
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        background: style.bg,
        color: style.fg,
        borderRadius: 12,
        fontSize: 11,
        fontWeight: 600,
        whiteSpace: "nowrap",
      }}
    >
      {style.label}
    </span>
  );
}

function RelTime({ iso }: { iso: string }) {
  const date = new Date(iso);
  const ms = Date.now() - date.getTime();
  const minutes = Math.round(ms / 60_000);
  let label: string;
  if (minutes < 1) label = "just now";
  else if (minutes < 60) label = `${minutes}m ago`;
  else if (minutes < 60 * 24) label = `${Math.round(minutes / 60)}h ago`;
  else label = `${Math.round(minutes / (60 * 24))}d ago`;
  return <span title={iso}>{label}</span>;
}
