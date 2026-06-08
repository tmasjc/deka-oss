type Props = {
  turn: number | null;
  audit?: boolean;
  /** Session phase from the snapshot. Drives the subtitle. */
  phase?: string | null;
};

export function HeaderBar({ turn, audit = false, phase }: Props) {
  const subtitle = renderSubtitle(turn, phase);
  return (
    <header className="hdr">
      <div className="hdr__title">
        <span className="hdr__brand">Sonar</span>
        <span className="hdr__sep">·</span>
        <span className="hdr__subtitle">{subtitle}</span>
        {audit && (
          <span
            className="hdr__badge"
            style={{
              marginLeft: 12,
              padding: "1px 8px",
              borderRadius: 6,
              background: "var(--warn, #c08400)",
              color: "#fff",
              fontWeight: 600,
              fontSize: "0.85em",
            }}
          >
            AUDIT
          </span>
        )}
      </div>
    </header>
  );
}

function renderSubtitle(turn: number | null, phase: string | null | undefined) {
  if (turn == null) return "Semantic Query — enter a query";
  switch (phase) {
    case "ANCHOR_RUNNING":
      return "Semantic Query — Phase 2 — harvesting";
    case "ANCHOR_DONE":
      return "Semantic Query — Phase 2 — review summary";
    case "ANCHOR_FAILED":
      return "Semantic Query — Phase 2 — failed";
    case "REFINE_DERIVING":
      return "Semantic Query — Phase 3 — deriving rubric";
    case "REFINE_EDITING":
      return "Semantic Query — Phase 3 — edit rubric (ctrl+s save · ctrl+↵ judge)";
    case "REFINE_JUDGING":
      return "Semantic Query — Phase 3 — judging sample";
    case "REFINE_REVIEW":
      return "Semantic Query — Phase 3 — review verdicts (j/k · f/n · ↵)";
    case "REFINE_FAILED":
      return "Semantic Query — Phase 3 — failed";
    case "DONE":
      return "Semantic Query — Done";
    case "TUNING":
    default:
      return `Semantic Query — Turn ${turn} — rate chunks (f/n)`;
  }
}
