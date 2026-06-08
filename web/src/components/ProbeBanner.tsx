import type { PathName, ProbeSummary } from "../types";

type Props = { probe: ProbeSummary; onDismiss: () => void };

const PATHS: PathName[] = ["dense", "sparse"];

export function ProbeBanner({ probe, onDismiss }: Props) {
  const lines: string[] = PATHS.map((path) => {
    const stats = probe.stats_by_path[path];
    if (!stats || stats.skipped) return `${path}: skipped`;
    if (stats.hit_count === 0) return `${path}: 0 hits`;
    const min = stats.score_min ?? 0;
    const max = stats.score_max ?? 0;
    const mean = stats.score_mean ?? 0;
    return `${path}: ${stats.hit_count} hits  scores ${min.toFixed(3)}–${max.toFixed(3)}  mean ${mean.toFixed(3)}`;
  });

  return (
    <div className="banner">
      <div>
        <strong>Probe summary</strong> — {lines.join("  ·  ")}
      </div>
      {probe.rationale.length > 0 && (
        <div style={{ marginTop: 4 }}>{probe.rationale.join("; ")}</div>
      )}
      {probe.flags.length > 0 && (
        <div style={{ marginTop: 4, color: "var(--warn, #b45309)" }}>
          ⚠ {probe.flags.join("; ")}
        </div>
      )}
      <button
        type="button"
        className="tab"
        onClick={onDismiss}
        style={{ marginTop: 6 }}
      >
        dismiss
      </button>
    </div>
  );
}
