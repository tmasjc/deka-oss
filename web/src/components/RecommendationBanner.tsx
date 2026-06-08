import type { PathDropRecommendation, RecommendationDecision } from "../types";

type Props = {
  recommendation: PathDropRecommendation;
  onDecision: (d: RecommendationDecision) => void;
  disabled?: boolean;
};

export function RecommendationBanner({
  recommendation,
  onDecision,
  disabled = false,
}: Props) {
  const { path, reason, confidence } = recommendation;
  return (
    <div className="banner">
      <strong>Agent recommends dropping path: {path}</strong>{" "}
      <span style={{ color: "var(--text-muted, #888)" }}>
        (confidence: {confidence})
      </span>
      <div style={{ marginTop: 4 }}>{reason}</div>
      <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
        <button
          type="button"
          className="tab"
          disabled={disabled}
          onClick={() => onDecision("apply")}
        >
          Apply
        </button>
        <button
          type="button"
          className="tab"
          disabled={disabled}
          onClick={() => onDecision("ignore")}
        >
          Ignore
        </button>
      </div>
    </div>
  );
}
