import type { Params } from "../../types";

type Props = {
  params: Params;
  drop: Record<string, unknown> | null;
};

export function ConfigTab({ params, drop }: Props) {
  return (
    <div className="cfgtab">
      <div className="cfgtab__title">Current config</div>
      <dl className="cfgtab__dl">
        <dt>rrf_k</dt>
        <dd>{params.rrf_k}</dd>
        <dt>per_path_limit</dt>
        <dd>{params.per_path_limit}</dd>
        <dt>top_k</dt>
        <dd>{params.top_k}</dd>
        <dt>active_paths</dt>
        <dd>{params.active_paths.join(", ")}</dd>
      </dl>

      <div className="cfgtab__title cfgtab__title--sub">Drop-impact preview</div>
      <DropPreview drop={drop} />
    </div>
  );
}

function DropPreview({ drop }: { drop: Record<string, unknown> | null }) {
  const entries = drop ? Object.entries(drop) : [];
  if (entries.length === 0) {
    return (
      <div className="cfgtab__empty">No drop previews for this turn.</div>
    );
  }
  return (
    <ul className="cfgtab__drop">
      {entries.map(([path, preview]) => (
        <li key={path}>
          <span className="cfgtab__drop-path">Drop {path}</span>
          <span className="cfgtab__drop-body">{formatPreview(preview)}</span>
        </li>
      ))}
    </ul>
  );
}

function formatPreview(preview: unknown): string {
  if (preview == null) return "—";
  if (Array.isArray(preview)) {
    return preview.map((x) => String(x)).join(", ") || "—";
  }
  if (typeof preview === "object") {
    // Common shape from backend: {added_pks: [...], lost_rated_pks: [...]}.
    const obj = preview as Record<string, unknown>;
    const parts: string[] = [];
    for (const [k, v] of Object.entries(obj)) {
      if (Array.isArray(v)) parts.push(`${k}: ${v.length}`);
      else parts.push(`${k}: ${String(v)}`);
    }
    return parts.join("  ·  ");
  }
  return String(preview);
}
