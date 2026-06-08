type Props = {
  title: string;
  defaults: Record<string, unknown> | undefined;
  overrides: Record<string, unknown> | undefined;
};

/**
 * Sidebar panel that lists the effective config for one phase
 * (defaults ← overrides). Keys are taken from `defaults`, which the
 * server already filters to the override allow-list.
 *
 * A key that the operator changed via [Edit parameters] is flagged
 * with a small `(override)` tag so the sidebar makes deltas visible
 * at a glance.
 */
export function PhaseConfigPanel({ title, defaults, overrides }: Props) {
  if (!defaults) return null;
  const keys = Object.keys(defaults);
  if (keys.length === 0) return null;

  return (
    <div className="phase-panel">
      <div className="phase-panel__title">{title}</div>
      <div className="phase-panel__rows">
        {keys.map((k) => {
          const ov = overrides?.[k];
          const overridden = ov !== undefined;
          const value = overridden ? ov : defaults[k];
          return (
            <div key={k} className="phase-panel__row">
              <span className="phase-panel__key">{k}</span>
              <span
                className={
                  "phase-panel__val" +
                  (overridden ? " phase-panel__val--override" : "")
                }
              >
                {formatValue(value)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "number") {
    return Number.isInteger(v) ? String(v) : v.toFixed(2);
  }
  return String(v);
}
