import type { TurnBreakdown } from "../../types";

type Props = {
  turns: TurnBreakdown[];
};

const ROWS: Array<{ key: string; label: string }> = [
  { key: "dense_only", label: "dense" },
  { key: "sparse_only", label: "sparse" },
  { key: "multi_path", label: "multi" },
];

function cell(turn: TurnBreakdown, key: string): string {
  const row = turn.breakdown[key];
  if (!row || row.total === 0) return "—";
  return `${row.fit}/${row.total}`;
}

export function PathsTab({ turns }: Props) {
  if (turns.length === 0) {
    return (
      <div className="pathstab">
        <div className="pathstab__title">Per-turn breakdown</div>
        <div className="pathstab__empty">No turns completed yet.</div>
      </div>
    );
  }
  return (
    <div className="pathstab">
      <div className="pathstab__title">Per-turn breakdown</div>
      <div className="pathstab__scroll">
        <table className="pathstab__table">
          <thead>
            <tr>
              <th />
              {turns.map((t) => (
                <th key={t.turn}>T{t.turn}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map(({ key, label }) => (
              <tr key={key}>
                <th scope="row">{label}</th>
                {turns.map((t) => (
                  <td key={t.turn}>{cell(t, key)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
