import type { BreakdownRow, Convergence } from "../types";

type Props = {
  convergence: Convergence;
  breakdown: Record<string, BreakdownRow>;
};

const BREAKDOWN_LABELS: Record<string, string> = {
  dense_only: "dense",
  sparse_only: "sparse",
  multi_path: "multi",
};

export function ConvergencePanel({ convergence, breakdown }: Props) {
  const pkMet = convergence.pk_current >= convergence.pk_threshold;
  const fitMet = convergence.fit_current >= convergence.fit_threshold;
  const notFitMet =
    convergence.not_fit_current >= convergence.not_fit_threshold;
  return (
    <div className="conv">
      <div className="conv__title">Convergence</div>
      <div className="conv__list">
        <div className="conv__row">
          <span className={"conv__box" + (pkMet ? " conv__box--on" : "")}>
            {pkMet ? "●" : "○"}
          </span>
          <span className="conv__label">
            P@K ≥ {convergence.pk_threshold.toFixed(2)}
          </span>
          <span className="conv__sep">current:</span>
          <span className="conv__val">{convergence.pk_current.toFixed(2)}</span>
        </div>
        <div className="conv__row">
          <span className={"conv__box" + (fitMet ? " conv__box--on" : "")}>
            {fitMet ? "●" : "○"}
          </span>
          <span className="conv__label">FIT ≥ {convergence.fit_threshold}</span>
          <span className="conv__sep">current:</span>
          <span className="conv__val">{convergence.fit_current}</span>
        </div>
        <div className="conv__row">
          <span className={"conv__box" + (notFitMet ? " conv__box--on" : "")}>
            {notFitMet ? "●" : "○"}
          </span>
          <span className="conv__label">
            NOT_FIT ≥ {convergence.not_fit_threshold}
          </span>
          <span className="conv__sep">current:</span>
          <span className="conv__val">{convergence.not_fit_current}</span>
        </div>
      </div>
      <div className="conv__sub">Breakdown (cumulative)</div>
      <div className="conv__breakdown">
        {Object.entries(BREAKDOWN_LABELS).map(([key, label]) => {
          const row = breakdown[key];
          if (!row) return null;
          return (
            <div key={key} style={{ display: "contents" }}>
              <span className="conv__bk-key">{label}</span>
              <span className="conv__bk-val">
                {row.fit}/{row.total}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
