import { Sparkline } from "./Sparkline";

type Props = {
  trend: number[];
  threshold: number;
};

export function PKTab({ trend, threshold }: Props) {
  return (
    <div className="pktab">
      <div className="pktab__title">P@K trend</div>
      <Sparkline values={trend} threshold={threshold} />
      <div className="pktab__threshold">
        Threshold: {threshold.toFixed(2)}
      </div>
      <ol className="pktab__list">
        {trend.length === 0 && (
          <li className="pktab__empty">—</li>
        )}
        {trend.map((v, i) => (
          <li key={i} className="pktab__row">
            <span className="pktab__turn">Turn {i + 1}</span>
            <span className="pktab__val">{v.toFixed(2)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
