type Props = {
  values: number[];
  threshold: number;
  width?: number;
  height?: number;
};

const DOMAIN_MIN = 0;
const DOMAIN_MAX = 1;

export function Sparkline({ values, threshold, width = 200, height = 40 }: Props) {
  if (values.length === 0) {
    return (
      <div className="spark spark--empty">
        No turns completed yet.
      </div>
    );
  }

  const xStep = values.length === 1 ? 0 : width / (values.length - 1);
  const scaleY = (v: number) =>
    height - ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * height;

  const points =
    values.length === 1
      ? `${width / 2},${scaleY(values[0])}`
      : values.map((v, i) => `${i * xStep},${scaleY(v)}`).join(" ");

  const thresholdY = scaleY(threshold);

  return (
    <svg
      className="spark"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
    >
      <line
        x1={0}
        x2={width}
        y1={thresholdY}
        y2={thresholdY}
        stroke="var(--pos-glow, #6bb26b)"
        strokeDasharray="2 3"
        strokeWidth={1}
      />
      <polyline
        fill="none"
        stroke="var(--accent)"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        points={points}
      />
      {values.map((v, i) => (
        <circle
          key={i}
          cx={values.length === 1 ? width / 2 : i * xStep}
          cy={scaleY(v)}
          r={2}
          fill="var(--accent)"
        />
      ))}
    </svg>
  );
}
