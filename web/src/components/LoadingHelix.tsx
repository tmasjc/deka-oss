const BAR_COUNT = 10;
const STAGGER_S = 0.15;

/**
 * "DNA helix" loading indicator — a row of vertical bars rising and
 * falling on a staggered loop. Replaces the prior spinning circle in
 * the progress overlays. Pure CSS keyframes (no framer-motion dep);
 * ``prefers-reduced-motion`` is honoured in theme.css.
 */
export function LoadingHelix() {
  return (
    <div className="loading-helix" aria-hidden="true">
      {Array.from({ length: BAR_COUNT }).map((_, i) => (
        <span
          key={i}
          className="loading-helix__bar"
          style={{ animationDelay: `${i * STAGGER_S}s` }}
        />
      ))}
    </div>
  );
}
