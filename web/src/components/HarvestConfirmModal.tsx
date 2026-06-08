import { useEffect, useState } from "react";
import { formatApiError } from "../api/client";
import {
  useHarvestPreflight,
  useHarvestRun,
} from "../hooks/useHarvest";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import type { HarvestPreflight } from "../types";

type Props = {
  sid: string;
  onClose: () => void;
  onConfirmed: () => void;
};

export function HarvestConfirmModal({ sid, onClose, onConfirmed }: Props) {
  const preflight = useHarvestPreflight(sid);
  const run = useHarvestRun(sid);
  const [data, setData] = useState<HarvestPreflight | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    preflight.mutate(undefined, {
      onSuccess: (d) => {
        if (alive) setData(d);
      },
      onError: (err) => {
        if (alive) setError(formatApiError(err) ?? "Could not load preflight.");
      },
    });
    return () => {
      alive = false;
    };
    // preflight is referentially stable from React Query.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onRun() {
    setError(null);
    run.mutate(undefined, {
      onSuccess: () => {
        onConfirmed();
      },
      onError: (err) =>
        setError(formatApiError(err) ?? "Could not start harvest."),
    });
  }

  useKeyboardShortcuts(
    {
      enter: () => {
        if (data && !run.isPending) onRun();
      },
      y: () => {
        if (data && !run.isPending) onRun();
      },
      escape: () => {
        if (!run.isPending) onClose();
      },
      n: () => {
        if (!run.isPending) onClose();
      },
    },
    true,
  );

  return (
    <div
      className="modal__backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget && !run.isPending) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-modal="true">
        <h2 className="modal__title">
          Phase 1 converged — run FIT-anchored retrieval?
        </h2>
        {data ? (
          <div className="modal__section">
            <div className="cfg__grid">
              <KvRow label="FITs accumulated" value={String(data.n_fit)} />
              <KvRow label="batch_size" value={String(data.batch_size)} />
              <KvRow label="max_k (safety)" value={String(data.max_k)} />
              <KvRow label="Radius scheme" value={data.radius_scheme} />
            </div>
          </div>
        ) : preflight.isPending ? (
          <p className="modal__hint">Loading preflight…</p>
        ) : null}

        {error && <p className="modal__error">{error}</p>}

        <div className="cfg__actions">
          <button
            type="button"
            className="btn"
            onClick={onClose}
            disabled={run.isPending}
          >
            <span className="btn__cap">Skip</span>
            <span className="btn__key">[esc]</span>
          </button>
          <button
            type="button"
            className="btn btn--fit"
            onClick={onRun}
            disabled={!data || run.isPending}
          >
            <span className="btn__cap">
              {run.isPending ? "Starting…" : "Run anchor"}
            </span>
            <span className="btn__key">[↵]</span>
          </button>
        </div>

        <p className="modal__hint">
          The pass calibrates a per-FIT radius, runs LOO recovery, then
          retrieves the surrounding cohort. Skipping ends the session at
          the converged Phase 1 logs.
        </p>
      </div>
    </div>
  );
}

function KvRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="cfg__field">
      <span className="cfg__field-label">{label}</span>
      <span style={{ fontWeight: 600 }}>{value}</span>
    </div>
  );
}
