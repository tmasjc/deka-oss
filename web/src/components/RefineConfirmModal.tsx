import { useEffect, useState } from "react";
import { formatApiError } from "../api/client";
import {
  useRefineDerive,
  useRefinePreflight,
} from "../hooks/useRefine";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import type { RefinePreflight } from "../types";

type Props = {
  sid: string;
  onClose: () => void;
  onConfirmed: () => void;
};

export function RefineConfirmModal({ sid, onClose, onConfirmed }: Props) {
  const preflight = useRefinePreflight(sid);
  const derive = useRefineDerive(sid);
  const [data, setData] = useState<RefinePreflight | null>(null);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onRun() {
    setError(null);
    derive.mutate(undefined, {
      onSuccess: () => onConfirmed(),
      onError: (err) =>
        setError(formatApiError(err) ?? "Could not start derive."),
    });
  }

  useKeyboardShortcuts(
    {
      enter: () => {
        if (data && !derive.isPending) onRun();
      },
      y: () => {
        if (data && !derive.isPending) onRun();
      },
      escape: () => {
        if (!derive.isPending) onClose();
      },
      n: () => {
        if (!derive.isPending) onClose();
      },
    },
    true,
  );

  return (
    <div
      className="modal__backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget && !derive.isPending) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-modal="true">
        <h2 className="modal__title">
          Phase 2 complete — derive a rubric and judge a sample?
        </h2>
        {data ? (
          <div className="modal__section">
            <div className="cfg__grid">
              <KvRow
                label="Phase 2 cohort"
                value={`${data.phase2_count} candidates`}
              />
              <KvRow
                label="Sample"
                value={`${data.sample_size} chunks across ${data.n_bins} deciles`}
              />
              <KvRow label="Derive model" value={data.derive_model} />
              <KvRow label="Judge model" value={data.judge_model} />
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
            disabled={derive.isPending}
          >
            <span className="btn__cap">Skip</span>
            <span className="btn__key">[esc]</span>
          </button>
          <button
            type="button"
            className="btn btn--fit"
            onClick={onRun}
            disabled={!data || derive.isPending}
          >
            <span className="btn__cap">
              {derive.isPending ? "Starting…" : "Derive rubric"}
            </span>
            <span className="btn__key">[↵]</span>
          </button>
        </div>
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
