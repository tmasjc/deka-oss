import { useMemo, useState } from "react";
import { formatApiError } from "../api/client";
import { useEditConfig } from "../hooks/useSession";
import type { Params, PathName, UpdateConfigRequest } from "../types";

const ALL_PATHS: PathName[] = ["dense", "sparse"];

type Props = {
  sid: string;
  params: Params;
  onClose: () => void;
};

export function ConfigEditorModal({ sid, params, onClose }: Props) {
  const editConfig = useEditConfig(sid);
  const [rrfK, setRrfK] = useState(params.rrf_k);
  const [perPathLimit, setPerPathLimit] = useState(params.per_path_limit);
  const [topK, setTopK] = useState(params.top_k);
  const [activePaths, setActivePaths] = useState<Set<PathName>>(
    () => new Set(params.active_paths),
  );

  const validation = useMemo(() => {
    if (!Number.isInteger(rrfK) || rrfK < 1) return "rrf_k must be an integer ≥ 1.";
    if (!Number.isInteger(perPathLimit) || perPathLimit < 1)
      return "per_path_limit must be an integer ≥ 1.";
    if (!Number.isInteger(topK) || topK < 1) return "top_k must be an integer ≥ 1.";
    if (activePaths.size === 0)
      return "At least one retrieval path must remain checked.";
    return null;
  }, [rrfK, perPathLimit, topK, activePaths]);

  function togglePath(p: PathName) {
    setActivePaths((prev) => {
      const next = new Set(prev);
      if (next.has(p)) next.delete(p);
      else next.add(p);
      return next;
    });
  }

  function onSave() {
    if (validation) return;
    const body: UpdateConfigRequest = {};
    if (rrfK !== params.rrf_k) body.rrf_k = rrfK;
    if (perPathLimit !== params.per_path_limit) body.per_path_limit = perPathLimit;
    if (topK !== params.top_k) body.top_k = topK;
    const currentPaths = new Set(params.active_paths);
    const changed =
      activePaths.size !== currentPaths.size ||
      [...activePaths].some((p) => !currentPaths.has(p));
    if (changed) body.active_paths = ALL_PATHS.filter((p) => activePaths.has(p));
    if (Object.keys(body).length === 0) {
      onClose();
      return;
    }
    editConfig.mutate(body, { onSuccess: onClose });
  }

  return (
    <div
      className="modal__backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-modal="true">
        <h2 className="modal__title">Edit config</h2>

        <div className="modal__section">
          <div className="modal__section-label">Tunable parameters</div>
          <div className="cfg__grid">
            <NumberField label="rrf_k" value={rrfK} onChange={setRrfK} />
            <NumberField
              label="per_path_limit"
              value={perPathLimit}
              onChange={setPerPathLimit}
            />
            <NumberField label="top_k" value={topK} onChange={setTopK} />
          </div>
        </div>

        <div className="modal__section">
          <div className="modal__section-label">Active paths</div>
          <div className="cfg__paths">
            {ALL_PATHS.map((p) => (
              <label key={p} className="cfg__check">
                <input
                  type="checkbox"
                  checked={activePaths.has(p)}
                  onChange={() => togglePath(p)}
                />
                <span>{p}</span>
              </label>
            ))}
          </div>
        </div>

        {validation && <p className="modal__error">{validation}</p>}
        {editConfig.isError && (
          <p className="modal__error">
            {formatApiError(editConfig.error) || "Failed to save config"}
          </p>
        )}

        <div className="cfg__actions">
          <button type="button" className="btn" onClick={onClose}>
            <span className="btn__cap">Cancel</span>
            <span className="btn__key">[esc]</span>
          </button>
          <button
            type="button"
            className="btn btn--fit"
            onClick={onSave}
            disabled={!!validation || editConfig.isPending}
          >
            <span className="btn__cap">
              {editConfig.isPending ? "Saving…" : "Save"}
            </span>
          </button>
        </div>

        <p className="modal__hint">
          Changes apply to the next turn; current turn's ratings stay valid.
        </p>
      </div>
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="cfg__field">
      <span className="cfg__field-label">{label}</span>
      <input
        type="number"
        min={1}
        step={1}
        value={Number.isNaN(value) ? "" : value}
        onChange={(e) => {
          const n = parseInt(e.target.value, 10);
          onChange(Number.isNaN(n) ? NaN : n);
        }}
      />
    </label>
  );
}
