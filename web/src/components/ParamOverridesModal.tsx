import { useEffect, useMemo, useState } from "react";
import { useConfigDefaults } from "../hooks/useConfigDefaults";
import { useParamOverrides } from "../state/paramOverrides";
import type {
  ApplyOverrides,
  ConfigDefaults,
  HarvestOverrides,
  PathName,
  RefineOverrides,
  SearchOverrides,
} from "../types";

const ALL_PATHS: PathName[] = ["dense", "sparse"];
const RADIUS_SCHEMES: Array<"per_fit" | "decoupled"> = ["per_fit", "decoupled"];

type Props = {
  onClose: () => void;
};

/**
 * Per-session [Edit parameters] modal opened from the query page.
 *
 * Reads/writes the Zustand `useParamOverrides` slice — that store
 * survives this modal's mount and is what QueryEntry submits on
 * Start session. Pre-fills placeholders from `/api/config/defaults`
 * so an empty input means "use server default".
 */
export function ParamOverridesModal({ onClose }: Props) {
  const defaultsQuery = useConfigDefaults();
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="modal__backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal modal--wide" role="dialog" aria-modal="true">
        <h2 className="modal__title">Edit parameters</h2>
        <p className="modal__hint">
          Per-session overrides for the four phases. Empty input = use server
          default. Fixed parameters (URLs, models, credentials) stay in the
          server config.
        </p>

        {defaultsQuery.isLoading && (
          <p className="modal__hint">Loading defaults…</p>
        )}
        {defaultsQuery.isError && (
          <p className="modal__error">
            Couldn't load server defaults. You can still set overrides; the
            server will validate them at submit time.
          </p>
        )}

        {defaultsQuery.data && (
          <ParamSections defaults={defaultsQuery.data} />
        )}

        <div className="cfg__actions">
          <ResetButton />
          <button type="button" className="btn" onClick={onClose}>
            <span className="btn__cap">Done</span>
            <span className="btn__key">[esc]</span>
          </button>
        </div>
      </div>
    </div>
  );
}

function ResetButton() {
  const reset = useParamOverrides((s) => s.resetAll);
  const diffCount = useParamOverrides((s) => s.diffCount());
  return (
    <button
      type="button"
      className="btn"
      onClick={reset}
      disabled={diffCount === 0}
    >
      <span className="btn__cap">Reset to defaults</span>
    </button>
  );
}

function ParamSections({ defaults }: { defaults: ConfigDefaults }) {
  return (
    <>
      <SearchSection defaults={defaults.search} />
      <HarvestSection defaults={defaults.harvest} />
      <RefineSection defaults={defaults.refine} />
      <ApplySection defaults={defaults.apply} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Phase sections
// ---------------------------------------------------------------------------

function SearchSection({ defaults }: { defaults: Required<SearchOverrides> }) {
  const block = useParamOverrides((s) => s.search) ?? {};
  const setPhase = useParamOverrides((s) => s.setPhase);
  const update = (patch: SearchOverrides) =>
    setPhase("search", pruneBlock({ ...block, ...patch }, defaults));

  return (
    <SectionCard title="Phase 1 · Search" hint="Affects retrieval fanout and final P@K.">
      <NumberField
        label="top_k"
        placeholder={defaults.top_k}
        value={block.top_k}
        min={1}
        onChange={(v) => update({ top_k: v })}
      />
      <NumberField
        label="per_path_limit"
        placeholder={defaults.per_path_limit}
        value={block.per_path_limit}
        min={1}
        onChange={(v) => update({ per_path_limit: v })}
      />
      <NumberField
        label="min_survivors"
        placeholder={defaults.min_survivors}
        value={block.min_survivors}
        min={1}
        onChange={(v) => update({ min_survivors: v })}
      />
      <PathsField
        defaults={defaults.active_paths}
        value={block.active_paths}
        onChange={(v) => update({ active_paths: v })}
      />
    </SectionCard>
  );
}

function HarvestSection({ defaults }: { defaults: Required<HarvestOverrides> }) {
  const block = useParamOverrides((s) => s.harvest) ?? {};
  const setPhase = useParamOverrides((s) => s.setPhase);
  const update = (patch: HarvestOverrides) =>
    setPhase("harvest", pruneBlock({ ...block, ...patch }, defaults));

  return (
    <SectionCard
      title="Phase 2 · Harvest"
      hint="Convergence gate + Phase 2 threshold scheme."
    >
      <NumberField
        label="min_fit"
        placeholder={defaults.min_fit}
        value={block.min_fit}
        min={1}
        onChange={(v) => update({ min_fit: v })}
      />
      <NumberField
        label="min_not_fit"
        placeholder={defaults.min_not_fit}
        value={block.min_not_fit}
        min={1}
        onChange={(v) => update({ min_not_fit: v })}
      />
      <NumberField
        label="precision_at_k"
        placeholder={defaults.precision_at_k}
        value={block.precision_at_k}
        min={0}
        max={1}
        step={0.01}
        onChange={(v) => update({ precision_at_k: v })}
      />
      <SelectField
        label="radius_scheme"
        placeholder={defaults.radius_scheme}
        value={block.radius_scheme}
        options={RADIUS_SCHEMES}
        onChange={(v) =>
          update({ radius_scheme: v as "per_fit" | "decoupled" | undefined })
        }
      />
      <NumberField
        label="s2c_outlier_multiple"
        placeholder={defaults.s2c_outlier_multiple}
        value={block.s2c_outlier_multiple}
        min={1.01}
        step={0.1}
        onChange={(v) => update({ s2c_outlier_multiple: v })}
      />
      <NumberField
        label="anchor_frequency_gate"
        placeholder={defaults.anchor_frequency_gate}
        value={block.anchor_frequency_gate}
        min={1}
        onChange={(v) => update({ anchor_frequency_gate: v })}
      />
    </SectionCard>
  );
}

function RefineSection({ defaults }: { defaults: Required<RefineOverrides> }) {
  const block = useParamOverrides((s) => s.refine) ?? {};
  const setPhase = useParamOverrides((s) => s.setPhase);
  const update = (patch: RefineOverrides) =>
    setPhase("refine", pruneBlock({ ...block, ...patch }, defaults));

  return (
    <SectionCard
      title="Phase 3 · Refine"
      hint="Stratified judging sample for rubric-driven boundary refinement."
    >
      <NumberField
        label="sample_size"
        placeholder={defaults.sample_size}
        value={block.sample_size}
        min={1}
        onChange={(v) => update({ sample_size: v })}
      />
      <NumberField
        label="n_bins"
        placeholder={defaults.n_bins}
        value={block.n_bins}
        min={1}
        onChange={(v) => update({ n_bins: v })}
      />
      <NumberField
        label="seed"
        placeholder={defaults.seed}
        value={block.seed}
        onChange={(v) => update({ seed: v })}
      />
      <NumberField
        label="max_fit_examples"
        placeholder={defaults.max_fit_examples}
        value={block.max_fit_examples}
        min={1}
        onChange={(v) => update({ max_fit_examples: v })}
      />
      <NumberField
        label="max_not_fit_examples"
        placeholder={defaults.max_not_fit_examples}
        value={block.max_not_fit_examples}
        min={1}
        onChange={(v) => update({ max_not_fit_examples: v })}
      />
      <BoolField
        label="auto_drop_known_intruders"
        placeholder={defaults.auto_drop_known_intruders}
        value={block.auto_drop_known_intruders}
        onChange={(v) => update({ auto_drop_known_intruders: v })}
      />
    </SectionCard>
  );
}

function ApplySection({ defaults }: { defaults: Required<ApplyOverrides> }) {
  const block = useParamOverrides((s) => s.apply) ?? {};
  const setPhase = useParamOverrides((s) => s.setPhase);
  const update = (patch: ApplyOverrides) =>
    setPhase("apply", pruneBlock({ ...block, ...patch }, defaults));

  return (
    <SectionCard
      title="Phase 4 · Apply"
      hint="Logistic-regression classifier threshold + acceptance bar."
    >
      <BoolField
        label="enabled"
        placeholder={defaults.enabled}
        value={block.enabled}
        onChange={(v) => update({ enabled: v })}
      />
      <NumberField
        label="confidence_threshold"
        placeholder={defaults.confidence_threshold}
        value={block.confidence_threshold}
        min={0}
        max={1}
        step={0.01}
        onChange={(v) => update({ confidence_threshold: v })}
      />
      <NumberField
        label="min_precision"
        placeholder={defaults.min_precision}
        value={block.min_precision}
        min={0}
        max={1}
        step={0.01}
        onChange={(v) => update({ min_precision: v })}
      />
      <NumberField
        label="kfold_splits"
        placeholder={defaults.kfold_splits}
        value={block.kfold_splits}
        min={2}
        onChange={(v) => update({ kfold_splits: v })}
      />
    </SectionCard>
  );
}

// ---------------------------------------------------------------------------
// Shared building blocks
// ---------------------------------------------------------------------------

function SectionCard({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="modal__section">
      <div className="modal__section-label">{title}</div>
      <p className="modal__hint" style={{ marginTop: 0 }}>
        {hint}
      </p>
      <div className="cfg__grid">{children}</div>
    </div>
  );
}

function NumberField({
  label,
  placeholder,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  placeholder: number;
  value: number | undefined;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number | undefined) => void;
}) {
  const [text, setText] = useState<string>(value === undefined ? "" : String(value));

  useEffect(() => {
    setText(value === undefined ? "" : String(value));
  }, [value]);

  const error = useMemo(() => {
    if (text.trim() === "") return null;
    const n = step && step < 1 ? parseFloat(text) : parseInt(text, 10);
    if (!Number.isFinite(n)) return "Not a number";
    if (min !== undefined && n < min) return `Must be ≥ ${min}`;
    if (max !== undefined && n > max) return `Must be ≤ ${max}`;
    if ((step === undefined || step >= 1) && !Number.isInteger(n))
      return "Must be an integer";
    return null;
  }, [text, min, max, step]);

  return (
    <label className="cfg__field">
      <span className="cfg__field-label">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step ?? 1}
        placeholder={String(placeholder)}
        value={text}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          if (next.trim() === "") {
            onChange(undefined);
            return;
          }
          const n = step && step < 1 ? parseFloat(next) : parseInt(next, 10);
          if (Number.isFinite(n)) onChange(n);
        }}
      />
      {error && <span className="modal__error">{error}</span>}
    </label>
  );
}

function BoolField({
  label,
  placeholder,
  value,
  onChange,
}: {
  label: string;
  placeholder: boolean;
  value: boolean | undefined;
  onChange: (v: boolean | undefined) => void;
}) {
  const current = value === undefined ? null : value;
  const labelText = (b: boolean) => (b ? "true" : "false");
  return (
    <label className="cfg__field">
      <span className="cfg__field-label">{label}</span>
      <select
        value={current === null ? "" : labelText(current)}
        onChange={(e) => {
          if (e.target.value === "") onChange(undefined);
          else onChange(e.target.value === "true");
        }}
      >
        <option value="">default ({labelText(placeholder)})</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    </label>
  );
}

function SelectField({
  label,
  placeholder,
  value,
  options,
  onChange,
}: {
  label: string;
  placeholder: string;
  value: string | undefined;
  options: readonly string[];
  onChange: (v: string | undefined) => void;
}) {
  return (
    <label className="cfg__field">
      <span className="cfg__field-label">{label}</span>
      <select
        value={value ?? ""}
        onChange={(e) =>
          onChange(e.target.value === "" ? undefined : e.target.value)
        }
      >
        <option value="">default ({placeholder})</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function PathsField({
  defaults,
  value,
  onChange,
}: {
  defaults: PathName[];
  value: PathName[] | undefined;
  onChange: (v: PathName[] | undefined) => void;
}) {
  const current = value ?? defaults;
  const isDefault =
    !value || (value.length === defaults.length && defaults.every((p) => value.includes(p)));

  function toggle(p: PathName) {
    const next = current.includes(p)
      ? current.filter((x) => x !== p)
      : [...current, p];
    // Sort to keep canonical ordering so equality check works.
    const ordered = ALL_PATHS.filter((x) => next.includes(x));
    if (
      ordered.length === defaults.length &&
      defaults.every((x) => ordered.includes(x))
    ) {
      onChange(undefined); // back to default
    } else {
      onChange(ordered);
    }
  }

  return (
    <label className="cfg__field" style={{ gridColumn: "1 / -1" }}>
      <span className="cfg__field-label">
        active_paths {isDefault && <em>(default)</em>}
      </span>
      <div className="cfg__paths">
        {ALL_PATHS.map((p) => (
          <label key={p} className="cfg__check">
            <input
              type="checkbox"
              checked={current.includes(p)}
              onChange={() => toggle(p)}
            />
            <span>{p}</span>
          </label>
        ))}
      </div>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Remove entries that equal the default so the diff stays minimal.
 * Empty / undefined values are also stripped. Returns null when the
 * resulting block has zero entries — the caller stores null for "no
 * overrides for this phase".
 */
function pruneBlock<T extends object>(block: T, defaults: T): T | null {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(block) as Array<keyof T>) {
    const v = block[key];
    if (v === undefined) continue;
    const d = defaults[key];
    if (Array.isArray(v) && Array.isArray(d)) {
      if (v.length === d.length && d.every((x, i) => v[i] === x)) continue;
    } else if (v === d) {
      continue;
    }
    out[key as string] = v;
  }
  return Object.keys(out).length === 0 ? null : (out as T);
}
