import type { Scope } from "../types";

type Props = {
  scopes: Scope[];
  value: string | null;
  onChange: (name: string) => void;
  isLoading: boolean;
  error: string | null;
};

export function ScopePicker({
  scopes,
  value,
  onChange,
  isLoading,
  error,
}: Props) {
  const done = value !== null;

  if (isLoading) {
    return (
      <fieldset className="phase-picker">
        <legend className="query-form__label">
          <span className="bullet">○</span> Scope
        </legend>
        <div className="phase-picker__hint">Loading scopes…</div>
      </fieldset>
    );
  }

  if (error) {
    return (
      <fieldset className="phase-picker">
        <legend className="query-form__label">
          <span className="bullet">○</span> Scope
        </legend>
        <div className="query-form__error">
          Failed to load scopes: {error}
        </div>
      </fieldset>
    );
  }

  if (scopes.length === 0) {
    return (
      <fieldset className="phase-picker">
        <legend className="query-form__label">
          <span className="bullet">○</span> Scope
        </legend>
        <div className="query-form__error">
          No scopes configured. Edit scopes.yaml on the server.
        </div>
      </fieldset>
    );
  }

  return (
    <fieldset className="phase-picker">
      <legend className="query-form__label">
        <span className={"bullet" + (done ? " bullet--done" : "")}>
          {done ? "●" : "○"}
        </span>{" "}
        Scope
      </legend>
      <div className="phase-picker__row">
        {scopes.map((scope) => (
          <label
            key={scope.name}
            className={
              "phase-picker__opt" +
              (value === scope.name ? " phase-picker__opt--on" : "")
            }
          >
            <input
              type="radio"
              name="scope"
              value={scope.name}
              checked={value === scope.name}
              onChange={() => onChange(scope.name)}
            />
            <span className="phase-picker__label">{scope.name}</span>
            <span className="phase-picker__hint">{scope.description}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
