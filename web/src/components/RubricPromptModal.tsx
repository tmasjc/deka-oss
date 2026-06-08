import { formatApiError } from "../api/client";
import { useRubricPrompt } from "../hooks/useRefine";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";

type Props = {
  sid: string;
  onClose: () => void;
};

export function RubricPromptModal({ sid, onClose }: Props) {
  const { data, isLoading, error } = useRubricPrompt(sid, true);

  useKeyboardShortcuts(
    {
      escape: () => onClose(),
    },
    true,
  );

  return (
    <div
      className="modal__backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        style={{ maxWidth: "min(960px, 90vw)", width: "100%" }}
      >
        <h2 className="modal__title">
          Rubric prompt{data && ` — v${data.metadata.version}`}
        </h2>
        {isLoading && <p>Loading…</p>}
        {error && (
          <p className="modal__error">
            {formatApiError(error) || "Failed to load rubric prompt."}
          </p>
        )}
        {data && (
          <pre
            style={{
              maxHeight: "60vh",
              overflow: "auto",
              padding: "12px",
              background: "#ffffff",
              color: "#000000",
              border: "1px solid #d0d0d0",
              borderRadius: 4,
              fontFamily: "ui-monospace, monospace",
              fontSize: 12,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {data.rubric_text}
          </pre>
        )}
        <div
          className="cfg__actions"
          style={{ justifyContent: "flex-end", marginTop: 12 }}
        >
          <button type="button" className="btn" onClick={onClose}>
            <span className="btn__cap">Close</span>
            <span className="btn__key">[esc]</span>
          </button>
        </div>
      </div>
    </div>
  );
}
