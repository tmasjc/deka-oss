import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";

type Props = {
  onClose: () => void;
  onConfirmed: () => void;
};

/**
 * Pre-Ship confirmation gate. Pure ceremonial — no fetch, no mutation.
 * Confirmation calls ``onConfirmed`` which fires POST /apply/finalize
 * from the parent panel. Shell mirrors :class:`RefineConfirmModal` so
 * the modal shortcut grammar (Enter/Y to confirm, Esc/N to dismiss)
 * stays consistent across phases.
 */
export function ShipConfirmModal({ onClose, onConfirmed }: Props) {
  useKeyboardShortcuts(
    {
      enter: onConfirmed,
      y: onConfirmed,
      escape: onClose,
      n: onClose,
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
      <div className="modal" role="dialog" aria-modal="true">
        <h2 className="modal__title">Confirm to proceed?</h2>
        <div className="modal__section">
          <p className="modal__hint">you cannot come back to edit again.</p>
        </div>
        <div className="cfg__actions">
          <button type="button" className="btn" onClick={onClose}>
            <span className="btn__cap">No</span>
            <span className="btn__key">[esc]</span>
          </button>
          <button type="button" className="btn btn--fit" onClick={onConfirmed}>
            <span className="btn__cap">Yes</span>
            <span className="btn__key">[↵]</span>
          </button>
        </div>
      </div>
    </div>
  );
}
