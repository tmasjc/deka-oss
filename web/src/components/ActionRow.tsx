type Props = {
  onFit: () => void;
  onNotFit: () => void;
  onDiscard: () => void;
  disabled?: boolean;
};

export function ActionRow({ onFit, onNotFit, onDiscard, disabled }: Props) {
  return (
    <div className="actions">
      <div className="actions__verdicts">
        <button
          className="btn btn--fit"
          onClick={onFit}
          disabled={disabled}
          type="button"
        >
          <span className="btn__cap">Fit</span>
          <span className="btn__key">[f]</span>
        </button>
        <button
          className="btn btn--nofit"
          onClick={onNotFit}
          disabled={disabled}
          type="button"
        >
          <span className="btn__cap">Not Fit</span>
          <span className="btn__key">[n]</span>
        </button>
      </div>
      <button
        className="btn--discard"
        onClick={onDiscard}
        disabled={disabled}
        type="button"
        title="Invalidate this chunk — neither FIT nor NOT_FIT. Use for STT errors, garbled grammar, or content that won't help the rubric."
      >
        <span className="btn--discard__icon" aria-hidden="true">
          {/* trash glyph */}
          ⌫
        </span>
        <span className="btn--discard__cap">discard chunk</span>
        <span className="btn--discard__key">[d]</span>
      </button>
    </div>
  );
}
