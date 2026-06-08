import type { Rating } from "../types";
import { TranscriptLine } from "./TranscriptLine";

type Props = {
  content: string;
  spanLines: number[];
  rating: Rating | null;
  originalContent?: string;
  isExpanding?: boolean;
};

export function ChunkCard({
  content,
  spanLines,
  rating,
  originalContent,
  isExpanding,
}: Props) {
  const expanded = originalContent !== undefined;
  const displayed = expanded ? originalContent! : content;
  const lines = displayed.split("\n");
  // Span highlights only make sense on the Milvus chunk text.
  const spanSet = expanded ? new Set<number>() : new Set(spanLines);
  const cardClass =
    "chunk__card" +
    (rating === "FIT"
      ? " chunk__card--fit"
      : rating === "NOT_FIT"
        ? " chunk__card--nofit"
        : rating === "DISCARD"
          ? " chunk__card--discard"
          : "") +
    (expanded ? " chunk__card--expanded" : "");
  const statusText =
    rating === "FIT"
      ? "— fit —"
      : rating === "NOT_FIT"
        ? "— not fit —"
        : rating === "DISCARD"
          ? "— discarded —"
          : "— unrated —";
  const statusClass =
    "chunk__status" +
    (rating === "FIT"
      ? " chunk__status--fit"
      : rating === "NOT_FIT"
        ? " chunk__status--nofit"
        : rating === "DISCARD"
          ? " chunk__status--discard"
          : "");
  return (
    <div className={cardClass}>
      <div className="chunk__body">
        {isExpanding && !expanded ? (
          <div className="chunk__expanding">Loading original content…</div>
        ) : (
          lines.map((line, i) => (
            <TranscriptLine key={i} line={line} highlighted={spanSet.has(i)} />
          ))
        )}
        <div className={statusClass}>{statusText}</div>
      </div>
    </div>
  );
}
