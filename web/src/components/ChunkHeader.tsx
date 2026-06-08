import type { CandidateRow, EvidenceRow, RatableItem } from "../types";

type Props = {
  item: RatableItem;
  cursor: number;
  total: number;
  expanded?: boolean;
};

function isRow(item: RatableItem): item is { kind: "row"; row: EvidenceRow } {
  return item.kind === "row";
}

export function ChunkHeader({ item, cursor, total, expanded }: Props) {
  const shared = isRow(item) ? item.row : item.candidate;
  const paths = isRow(item)
    ? item.row.source_paths.join(",")
    : `candidate:${(item as { candidate: CandidateRow }).candidate.path}`;
  const scores = isRow(item)
    ? item.row.scores
    : { dense: 0, sparse: 0, [item.candidate.path]: item.candidate.score };
  return (
    <div className="chunk__head">
      <div className="chunk__idline">
        <span className="chunk__idx">
          [{cursor + 1}/{total}]
        </span>
        <span className="chunk__pk">{shared.chunk_id}</span>
        {expanded && <span className="chunk__expanded-badge">expanded</span>}
      </div>
      <div className="chunk__metaline">
        <span className="kv">
          <span className="k">paths:</span> <span className="v">{paths}</span>
        </span>
        <span className="kv">
          <span className="k">counselor:</span>{" "}
          <span className="v">{shared.counselor_id}</span>
        </span>
        <span className="kv">
          <span className="k">term:</span> <span className="v">{shared.term}</span>
        </span>
      </div>
      <div className="chunk__metaline">
        <span className="kv">
          <span className="k">scores:</span>
        </span>
        <span className="kv">
          <span className="k">d=</span>
          <span className="v">{(scores.dense ?? 0).toFixed(4)}</span>
        </span>
        <span className="kv">
          <span className="k">s=</span>
          <span className="v">{(scores.sparse ?? 0).toFixed(4)}</span>
        </span>
      </div>
    </div>
  );
}
