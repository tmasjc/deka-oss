import type { EvidenceTable } from "../types";

type Props = { table: EvidenceTable; onDismiss: () => void };

export function FilterBanner({ table, onDismiss }: Props) {
  const parts: string[] = [];
  if (table.filtered_short_chunk > 0)
    parts.push(`${table.filtered_short_chunk} short chunk(s)`);
  if (table.filtered_duplicate_sample > 0)
    parts.push(`${table.filtered_duplicate_sample} duplicate-sample chunk(s)`);
  if (table.dropped_by_extractor > 0)
    parts.push(`${table.dropped_by_extractor} extractor failure(s)`);
  if (parts.length === 0) return null;

  return (
    <div className="banner">
      This turn dropped {parts.join(", ")}.{" "}
      <button type="button" className="tab" onClick={onDismiss}>
        dismiss
      </button>
    </div>
  );
}
