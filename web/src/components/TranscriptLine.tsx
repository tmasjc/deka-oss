import { Fragment } from "react";
import { Chip } from "./Chip";

// Matches patterns like "[入学档案]" in the transcript body so they render as chips.
const CHIP_RE = /\[([^\]\n]+)\]/g;

function renderWithChips(text: string) {
  const parts: (string | { chip: string })[] = [];
  let cursor = 0;
  for (const match of text.matchAll(CHIP_RE)) {
    const start = match.index ?? 0;
    if (start > cursor) parts.push(text.slice(cursor, start));
    parts.push({ chip: match[1] });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts;
}

export type TranscriptLineProps = {
  line: string;
  highlighted?: boolean;
};

export function TranscriptLine({ line, highlighted }: TranscriptLineProps) {
  const sepIdx = line.indexOf(":");
  const hasSpeaker =
    sepIdx > 0 && sepIdx < 16 && !line.slice(0, sepIdx).includes(" ");
  const who = hasSpeaker ? line.slice(0, sepIdx) : "";
  const body = hasSpeaker ? line.slice(sepIdx + 1).trimStart() : line;

  return (
    <div className={"tline" + (highlighted ? " tline--hl" : "")}>
      {hasSpeaker && (
        <>
          <span className="who">{who}</span>
          <span className="sep">: </span>
        </>
      )}
      <span className="body">
        {renderWithChips(body).map((part, i) =>
          typeof part === "string" ? (
            <Fragment key={i}>{part}</Fragment>
          ) : (
            <Chip key={i}>{part.chip}</Chip>
          ),
        )}
      </span>
    </div>
  );
}
