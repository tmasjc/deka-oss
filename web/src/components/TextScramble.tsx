import { useEffect, useRef, useState } from "react";

const SCRAMBLE_GLYPHS = "!<>-_\\/[]{}—=+*^?#________";

type TextScrambleProps = {
  text: string;
  // Per-character settle order is randomised; this controls how long
  // (in ms) it takes the slowest character to lock in. Total wall time
  // is roughly this value.
  durationMs?: number;
  className?: string;
  // Cadence between RAF ticks; 50ms feels like a CRT scramble without
  // burning frames.
  frameMs?: number;
  onDone?: () => void;
};

type Frame = {
  from: string;
  to: string;
  start: number;
  end: number;
  glyph: string;
};

/**
 * Scramble each character through random glyphs before settling on the
 * target letter. Per-character start/end offsets are randomised so the
 * effect feels organic rather than a synchronised wave.
 *
 * Pure RAF loop — no animation library. Honours
 * ``prefers-reduced-motion`` by skipping straight to the final string.
 */
export function TextScramble({
  text,
  durationMs = 700,
  className,
  frameMs = 50,
  onDone,
}: TextScrambleProps) {
  const [output, setOutput] = useState("");
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ) {
      setOutput(text);
      onDoneRef.current?.();
      return;
    }

    const frames: Frame[] = [];
    const totalFrames = Math.max(1, Math.round(durationMs / frameMs));
    for (let i = 0; i < text.length; i++) {
      const start = Math.floor(Math.random() * (totalFrames * 0.4));
      const end = start + Math.floor(totalFrames * 0.4 + Math.random() * totalFrames * 0.6);
      frames.push({
        from: " ",
        to: text[i],
        start,
        end,
        glyph: pickGlyph(),
      });
    }

    let frame = 0;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    function tick() {
      let complete = 0;
      let acc = "";
      for (let i = 0; i < frames.length; i++) {
        const f = frames[i];
        if (frame >= f.end) {
          complete++;
          acc += f.to;
        } else if (frame >= f.start) {
          if (Math.random() < 0.28) f.glyph = pickGlyph();
          acc += f.glyph;
        } else {
          acc += f.from;
        }
      }
      setOutput(acc);
      if (complete < frames.length && !cancelled) {
        frame++;
        timer = setTimeout(tick, frameMs);
      } else if (!cancelled) {
        onDoneRef.current?.();
      }
    }
    tick();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [text, durationMs, frameMs]);

  return <span className={className}>{output}</span>;
}

function pickGlyph(): string {
  return SCRAMBLE_GLYPHS[
    Math.floor(Math.random() * SCRAMBLE_GLYPHS.length)
  ];
}
