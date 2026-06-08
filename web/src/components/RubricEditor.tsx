import { useEffect, useRef, useState } from "react";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { markdown } from "@codemirror/lang-markdown";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { formatApiError } from "../api/client";
import {
  useDeriveResult,
  useRefineDerive,
  useRefineJudge,
  useSaveRubric,
} from "../hooks/useRefine";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import type { RubricMetadata } from "../types";

type Props = { sid: string };

export function RubricEditor({ sid }: Props) {
  const derive = useDeriveResult(sid, true);
  const save = useSaveRubric(sid);
  const judge = useRefineJudge(sid);
  const retry = useRefineDerive(sid);

  const editorRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const [text, setText] = useState<string>("");
  const [parsedMeta, setParsedMeta] = useState<RubricMetadata | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  // Once derive returns, hydrate the editor + parsed-side panel.
  useEffect(() => {
    if (derive.data && !viewRef.current && editorRef.current) {
      const initialText = derive.data.rubric_text;
      setText(initialText);
      setParsedMeta(derive.data.metadata);

      const view = new EditorView({
        state: EditorState.create({
          doc: initialText,
          extensions: [
            history(),
            lineNumbers(),
            keymap.of([
              ...defaultKeymap,
              ...historyKeymap,
              {
                key: "Mod-s",
                run: () => {
                  onSave();
                  return true;
                },
              },
              {
                key: "Mod-Enter",
                run: () => {
                  onJudge();
                  return true;
                },
              },
            ]),
            markdown(),
            EditorView.updateListener.of((update) => {
              if (update.docChanged) {
                const next = update.state.doc.toString();
                setText(next);
                setDirty(true);
              }
            }),
            EditorView.theme({
              "&": { height: "100%" },
              ".cm-content": { fontFamily: "ui-monospace, monospace" },
              ".cm-scroller": { overflow: "auto" },
            }),
          ],
        }),
        parent: editorRef.current,
      });
      viewRef.current = view;
    }
    return () => {
      // viewRef.current is owned across the editor lifetime — only
      // tear down when the parent unmounts.
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [derive.data]);

  useEffect(() => {
    return () => {
      viewRef.current?.destroy();
      viewRef.current = null;
    };
  }, []);

  function onSave() {
    setParseError(null);
    save.mutate(
      { rubric_text: viewRef.current?.state.doc.toString() ?? text },
      {
        onSuccess: (meta) => {
          setParsedMeta(meta);
          setDirty(false);
        },
        onError: (err) =>
          setParseError(formatApiError(err) ?? "Could not save rubric."),
      },
    );
  }

  function onJudge() {
    if (dirty) {
      setParseError("Save the rubric before running judge (Ctrl+S).");
      return;
    }
    judge.mutate();
  }

  function onRetry() {
    setParseError(null);
    retry.mutate(undefined, {
      onError: (err) =>
        setParseError(formatApiError(err) ?? "Could not retry derive."),
    });
  }

  // Mirror CodeMirror's internal Mod-s / Mod-Enter at the window
  // level so the shortcuts still fire when focus is on a button (e.g.
  // the operator just clicked Save and the editor doesn't have focus).
  // ``useKeyboardShortcuts`` skips inputs/textareas/contentEditable so
  // typing inside CodeMirror itself isn't double-handled.
  useKeyboardShortcuts(
    {
      "ctrl+s": () => {
        if (dirty && !save.isPending) onSave();
      },
      "ctrl+enter": () => {
        if (!dirty && !judge.isPending) onJudge();
      },
    },
    true,
  );

  if (derive.isLoading) {
    return <div className="loading">Loading derived rubric…</div>;
  }
  if (derive.isError) {
    return (
      <div className="loading">
        {formatApiError(derive.error) ?? "Derive result unavailable."}
      </div>
    );
  }

  return (
    <div className="rubric-editor">
      <div className="rubric-editor__top">
        <div>
          <h2 className="modal__title" style={{ marginBottom: 4 }}>
            Rubric editor
          </h2>
          <div className="rubric-editor__meta">
            v{parsedMeta?.version ?? 1}
            {dirty && <span className="rubric-editor__dirty"> · unsaved</span>}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            type="button"
            className="btn"
            onClick={onSave}
            disabled={save.isPending || !dirty}
          >
            <span className="btn__cap">
              {save.isPending ? "Saving…" : "Save"}
            </span>
            <span className="btn__key">[ctrl+s]</span>
          </button>
          <button
            type="button"
            className="btn"
            onClick={onRetry}
            disabled={retry.isPending || save.isPending || judge.isPending}
          >
            <span className="btn__cap">
              {retry.isPending ? "Retrying…" : "Retry"}
            </span>
          </button>
          <button
            type="button"
            className="btn btn--fit"
            onClick={onJudge}
            disabled={judge.isPending || dirty}
          >
            <span className="btn__cap">
              {judge.isPending ? "Starting…" : "Run judge"}
            </span>
            <span className="btn__key">[ctrl+↵]</span>
          </button>
        </div>
      </div>

      {parseError && <p className="modal__error">{parseError}</p>}

      <div className="rubric-editor__main">
        <div ref={editorRef} className="rubric-editor__codearea" />
        <aside className="rubric-editor__sidebar">
          <ChecksList meta={parsedMeta} />
          <ExamplesList
            title="FIT examples"
            examples={parsedMeta?.fit_examples ?? []}
            kind="fit"
          />
          <ExamplesList
            title="NOT_FIT examples"
            examples={parsedMeta?.not_fit_examples ?? []}
            kind="not_fit"
          />
        </aside>
      </div>
    </div>
  );
}

function ChecksList({ meta }: { meta: RubricMetadata | null }) {
  if (!meta) return null;
  return (
    <div className="rubric-editor__section">
      <div className="modal__section-label">Checks</div>
      <ul className="rubric-editor__list">
        {meta.checks.map((c) => (
          <li key={c.id}>
            <strong>{c.id}</strong> — {c.description}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ExamplesList({
  title,
  examples,
  kind,
}: {
  title: string;
  examples: { pk: string | number; span_text: string; fails: string[] | null }[];
  kind: "fit" | "not_fit";
}) {
  if (examples.length === 0) return null;
  return (
    <div className="rubric-editor__section">
      <div className="modal__section-label">{title}</div>
      <ul className="rubric-editor__list">
        {examples.map((ex) => (
          <li key={String(ex.pk)}>
            <code>{String(ex.pk)}</code>
            {kind === "not_fit" && ex.fails && (
              <span className="rubric-editor__fails">
                {" "}
                ({ex.fails.join(", ")})
              </span>
            )}
            <div className="rubric-editor__span">{ex.span_text}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
