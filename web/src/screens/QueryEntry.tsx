import { useState } from "react";
import { formatApiError } from "../api/client";
import { randomUUID } from "../lib/uuid";
import { useStartSession } from "../hooks/useSession";
import { useProgress } from "../hooks/useProgress";
import { useScopes } from "../hooks/useScopes";
import { HeaderBar } from "../components/HeaderBar";
import { ParamOverridesModal } from "../components/ParamOverridesModal";
import { ScopePicker } from "../components/ScopePicker";
import {
  TurnAdvanceOverlay,
  START_STAGES,
} from "../components/TurnAdvanceOverlay";
import { Preflight } from "./Preflight";
import { useParamOverrides } from "../state/paramOverrides";
import { useUi } from "../state/ui";

type SubmitPhase = "form" | "preflight";

export function QueryEntry() {
  const start = useStartSession();
  const banner = useUi((s) => s.banner);
  const setBanner = useUi((s) => s.setBanner);
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState<string | null>(null);
  const [pendingSid, setPendingSid] = useState<string | null>(null);
  const [phase, setPhase] = useState<SubmitPhase>("form");
  const [paramsOpen, setParamsOpen] = useState(false);
  const overrideCount = useParamOverrides((s) => s.diffCount());
  const buildOverrides = useParamOverrides((s) => s.buildSubmission);
  const progressQuery = useProgress(pendingSid, start.isPending);
  const scopesQuery = useScopes();

  const canSubmit = !!query.trim() && scope !== null;

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || !scope) return;
    setPhase("preflight");
  }

  function onPreflightPass() {
    if (!scope) return;
    const sid = randomUUID();
    setPendingSid(sid);
    start.mutate({
      query: query.trim(),
      scope: scope,
      session_id: sid,
      overrides: buildOverrides(),
    });
  }

  if (phase === "preflight" && scope) {
    return (
      <div className="app">
        <HeaderBar turn={null} />
        <Preflight
          scope={scope}
          onPass={onPreflightPass}
          onCancel={() => setPhase("form")}
        />
        {start.isPending && (
          <TurnAdvanceOverlay
            phase="pending"
            pendingTitle="Starting session"
            stages={START_STAGES}
            progress={progressQuery.data ?? null}
          />
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <HeaderBar turn={null} />
      {banner && (
        <div
          className={
            "banner" + (banner.kind === "error" ? " banner--error" : "")
          }
        >
          {banner.text}{" "}
          <button
            type="button"
            className="tab"
            onClick={() => setBanner(null)}
            style={{ marginLeft: 10 }}
          >
            dismiss
          </button>
        </div>
      )}
      <section className="query-screen">
        <form className="query-form" onSubmit={onSubmit}>
          <FormLabel done={!!query.trim()}>Enter query</FormLabel>
          <div className="query-form__title">
            What concept do you want to pull from the corpus?
          </div>
          <textarea
            className="query-form__textarea"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. 家长对课程效果的疑虑"
            autoFocus
          />
          <ScopePicker
            scopes={scopesQuery.data?.scopes ?? []}
            value={scope}
            onChange={setScope}
            isLoading={scopesQuery.isLoading}
            error={
              scopesQuery.error
                ? formatApiError(scopesQuery.error) || "request failed"
                : null
            }
          />
          <div className="query-form__hint">
            Enter to submit. The agent will probe each retrieval path, run a
            fused search, and load the first turn's candidates for rating.
          </div>
          {start.isError && (
            <div className="query-form__error">
              {formatApiError(start.error) || "Failed to start session"}
            </div>
          )}
          <div className="query-form__row">
            <button
              type="button"
              className="btn"
              onClick={() => setParamsOpen(true)}
              disabled={start.isPending}
            >
              <span className="btn__cap">
                Edit parameters
                {overrideCount > 0 ? ` (${overrideCount})` : ""}
              </span>
            </button>
            <button
              type="submit"
              className="btn btn--fit"
              disabled={start.isPending || !canSubmit}
            >
              <span className="btn__cap">
                {start.isPending ? "Starting…" : "Start session"}
              </span>
              <span className="btn__key">[↵]</span>
            </button>
          </div>
        </form>
      </section>
      {paramsOpen && <ParamOverridesModal onClose={() => setParamsOpen(false)} />}
      {start.isPending && (
        <TurnAdvanceOverlay
          phase="pending"
          pendingTitle="Starting session"
          stages={START_STAGES}
          progress={progressQuery.data ?? null}
        />
      )}
    </div>
  );
}

function FormLabel({
  done,
  children,
}: {
  done: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="query-form__label">
      <span className={"bullet" + (done ? " bullet--done" : "")}>
        {done ? "●" : "○"}
      </span>{" "}
      {children}
    </div>
  );
}
