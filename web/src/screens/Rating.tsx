import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, downloadBlob, formatApiError } from "../api/client";
import { HeaderBar } from "../components/HeaderBar";
import { LeftPanel } from "../components/LeftPanel";
import { RightPanel } from "../components/RightPanel";
import { FooterBar } from "../components/FooterBar";
import { ChunkHeader } from "../components/ChunkHeader";
import { ChunkCard } from "../components/ChunkCard";
import { ActionRow } from "../components/ActionRow";
import { ReflectionModal } from "../components/ReflectionModal";
import { ConfigEditorModal } from "../components/ConfigEditorModal";
import {
  TurnAdvanceOverlay,
  type OverlayDiagnostics,
} from "../components/TurnAdvanceOverlay";
import { FilterBanner } from "../components/FilterBanner";
import { HarvestConfirmModal } from "../components/HarvestConfirmModal";
import { HarvestProgressOverlay } from "../components/HarvestProgressOverlay";
import { HarvestSummary } from "../components/HarvestSummary";
import { ProbeBanner } from "../components/ProbeBanner";
import { RecommendationBanner } from "../components/RecommendationBanner";
import { RefineConfirmModal } from "../components/RefineConfirmModal";
import { RefineProgressOverlay } from "../components/RefineProgressOverlay";
import { ApplyDoneSummary } from "../components/ApplyDoneSummary";
import { ApplyProgressOverlay } from "../components/ApplyProgressOverlay";
import { ReplayLoader } from "../components/ReplayLoader";
import { RefineSummary } from "../components/RefineSummary";
import { RubricEditor } from "../components/RubricEditor";
import { ThresholdCalibrationPanel } from "../components/ThresholdCalibrationPanel";
import { VerdictReviewPanel } from "../components/VerdictReviewPanel";
import {
  useDropPath,
  useEndSession,
  useNextTurn,
  useRate,
  useRecommendationDecision,
  useSession,
  useTriggerAudit,
} from "../hooks/useSession";
import { useRefineSummary } from "../hooks/useRefine";
import { useApplyTrain, useApplySummary } from "../hooks/useApply";
import { useReplayAdvance } from "../hooks/useSessions";
import { useProgress } from "../hooks/useProgress";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";
import { useUi } from "../state/ui";
import type {
  ApplySummary,
  RatableItem,
  Rating,
  RecommendationDecision,
  RefineSummary as RefineSummaryT,
  Reflection,
  SessionSnapshot,
} from "../types";

type OverlayPhase = "idle" | "pending" | "flash";
const FLASH_MS = 1200;
const REPLAY_LOADER_MS = 3000;
const OBSERVE_MAX_CHARS = 100;

function buildDiagnostics(
  snap: SessionSnapshot,
  reflection: Reflection | null | undefined,
): OverlayDiagnostics {
  const trend = snap.precision_trend;
  const observe = reflection?.observe ?? null;
  return {
    fetched: snap.table.rows.length,
    precision: trend.length > 0 ? trend[trend.length - 1] : null,
    observe:
      observe && observe.length > OBSERVE_MAX_CHARS
        ? observe.slice(0, OBSERVE_MAX_CHARS - 1).trimEnd() + "…"
        : observe,
    converged: snap.convergence.converged,
  };
}

function flattenItems(snap: SessionSnapshot): RatableItem[] {
  const items: RatableItem[] = snap.table.rows.map((row) => ({
    kind: "row" as const,
    row,
  }));
  // Per-path candidates are presented for rating only on audit turns
  // (the operator triggered ``POST /audit``). Outside audit mode they
  // are still in the snapshot but stay hidden — mirrors the TUI.
  if (snap.audit_mode_active) {
    for (const path of ["dense", "sparse"] as const) {
      for (const cand of snap.table.per_path_candidates[path] ?? []) {
        items.push({ kind: "candidate" as const, candidate: cand });
      }
    }
  }
  return items;
}

export function RatingScreen({ sid }: { sid: string }) {
  const { data: snap, isLoading, error } = useSession(sid);
  const rate = useRate(sid);
  const nextTurn = useNextTurn(sid);
  const end = useEndSession();

  const triggerAudit = useTriggerAudit(sid);
  const dropPath = useDropPath(sid);
  const recommendationDecision = useRecommendationDecision(sid);
  const replayAdvance = useReplayAdvance(sid);

  const cursor = useUi((s) => s.cursor);
  const setCursor = useUi((s) => s.setCursor);
  const bumpCursor = useUi((s) => s.bumpCursor);
  const reflectionOpen = useUi((s) => s.reflectionOpen);
  const setReflectionOpen = useUi((s) => s.setReflectionOpen);
  const configOpen = useUi((s) => s.configOpen);
  const setConfigOpen = useUi((s) => s.setConfigOpen);
  const banner = useUi((s) => s.banner);
  const setBanner = useUi((s) => s.setBanner);
  const setSession = useUi((s) => s.setSession);
  const filtersBannerDismissed = useUi((s) => s.filtersBannerDismissed);
  const dismissFilterBanner = useUi((s) => s.dismissFilterBanner);
  const probeBannerDismissed = useUi((s) => s.probeBannerDismissed);
  const dismissProbeBanner = useUi((s) => s.dismissProbeBanner);
  const recommendationDismissed = useUi((s) => s.recommendationDismissed);
  const dismissRecommendation = useUi((s) => s.dismissRecommendation);
  const expandedPks = useUi((s) => s.expandedPks);
  const originalCache = useUi((s) => s.originalCache);
  const expandingPks = useUi((s) => s.expandingPks);
  const toggleExpanded = useUi((s) => s.toggleExpanded);

  const [overlayPhase, setOverlayPhase] = useState<OverlayPhase>("idle");
  const [harvestModalOpen, setHarvestModalOpen] = useState(false);
  const [harvestModalSeen, setHarvestModalSeen] = useState(false);
  const [refineModalOpen, setRefineModalOpen] = useState(false);
  const [refineModalSeen, setRefineModalSeen] = useState(false);
  const [refineSummary, setRefineSummary] = useState<RefineSummaryT | null>(
    null,
  );
  const [applySummary, setApplySummary] = useState<ApplySummary | null>(null);
  const [applyDefaultThreshold, setApplyDefaultThreshold] = useState<number>(
    0.7,
  );
  // Frontend-only: when in DONE phase, lets the operator pop back to a
  // read-only verdict browser without touching the session state. The
  // verdicts are still served by GET /refine/verdicts (DONE_VIEW
  // resume re-hydrates them from disk).
  const [revisitingVerdicts, setRevisitingVerdicts] = useState(false);
  const applyTrain = useApplyTrain(sid);
  // Resume-from-disk fallback: when phase=DONE but the local
  // ``refineSummary`` state was never populated (DONE_VIEW resume,
  // since the live finalize mutation didn't run this session), fetch
  // the same DTO from GET /refine/summary so we don't fall through to
  // the candidate-rating UI.
  const refineSummaryQuery = useRefineSummary(
    sid,
    snap?.phase === "DONE" && refineSummary === null,
  );
  const effectiveRefineSummary: RefineSummaryT | null =
    refineSummary ?? refineSummaryQuery.data ?? null;
  // Mirror of the refine fallback: on DONE_VIEW resume the live
  // finalize mutation never ran in this session, so the local
  // ``applySummary`` state is null. Fetch the same DTO from
  // GET /apply/summary (the resume code rehydrates the in-memory
  // apply_state from phase4 sidecars, so the endpoint returns 200
  // instead of 404). Without this the conditional render below falls
  // through to ``effectiveRefineSummary`` and the user sees the
  // Phase 3 view on a fully-applied session.
  const applySummaryQuery = useApplySummary(
    sid,
    snap?.phase === "DONE" && applySummary === null,
  );
  const effectiveApplySummary: ApplySummary | null =
    applySummary ?? applySummaryQuery.data ?? null;
  const progressQuery = useProgress(sid, overlayPhase === "pending");
  const flashTimerRef = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (flashTimerRef.current != null)
        window.clearTimeout(flashTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (error instanceof ApiError && error.status === 404) {
      window.history.pushState(null, "", "/");
      setSession(null);
      setBanner({
        kind: "error",
        text: "Session no longer exists (server restarted). Start a new one.",
      });
    }
  }, [error, setSession, setBanner]);

  // Surface the harvest confirm modal once the session converges.
  // Only auto-open once per session — if the user dismisses or runs,
  // ``harvestModalSeen`` keeps it closed on subsequent renders.
  // Replay walks past the convergence boundary via /replay/advance,
  // so the modal is intentionally suppressed in replay.
  useEffect(() => {
    if (
      snap?.convergence.converged &&
      snap.phase === "TUNING" &&
      !snap.replay &&
      !harvestModalSeen
    ) {
      setHarvestModalOpen(true);
      setHarvestModalSeen(true);
    }
  }, [snap, harvestModalSeen]);

  // The refine modal does NOT auto-open on ANCHOR_DONE — the
  // operator should review the harvest summary first. The
  // ``HarvestSummary``'s "Continue → Refine" button is the entry
  // point. ``refineModalSeen`` still gates re-opens so dismissing
  // and ending the session doesn't surface the modal again.

  const items = useMemo(() => (snap ? flattenItems(snap) : []), [snap]);
  const safeCursor = items.length
    ? Math.min(cursor, items.length - 1)
    : 0;
  const current = items[safeCursor];

  function onRate(rating: Rating) {
    if (!current || rate.isPending) return;
    if (snap?.replay) return;
    const payload =
      current.kind === "row"
        ? { rank: current.row.rank, rating }
        : {
            path: current.candidate.path,
            rank_in_path: current.candidate.rank_in_path,
            rating,
          };
    rate.mutate(payload, {
      onSuccess: () => {
        // Auto-advance to next unrated item; fall back to next index.
        const nextIdx = findNextUnrated(items, safeCursor);
        if (nextIdx !== null) setCursor(nextIdx);
        else bumpCursor();
      },
      onError: (err) => setBanner({ kind: "error", text: formatApiError(err) }),
    });
  }

  function onNextTurn() {
    if (!snap || overlayPhase !== "idle") return;
    // Replay: 'a' walks to the next replay section. Same affordance,
    // but races a 3 s minimum delay against the network round-trip so
    // the unified "in replay-mode..." overlay is always on screen for
    // at least 3 s. Skips the turn_complete / convergence checks the
    // live path needs — replay sessions have already converged.
    if (snap.replay) {
      setOverlayPhase("pending");
      const minDelay = new Promise<void>((resolve) =>
        window.setTimeout(resolve, REPLAY_LOADER_MS),
      );
      Promise.all([minDelay, replayAdvance.mutateAsync()])
        .then(() => setOverlayPhase("idle"))
        .catch((err) => {
          setOverlayPhase("idle");
          setBanner({ kind: "error", text: formatApiError(err) });
        });
      return;
    }
    if (!snap.turn_complete) return;
    setOverlayPhase("pending");
    nextTurn.mutate(undefined, {
      onSuccess: (res) => {
        if (res.snapshot.convergence.converged) {
          setOverlayPhase("idle");
          return;
        }
        setOverlayPhase("flash");
        flashTimerRef.current = window.setTimeout(() => {
          setOverlayPhase("idle");
          flashTimerRef.current = null;
        }, FLASH_MS);
      },
      onError: (err) => {
        setOverlayPhase("idle");
        setBanner({ kind: "error", text: formatApiError(err) });
      },
    });
  }

  function onQuit() {
    end.mutate(sid);
  }

  function onToggleExpand() {
    if (!current) return;
    const pk = current.kind === "row" ? current.row.pk : current.candidate.pk;
    void toggleExpanded(pk, sid);
  }

  function onRecommendationDecision(decision: RecommendationDecision) {
    if (!snap || recommendationDecision.isPending) return;
    if (snap.replay) return;
    const recTurn = snap.turn_number - 1;
    recommendationDecision.mutate(
      { decision },
      {
        onSuccess: () => {
          dismissRecommendation(recTurn);
        },
        onError: (err) =>
          setBanner({ kind: "error", text: formatApiError(err) }),
      },
    );
  }

  function onDownloadLogs() {
    downloadBlob(
      `/api/session/${sid}/logs/download`,
      `session-${sid.slice(0, 8)}.zip`,
    ).catch((err) => setBanner({ kind: "error", text: formatApiError(err) }));
  }

  function onTriggerAudit() {
    if (!snap || snap.audit_mode_active) return;
    if (snap.replay) return;
    if (!snap.table) return;
    triggerAudit.mutate(undefined, {
      onSuccess: () =>
        setBanner({
          kind: "info",
          text: "Audit mode: rate the per-path candidates, then 'o' to drop a path.",
        }),
      onError: (err) => setBanner({ kind: "error", text: formatApiError(err) }),
    });
  }

  function onDropPath() {
    if (snap?.replay) return;
    if (!snap || !snap.audit_mode_active) {
      setBanner({
        kind: "error",
        text: "Path drop is only available in audit mode (press 'p' first).",
      });
      return;
    }
    if (!snap.turn_complete) {
      setBanner({
        kind: "error",
        text: "Rate every fused row and every candidate before dropping a path.",
      });
      return;
    }
    const active = snap.params.active_paths;
    if (active.length <= 1) {
      setBanner({ kind: "error", text: "Cannot drop the last active path." });
      return;
    }
    const choice = window.prompt(
      `Drop which path? (${active.join(" / ")})`,
      active[0],
    );
    if (!choice) return;
    if (!active.includes(choice as (typeof active)[number])) {
      setBanner({
        kind: "error",
        text: `${choice} is not currently active.`,
      });
      return;
    }
    dropPath.mutate(
      { path: choice as (typeof active)[number] },
      {
        onSuccess: (snapshot) =>
          setBanner({
            kind: "info",
            text: `Dropped ${choice}. Active paths: ${snapshot.params.active_paths.join(",")}.`,
          }),
        onError: (err) => setBanner({ kind: "error", text: formatApiError(err) }),
      },
    );
  }

  useKeyboardShortcuts(
    {
      j: () => {
        if (items.length > 0) setCursor(Math.max(safeCursor - 1, 0));
      },
      arrowdown: () => {
        if (items.length > 0) setCursor(Math.max(safeCursor - 1, 0));
      },
      k: () => {
        if (items.length > 0) setCursor(Math.min(safeCursor + 1, items.length - 1));
      },
      arrowup: () => {
        if (items.length > 0) setCursor(Math.min(safeCursor + 1, items.length - 1));
      },
      f: () => onRate("FIT"),
      n: () => onRate("NOT_FIT"),
      d: () => onRate("DISCARD"),
      a: () => onNextTurn(),
      r: () => setReflectionOpen(true),
      e: () => setConfigOpen(true),
      x: () => onToggleExpand(),
      p: () => onTriggerAudit(),
      o: () => onDropPath(),
      "ctrl+l": () => onDownloadLogs(),
      q: () => onQuit(),
      escape: () => {
        setReflectionOpen(false);
        setConfigOpen(false);
      },
    },
    !!snap &&
      overlayPhase === "idle" &&
      // In replay, 'a' must work past TUNING too so the user can walk
      // through the HARVEST / REFINE / APPLY summary views. Outside
      // replay the shortcut block is scoped to TUNING (the original
      // intent: rating + advance keys only apply mid-loop).
      (snap.phase === "TUNING" ||
        (snap.replay &&
          (snap.phase === "ANCHOR_DONE" || snap.phase === "DONE"))),
  );

  if (isLoading) return <div className="loading">Loading session…</div>;
  // 404 path returns to QueryEntry via setSession(null) in the effect above.
  if (error || !snap)
    return (
      <div className="loading">
        {formatApiError(error) ?? "Loading session…"}
      </div>
    );

  return (
    <div className="app">
      <HeaderBar
        turn={snap.turn_number}
        audit={snap.audit_mode_active}
        phase={snap.phase}
      />
      {(() => {
        // The recommendation lives on the just-completed turn's
        // reflection (turn_number - 1, since the snapshot has already
        // rolled forward). Hide once the operator decides (snapshot
        // re-fetch elides it via the consumed marker), once audit
        // mode is already active for this turn, or once the session
        // has converged.
        const rec = nextTurn.data?.reflection?.path_drop_recommendation;
        const recTurn = snap.turn_number - 1;
        const show =
          rec &&
          !snap.audit_mode_active &&
          !snap.convergence.converged &&
          !recommendationDismissed.has(recTurn);
        return show ? (
          <RecommendationBanner
            recommendation={rec}
            onDecision={onRecommendationDecision}
            disabled={recommendationDecision.isPending}
          />
        ) : null;
      })()}
      {snap.probe_summary && !probeBannerDismissed && (
        <ProbeBanner
          probe={snap.probe_summary}
          onDismiss={dismissProbeBanner}
        />
      )}
      {!filtersBannerDismissed.has(snap.turn_number) && (
        <FilterBanner
          table={snap.table}
          onDismiss={() => dismissFilterBanner(snap.turn_number)}
        />
      )}
      {banner && (
        <div className={"banner" + (banner.kind === "error" ? " banner--error" : "")}>
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
      <div className="app__main">
        <LeftPanel snap={snap} />
        <section className="center">
          {snap.phase === "DONE" ? (
            effectiveApplySummary ? (
              // Phase 4 owns the screen once the summary is in hand.
              // The pre-Ship confirm modal already fired before
              // finalize, so there's no separate post-finalize gate to
              // wait on. Scramble plays on the live-finalize landing
              // (!read_only); resume skips the animation.
              <ApplyDoneSummary
                sid={sid}
                summary={effectiveApplySummary}
                playScramble={!snap.read_only}
                readOnly={snap.read_only}
                onEnd={() => {
                  if (snap.read_only) {
                    // No DELETE on read-only resume —
                    // _require_writable would 409. Just navigate home.
                    window.history.pushState(null, "", "/");
                    setSession(null);
                  } else {
                    end.mutate(sid);
                  }
                }}
              />
            ) : revisitingVerdicts ? (
              <VerdictReviewPanel
                sid={sid}
                mode="revisit"
                onBack={() => setRevisitingVerdicts(false)}
                onFinalized={() => {}}
              />
            ) : effectiveRefineSummary ? (
              <RefineSummary
                sid={sid}
                summary={effectiveRefineSummary}
                onEnd={onQuit}
                onRevisit={() => setRevisitingVerdicts(true)}
                // Hide the Apply CTA on read-only (already-shipped)
                // sessions — clicking it would 409 with "Session state
                // conflict" since _require_writable rejects on
                // read_only contexts.
                onApply={
                  snap.read_only
                    ? undefined
                    : () => {
                        applyTrain.mutate(undefined, {
                          onError: (err) =>
                            setBanner({
                              kind: "error",
                              text:
                                formatApiError(err) ??
                                "Apply training failed.",
                            }),
                        });
                      }
                }
              />
            ) : refineSummaryQuery.isError ? (
              <div className="loading">
                {formatApiError(refineSummaryQuery.error) ||
                  "Could not load refine summary."}
              </div>
            ) : (
              <div className="loading">Loading refine summary…</div>
            )
          ) : snap.phase === "APPLY_TRAINING" ||
            snap.phase === "APPLY_PREPARING" ||
            snap.phase === "APPLY_APPLYING" ? (
            // Inline content stays empty while the overlay is up — same
            // pattern as ANCHOR_RUNNING / REFINE_DERIVING / REFINE_JUDGING.
            null
          ) : snap.phase === "APPLY_REVIEW" ? (
            <ThresholdCalibrationPanel
              sid={sid}
              defaultThreshold={applyDefaultThreshold}
              onFinalized={(summary) => setApplySummary(summary)}
              onCancelled={() => {
                setApplyDefaultThreshold(0.7);
              }}
            />
          ) : snap.phase === "APPLY_FAILED" ? (
            <div className="loading">
              Apply failed — check the run log for details.
            </div>
          ) : snap.phase === "REFINE_REVIEW" ? (
            <VerdictReviewPanel
              sid={sid}
              onFinalized={(summary) => setRefineSummary(summary)}
            />
          ) : snap.phase === "REFINE_EDITING" ? (
            <RubricEditor sid={sid} />
          ) : snap.phase === "REFINE_FAILED" ? (
            <div className="loading">
              Refine failed — check the run log for details.
            </div>
          ) : snap.phase === "ANCHOR_DONE" ? (
            <HarvestSummary
              sid={sid}
              onEnd={onQuit}
              onContinue={
                snap.replay || refineModalSeen
                  ? undefined
                  : () => {
                      setRefineModalOpen(true);
                      setRefineModalSeen(true);
                    }
              }
            />
          ) : snap.phase === "ANCHOR_FAILED" ? (
            <div className="loading">
              Harvest failed — see /api/session/{sid}/harvest/result for details.
            </div>
          ) : current ? (
            (() => {
              const currentPk =
                current.kind === "row" ? current.row.pk : current.candidate.pk;
              const currentKey = String(currentPk);
              const isExpanded = expandedPks.has(currentKey);
              const isExpanding = expandingPks.has(currentKey);
              const expandedText = isExpanded
                ? originalCache.get(currentKey)
                : undefined;
              return (
                <>
                  <ChunkHeader
                    item={current}
                    cursor={safeCursor}
                    total={items.length}
                    expanded={isExpanded && expandedText !== undefined}
                  />
                  <ChunkCard
                    content={
                      current.kind === "row"
                        ? current.row.chunk_content
                        : current.candidate.chunk_content
                    }
                    spanLines={
                      current.kind === "row"
                        ? current.row.span_line_indices
                        : current.candidate.span_line_indices
                    }
                    rating={
                      current.kind === "row"
                        ? current.row.rating
                        : current.candidate.rating
                    }
                    originalContent={expandedText}
                    isExpanding={isExpanding}
                  />
                  <ActionRow
                    onFit={() => onRate("FIT")}
                    onNotFit={() => onRate("NOT_FIT")}
                    onDiscard={() => onRate("DISCARD")}
                    disabled={rate.isPending || snap.replay}
                  />
                </>
              );
            })()
          ) : (
            <div className="loading">No candidates in this turn.</div>
          )}
          {snap.phase === "TUNING" && snap.turn_complete && (
            <div className="query-form__hint" style={{ marginTop: 12 }}>
              All rated. Press{" "}
              <span className="kc__k" style={{ padding: "1px 5px" }}>
                a
              </span>{" "}
              to run the next turn.
            </div>
          )}
        </section>
        <RightPanel steps={snap.workflow} />
      </div>
      <FooterBar params={snap.params} phase={snap.phase} />
      {reflectionOpen && (
        <ReflectionModal sid={sid} onClose={() => setReflectionOpen(false)} />
      )}
      {configOpen && (
        <ConfigEditorModal
          sid={sid}
          params={snap.params}
          onClose={() => setConfigOpen(false)}
        />
      )}
      {overlayPhase !== "idle" &&
        (snap.replay ? (
          <ReplayLoader />
        ) : (
          <TurnAdvanceOverlay
            phase={overlayPhase}
            pendingTitle={`Advancing to turn ${snap.turn_number + 1}`}
            flashTitle={`Turn ${snap.turn_number} ready`}
            diagnostics={
              overlayPhase === "flash"
                ? buildDiagnostics(snap, nextTurn.data?.reflection)
                : null
            }
            progress={
              overlayPhase === "pending" ? (progressQuery.data ?? null) : null
            }
          />
        ))}
      {harvestModalOpen && !snap.replay && (
        <HarvestConfirmModal
          sid={sid}
          onClose={() => setHarvestModalOpen(false)}
          onConfirmed={() => setHarvestModalOpen(false)}
        />
      )}
      {refineModalOpen && !snap.replay && (
        <RefineConfirmModal
          sid={sid}
          onClose={() => setRefineModalOpen(false)}
          onConfirmed={() => setRefineModalOpen(false)}
        />
      )}
      {/* Live phase-progress overlays are noisy and stage-specific —
          replay collapses them all into the single ReplayLoader above. */}
      {!snap.replay && snap.phase === "ANCHOR_RUNNING" && (
        <HarvestProgressOverlay sid={sid} />
      )}
      {!snap.replay && snap.phase === "REFINE_DERIVING" && (
        <RefineProgressOverlay sid={sid} mode="deriving" />
      )}
      {!snap.replay && snap.phase === "REFINE_JUDGING" && (
        <RefineProgressOverlay sid={sid} mode="judging" />
      )}
      {!snap.replay &&
        (snap.phase === "APPLY_TRAINING" ||
          snap.phase === "APPLY_PREPARING" ||
          snap.phase === "APPLY_APPLYING") && (
          <ApplyProgressOverlay sid={sid} phase={snap.phase} />
        )}
    </div>
  );
}

function findNextUnrated(items: RatableItem[], from: number): number | null {
  for (let i = from + 1; i < items.length; i++) {
    const it = items[i];
    const rating = it.kind === "row" ? it.row.rating : it.candidate.rating;
    if (rating == null) return i;
  }
  for (let i = 0; i <= from && i < items.length; i++) {
    const it = items[i];
    const rating = it.kind === "row" ? it.row.rating : it.candidate.rating;
    if (rating == null) return i;
  }
  return null;
}
