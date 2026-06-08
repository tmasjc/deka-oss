import type { SessionSnapshot } from "../types";
import { useUi, type LeftTab } from "../state/ui";
import { TurnTab } from "./tabs/TurnTab";
import { PKTab } from "./tabs/PKTab";
import { PathsTab } from "./tabs/PathsTab";
import { ConfigTab } from "./tabs/ConfigTab";

type Props = {
  snap: SessionSnapshot;
};

const TABS: Array<{ key: LeftTab; label: string }> = [
  { key: "turn", label: "Turn" },
  { key: "pk", label: "P@K" },
  { key: "paths", label: "Paths" },
  { key: "config", label: "Config" },
];

export function LeftPanel({ snap }: Props) {
  const activeTab = useUi((s) => s.activeTab);
  const setActiveTab = useUi((s) => s.setActiveTab);

  return (
    <aside className="left">
      <div className="left__queryblk">
        <div className="left__qlabel">
          <span className="bullet bullet--done">●</span> Query:
        </div>
        <div className="left__qtext">{snap.query}</div>
      </div>
      <nav className="left__tabs">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            className={"tab" + (activeTab === t.key ? " tab--on" : "")}
            onClick={() => setActiveTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="left__body">
        {activeTab === "turn" && (
          <TurnTab
            sid={snap.session_id}
            convergence={snap.convergence}
            breakdown={snap.breakdown_cumulative}
          />
        )}
        {activeTab === "pk" && (
          <PKTab
            trend={snap.precision_trend}
            threshold={snap.convergence.pk_threshold}
          />
        )}
        {activeTab === "paths" && (
          <PathsTab turns={snap.breakdown_by_turn} />
        )}
        {activeTab === "config" && (
          <ConfigTab params={snap.params} drop={snap.drop_impact_preview} />
        )}
      </div>
      <div className="left__spacer" />
    </aside>
  );
}
