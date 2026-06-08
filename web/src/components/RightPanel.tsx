import type { WorkflowStep } from "../types";
import { WorkflowTimeline } from "./WorkflowTimeline";

export function RightPanel({ steps }: { steps: WorkflowStep[] }) {
  return (
    <aside className="right">
      <WorkflowTimeline steps={steps} />
    </aside>
  );
}
