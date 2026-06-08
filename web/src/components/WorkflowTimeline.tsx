import type { WorkflowStep } from "../types";

type Props = { steps: WorkflowStep[] };

export function WorkflowTimeline({ steps }: Props) {
  return (
    <div className="wf">
      <div className="wf__title">Workflow</div>
      <div className="wf__track">
        {steps.map((step, i) => (
          <div key={step.key} className={`wf__step wf__step--${step.status}`}>
            <div className="wf__node">
              <div className="wf__dot" />
              {i < steps.length - 1 && <div className="wf__line" />}
            </div>
            <div className="wf__meta">
              <div className="wf__label">{step.label}</div>
              {step.detail && <div className="wf__detail">{step.detail}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
