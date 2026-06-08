import type { BreakdownRow, Convergence } from "../../types";
import { ConvergencePanel } from "../ConvergencePanel";
import { PhaseConfigPanel } from "../PhaseConfigPanel";
import { useConfigDefaults } from "../../hooks/useConfigDefaults";
import { useSessionOverrides } from "../../hooks/useSessionOverrides";

type Props = {
  sid: string;
  convergence: Convergence;
  breakdown: Record<string, BreakdownRow>;
};

export function TurnTab({ sid, convergence, breakdown }: Props) {
  const defaultsQuery = useConfigDefaults();
  const overridesQuery = useSessionOverrides(sid);

  const defaults = defaultsQuery.data as
    | Record<string, Record<string, unknown>>
    | undefined;
  const overrides = overridesQuery.data;

  return (
    <>
      <ConvergencePanel convergence={convergence} breakdown={breakdown} />
      <PhaseConfigPanel
        title="Harvest (Phase 2)"
        defaults={defaults?.harvest}
        overrides={overrides?.harvest}
      />
      <PhaseConfigPanel
        title="Refine (Phase 3)"
        defaults={defaults?.refine}
        overrides={overrides?.refine}
      />
      <PhaseConfigPanel
        title="Apply (Phase 4)"
        defaults={defaults?.apply}
        overrides={overrides?.apply}
      />
    </>
  );
}
