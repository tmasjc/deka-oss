import { useQuery } from "@tanstack/react-query";
import { formatApiError } from "../api/client";
import { sessionApi } from "../api/session";

type Props = {
  sid: string;
  onClose: () => void;
};

export function ReflectionModal({ sid, onClose }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["reflection", sid],
    queryFn: () => sessionApi.reflection(sid),
    retry: false,
  });

  return (
    <div
      className="modal__backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-modal="true">
        <h2 className="modal__title">Reflection</h2>
        {isLoading && <p>Loading…</p>}
        {error && (
          <p className="modal__error">
            {formatApiError(error) || "Failed to load reflection"}
          </p>
        )}
        {data && (
          <>
            <Section label="Observe" body={data.observe} />
            <Section label="Diagnose" body={data.diagnose} />
            <Section label="Hypothesis" body={data.hypothesis} />
            {data.previous_hypothesis_verdict && (
              <Section
                label="Previous hypothesis"
                body={data.previous_hypothesis_verdict}
              />
            )}
            {data.status === "CONVERGED" && (
              <Section
                label="Status"
                body={
                  data.turns_to_converge
                    ? `CONVERGED — ${data.turns_to_converge} turns`
                    : "CONVERGED"
                }
              />
            )}
          </>
        )}
        <p className="modal__hint">Press Esc to close.</p>
      </div>
    </div>
  );
}

function Section({ label, body }: { label: string; body: string | null | undefined }) {
  if (!body) return null;
  return (
    <div className="modal__section">
      <div className="modal__section-label">{label}</div>
      <div className="modal__section-body">{body}</div>
    </div>
  );
}
