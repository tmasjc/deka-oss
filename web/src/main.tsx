import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./App";
import { useUi } from "./state/ui";
import "./styles/global.css";
import "./styles/variants.css";
import "./styles/theme.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

const SID_PATTERN = /^\/s\/([a-f0-9]+)$/;

function sidFromLocation(): string | null {
  const match = window.location.pathname.match(SID_PATTERN);
  return match ? match[1] : null;
}

// Initial URL → session state (before first render).
const initialSid = sidFromLocation();
if (initialSid) useUi.getState().setSession(initialSid);

// Browser back/forward re-reads the URL and updates the store.
window.addEventListener("popstate", () => {
  useUi.getState().setSession(sidFromLocation());
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
