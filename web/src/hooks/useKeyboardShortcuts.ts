import { useEffect } from "react";

export type KeyHandler = (event: KeyboardEvent) => void;
export type KeyMap = Record<string, KeyHandler>;

/**
 * Register a global keydown listener for the given map.
 *
 * Keys are matched case-insensitively. Prefix `ctrl+` for Ctrl/Cmd combos.
 * Ignores events when the focus is inside a text input or textarea so
 * query entry doesn't trigger rating shortcuts.
 */
export function useKeyboardShortcuts(map: KeyMap, enabled = true): void {
  useEffect(() => {
    if (!enabled) return;

    function handler(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable) {
          return;
        }
      }
      const key = e.key.toLowerCase();
      const combo =
        (e.ctrlKey || e.metaKey ? "ctrl+" : "") + key;
      const fn = map[combo] || map[key];
      if (fn) {
        e.preventDefault();
        // When focus is on a button, the browser synthesises a click
        // on Enter (and Space). preventDefault on keydown does NOT
        // reliably suppress that synthetic click in every browser, so
        // the focused button's onClick can fire on top of our shortcut
        // and re-trigger the previous action. Blur the active button
        // so Enter only runs our handler.
        const active = document.activeElement as HTMLElement | null;
        if (active && active.tagName === "BUTTON") {
          active.blur();
        }
        fn(e);
      }
    }

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [map, enabled]);
}
