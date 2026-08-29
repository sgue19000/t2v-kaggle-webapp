import type { HistoryItem } from "./types";

const KEY = "t2v.history.v1";
const LIMIT = 20;

export function loadHistory(): HistoryItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryItem[];
    return Array.isArray(parsed) ? parsed.slice(0, LIMIT) : [];
  } catch {
    return [];
  }
}

export function saveHistory(items: HistoryItem[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(KEY, JSON.stringify(items.slice(0, LIMIT)));
}

export function upsertHistory(item: HistoryItem): HistoryItem[] {
  const current = loadHistory().filter((x) => x.id !== item.id);
  const next = [item, ...current].slice(0, LIMIT);
  saveHistory(next);
  return next;
}

export function clearHistory(): HistoryItem[] {
  saveHistory([]);
  return [];
}
