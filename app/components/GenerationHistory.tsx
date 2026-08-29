"use client";

import type { HistoryItem } from "@/lib/types";

export default function GenerationHistory({
  items,
  onPick,
  onClear,
}: {
  items: HistoryItem[];
  onPick: (item: HistoryItem) => void;
  onClear: () => void;
}) {
  return (
    <div className="rounded-3xl border border-line bg-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-medium text-zinc-200">History</h2>
        {items.length ? (
          <button type="button" onClick={onClear} className="text-xs text-zinc-500">
            Clear
          </button>
        ) : null}
      </div>
      {items.length === 0 ? (
        <p className="text-xs text-zinc-500">Stored only in this browser.</p>
      ) : (
        <ul className="space-y-2">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onPick(item)}
                className="w-full rounded-2xl border border-line bg-ink px-3 py-2 text-left"
              >
                <p className="line-clamp-2 text-sm text-zinc-200">{item.prompt}</p>
                <p className="mt-1 text-[11px] uppercase tracking-wide text-zinc-500">
                  {item.status} · {new Date(item.createdAt).toLocaleString()}
                </p>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
