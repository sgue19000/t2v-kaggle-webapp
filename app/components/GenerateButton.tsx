"use client";

import type { JobStatus } from "@/lib/types";

export default function GenerateButton({
  disabled,
  busy,
  status,
  progress,
  message,
  onClick,
}: {
  disabled: boolean;
  busy: boolean;
  status: JobStatus | "idle";
  progress: number;
  message: string;
  onClick: () => void;
}) {
  return (
    <div className="space-y-3">
      <button
        type="button"
        disabled={disabled}
        onClick={onClick}
        className="w-full rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-white shadow-glow transition enabled:hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {busy ? "Generating video…" : "Generate Video"}
      </button>
      {busy || status !== "idle" ? (
        <div>
          <div className="mb-1 flex justify-between text-[11px] uppercase tracking-wide text-zinc-500">
            <span>{status}</span>
            <span>{Math.min(100, Math.max(0, Math.round(progress)))}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-line">
            <div
              className="h-full rounded-full bg-mint transition-all"
              style={{ width: `${Math.min(100, Math.max(6, progress))}%` }}
            />
          </div>
          {message ? <p className="mt-2 text-xs text-zinc-400">{message}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
