"use client";

import { MAX_PROMPT_CHARS } from "@/lib/types";

export default function PromptBox({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (next: string) => void;
  disabled?: boolean;
}) {
  const count = value.length;
  const over = count > MAX_PROMPT_CHARS;
  return (
    <div className="rounded-3xl border border-line bg-panel p-4 shadow-glow">
      <label className="mb-2 block text-sm font-medium text-zinc-200">Prompt</label>
      <textarea
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        rows={6}
        placeholder="Describe motion, lighting, camera, and subject…"
        className="w-full resize-y rounded-2xl border border-line bg-ink px-4 py-3 text-zinc-100 outline-none ring-accent/40 placeholder:text-zinc-600 focus:ring-2 disabled:opacity-60"
      />
      <div className={`mt-2 text-right text-xs ${over ? "text-rose-300" : "text-zinc-500"}`}>
        {count}/{MAX_PROMPT_CHARS}
      </div>
    </div>
  );
}
