"use client";

import type { ReactNode } from "react";
import { DEFAULT_SETTINGS, type VideoSettings } from "@/lib/types";

export default function VideoSettingsPanel({
  value,
  onChange,
  disabled,
}: {
  value: VideoSettings;
  onChange: (next: VideoSettings) => void;
  disabled?: boolean;
}) {
  function patch(partial: Partial<VideoSettings>) {
    onChange({ ...value, ...partial });
  }

  return (
    <div className="rounded-3xl border border-line bg-panel p-4">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-medium text-zinc-200">Settings for P100</h2>
        <button type="button" disabled={disabled} onClick={() => onChange(DEFAULT_SETTINGS)} className="text-xs text-mint disabled:opacity-50">
          Reset
        </button>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Frames">
          <select disabled={disabled} value={value.numFrames} onChange={(e) => patch({ numFrames: Number(e.target.value) })} className="w-full rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none">
            <option value={8}>8 (~1s)</option>
            <option value={16}>16 (~2s)</option>
          </select>
        </Field>
        <Field label="Resolution">
          <select disabled={disabled} value={`${value.width}x${value.height}`} onChange={(e) => {
            const [w, h] = e.target.value.split("x").map(Number);
            patch({ width: w, height: h });
          }} className="w-full rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none">
            <option value="256x256">256x256</option>
          </select>
        </Field>
        <Field label="FPS">
          <select disabled={disabled} value={value.fps} onChange={(e) => patch({ fps: Number(e.target.value) })} className="w-full rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none">
            <option value={6}>6</option>
            <option value={8}>8</option>
            <option value={12}>12</option>
          </select>
        </Field>
        <Field label="Steps">
          <select disabled={disabled} value={value.numInferenceSteps} onChange={(e) => patch({ numInferenceSteps: Number(e.target.value) })} className="w-full rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none">
            <option value={15}>15 faster</option>
            <option value={20}>20 default</option>
            <option value={25}>25 sharper</option>
          </select>
        </Field>
        <Field label="Guidance">
          <input type="number" min={1} max={15} step={0.5} disabled={disabled} value={value.guidanceScale} onChange={(e) => patch({ guidanceScale: Number(e.target.value) })} className="w-full rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none" />
        </Field>
        <Field label="Seed">
          <input type="number" disabled={disabled} value={value.seed} onChange={(e) => patch({ seed: Number(e.target.value) })} className="w-full rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none" />
        </Field>
      </div>
      <label className="mt-3 block text-xs text-zinc-400">Negative prompt</label>
      <textarea disabled={disabled} rows={3} value={value.negativePrompt} onChange={(e) => patch({ negativePrompt: e.target.value })} className="mt-1 min-h-[4.5rem] w-full resize-y rounded-[0.9rem] border border-line bg-ink px-3 py-2 text-zinc-100 outline-none" />
      <p className="mt-3 text-[11px] leading-relaxed text-zinc-500">
        Stay on 16 frames / 256px / 20 steps on a free Kaggle P100. Higher settings are rejected to avoid OOM.
      </p>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-xs text-zinc-400">
      <span className="mb-1 block">{label}</span>
      {children}
    </label>
  );
}
