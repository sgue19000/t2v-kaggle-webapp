"use client";

import { useEffect, useMemo, useState } from "react";
import ExamplePrompts from "./components/ExamplePrompts";
import GenerateButton from "./components/GenerateButton";
import GenerationHistory from "./components/GenerationHistory";
import PromptBox from "./components/PromptBox";
import VideoPlayer from "./components/VideoPlayer";
import VideoSettingsPanel from "./components/VideoSettings";
import {
  getApiBase,
  getJobStatus,
  getStoredApiUrl,
  healthCheck,
  resolveVideoUrl,
  setStoredApiUrl,
  startGeneration,
} from "@/lib/api";
import { friendlyClientError } from "@/lib/errors";
import { clearHistory, loadHistory, upsertHistory } from "@/lib/history";
import {
  DEFAULT_SETTINGS,
  MAX_PROMPT_CHARS,
  POLL_TIMEOUT_MS,
  type HistoryItem,
  type JobStatus,
  type VideoSettings,
} from "@/lib/types";

export default function HomePage() {
  const [prompt, setPrompt] = useState(
    "A cinematic futuristic city at night, flying cars, rain, realistic camera movement",
  );
  const [settings, setSettings] = useState<VideoSettings>(DEFAULT_SETTINGS);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<JobStatus | "idle">("idle");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [apiInput, setApiInput] = useState("");

  const apiBase = useMemo(() => getApiBase(), [apiInput]);

  useEffect(() => {
    setHistory(loadHistory());
    setApiInput(getStoredApiUrl() || getApiBase());
    let cancelled = false;
    healthCheck().then((ok) => {
      if (!cancelled) setApiOnline(ok);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  async function saveApiUrl() {
    setStoredApiUrl(apiInput);
    setApiOnline(null);
    const ok = await healthCheck();
    setApiOnline(ok);
    if (!ok) setError(friendlyClientError("failed to fetch"));
    else setError("");
  }

  async function generate() {
    const trimmed = prompt.trim();
    if (!trimmed) {
      setError("Enter a prompt first.");
      return;
    }
    if (trimmed.length > MAX_PROMPT_CHARS) {
      setError(`Prompt must be ${MAX_PROMPT_CHARS} characters or fewer.`);
      return;
    }
    setBusy(true);
    setError("");
    setVideoUrl("");
    setStatus("queued");
    setProgress(5);
    setMessage("Sending job to Colab…");

    try {
      const started = await startGeneration(trimmed, settings);
      const item: HistoryItem = {
        id: started.job_id,
        prompt: trimmed,
        createdAt: Date.now(),
        status: started.status ?? "queued",
        settings,
      };
      setHistory(upsertHistory(item));

      let done = false;
      const startedAt = Date.now();
      while (!done) {
        if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
          throw new Error("Timed out waiting for Colab to finish the video.");
        }
        await new Promise((r) => setTimeout(r, 2500));
        const snap = await getJobStatus(started.job_id);
        setStatus(snap.status);
        setProgress(snap.progress ?? 10);
        setMessage(snap.message ?? snap.status);
        if (snap.status === "completed") {
          const url = snap.video_url ? resolveVideoUrl(snap.video_url) : "";
          setVideoUrl(url);
          setHistory(upsertHistory({ ...item, status: "completed", videoUrl: url }));
          done = true;
        } else if (snap.status === "failed") {
          const fail = snap.error || "Generation failed on the Colab GPU.";
          setError(friendlyClientError(fail));
          setHistory(upsertHistory({ ...item, status: "failed", error: fail }));
          done = true;
        }
      }
    } catch (err) {
      const text = err instanceof Error ? err.message : "Unknown error";
      setError(friendlyClientError(text));
      setStatus("failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="noise min-h-screen">
      <div className="mx-auto max-w-5xl px-4 pb-16 pt-8 sm:px-6">
        <header className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-mint">Open source · Colab GPU</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Lumen Clip</h1>
            <p className="mt-2 max-w-xl text-sm text-zinc-400">
              Type a scene. A free Google Colab notebook runs ModelScope Text-to-Video 1.7B and returns an MP4.
            </p>
          </div>
          <div className="rounded-2xl border border-line bg-panel px-4 py-3 text-xs text-zinc-400">
            <div className="flex items-center gap-2">
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  apiOnline === null ? "bg-zinc-500" : apiOnline ? "bg-mint" : "bg-rose-400"
                }`}
              />
              {apiOnline === null
                ? "Checking API"
                : apiOnline
                  ? "Colab API reachable"
                  : "Colab API offline"}
            </div>
            <p className="mt-1 max-w-[16rem] break-all text-[11px] text-zinc-500">
              {apiBase || "Paste the trycloudflare URL"}
            </p>
            <input
              value={apiInput}
              onChange={(e) => setApiInput(e.target.value)}
              placeholder="https://xxxx.trycloudflare.com"
              className="mt-2 w-full rounded-xl border border-line bg-ink px-2 py-1.5 text-[11px] text-zinc-200"
            />
            <button type="button" onClick={saveApiUrl} className="mt-2 w-full rounded-xl border border-line px-2 py-1 text-[11px] text-mint">
              Save API URL
            </button>
          </div>
        </header>

        <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
          <section className="space-y-5">
            <PromptBox value={prompt} onChange={setPrompt} disabled={busy} />
            <ExamplePrompts disabled={busy} onPick={(text) => { setPrompt(text); setError(""); }} />
            <GenerateButton disabled={busy || !prompt.trim()} busy={busy} status={status} progress={progress} message={message} onClick={generate} />
            {error ? (
              <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{error}</div>
            ) : null}
            <VideoPlayer src={videoUrl} />
          </section>
          <aside className="space-y-5">
            <VideoSettingsPanel value={settings} onChange={setSettings} disabled={busy} />
            <GenerationHistory
              items={history}
              onPick={(item) => {
                setPrompt(item.prompt);
                setSettings(item.settings);
                if (item.videoUrl) setVideoUrl(item.videoUrl);
                setError(item.error ?? "");
              }}
              onClear={() => setHistory(clearHistory())}
            />
          </aside>
        </div>
      </div>
    </main>
  );
}
