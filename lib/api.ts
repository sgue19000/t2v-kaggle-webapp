import { friendlyClientError } from "./errors";
import type {
  GenerateRequest,
  GenerateResponse,
  StatusResponse,
  VideoSettings,
} from "./types";

const API_STORE = "t2v.apiUrl";

export function getStoredApiUrl(): string {
  if (typeof window === "undefined") return "";
  return (localStorage.getItem(API_STORE) || "").replace(/\/$/, "");
}

export function setStoredApiUrl(url: string): void {
  if (typeof window === "undefined") return;
  const clean = url.trim().replace(/\/$/, "");
  if (clean) localStorage.setItem(API_STORE, clean);
  else localStorage.removeItem(API_STORE);
}

export function getApiBase(): string {
  const stored = getStoredApiUrl();
  if (stored) return stored;
  return (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
}

export function buildGeneratePayload(
  prompt: string,
  settings: VideoSettings,
): GenerateRequest {
  return {
    prompt: prompt.trim(),
    negative_prompt: settings.negativePrompt.trim() || undefined,
    num_frames: settings.numFrames,
    height: settings.height,
    width: settings.width,
    fps: settings.fps,
    guidance_scale: settings.guidanceScale,
    seed: settings.seed,
    steps: settings.numInferenceSteps,
  };
}

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(
      friendlyClientError(
        res.ok ? "Backend returned a non-JSON response." : `Backend error ${res.status}`,
      ),
    );
  }
}

function detailMessage(data: { detail?: unknown; error?: string }, fallback: string): string {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    const first = data.detail[0] as { msg?: string } | undefined;
    if (first?.msg) return first.msg;
  }
  return data.error || fallback;
}

export async function healthCheck(timeoutMs = 8000): Promise<boolean> {
  const base = getApiBase();
  if (!base) return false;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/health`, { signal: controller.signal, cache: "no-store" });
    if (!res.ok) return false;
    const data = await parseJson<{ ok?: boolean; status?: string }>(res);
    return Boolean(data.ok || data.status === "ok");
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export async function startGeneration(
  prompt: string,
  settings: VideoSettings,
): Promise<GenerateResponse> {
  const base = getApiBase();
  if (!base) {
    throw new Error("Set the Colab HTTPS tunnel URL in the API field first.");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const res = await fetch(`${base}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildGeneratePayload(prompt, settings)),
      signal: controller.signal,
    });
    const data = await parseJson<GenerateResponse & { detail?: unknown; error?: string }>(res);
    if (!res.ok) {
      throw new Error(friendlyClientError(detailMessage(data, `Request failed (${res.status})`)));
    }
    if (!data.job_id) throw new Error("Backend did not return a job id.");
    return data;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(friendlyClientError("timeout"));
    }
    if (err instanceof TypeError) {
      throw new Error(friendlyClientError("failed to fetch"));
    }
    throw err instanceof Error ? new Error(friendlyClientError(err.message)) : err;
  } finally {
    clearTimeout(timer);
  }
}

export async function getJobStatus(jobId: string): Promise<StatusResponse> {
  const base = getApiBase();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const res = await fetch(`${base}/status/${encodeURIComponent(jobId)}`, {
      signal: controller.signal,
      cache: "no-store",
    });
    const data = await parseJson<StatusResponse & { detail?: unknown; error?: string }>(res);
    if (!res.ok) {
      throw new Error(friendlyClientError(detailMessage(data, `Status check failed (${res.status})`)));
    }
    return data;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(friendlyClientError("timeout"));
    }
    if (err instanceof TypeError) {
      throw new Error(friendlyClientError("failed to fetch"));
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export function resolveVideoUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
    return pathOrUrl;
  }
  const base = getApiBase();
  if (!base) return pathOrUrl;
  return `${base}${pathOrUrl.startsWith("/") ? "" : "/"}${pathOrUrl}`;
}
