export type JobStatus =
  | "queued"
  | "loading"
  | "generating"
  | "completed"
  | "failed";

export interface VideoSettings {
  numFrames: number;
  height: number;
  width: number;
  fps: number;
  guidanceScale: number;
  seed: number;
  negativePrompt: string;
  numInferenceSteps: number;
}

export interface GenerateRequest {
  prompt: string;
  negative_prompt?: string;
  num_frames: number;
  height: number;
  width: number;
  fps: number;
  guidance_scale: number;
  seed: number;
  num_inference_steps: number;
}

export interface GenerateResponse {
  job_id: string;
  status: JobStatus;
}

export interface StatusResponse {
  job_id: string;
  status: JobStatus;
  progress?: number;
  message?: string;
  error?: string;
  video_url?: string;
}

export interface HistoryItem {
  id: string;
  prompt: string;
  createdAt: number;
  status: JobStatus;
  videoUrl?: string;
  settings: VideoSettings;
  error?: string;
}

export const DEFAULT_SETTINGS: VideoSettings = {
  numFrames: 16,
  height: 256,
  width: 256,
  fps: 8,
  guidanceScale: 9,
  seed: 42,
  negativePrompt: "blurry, low quality, watermark, text, logo",
  numInferenceSteps: 25,
};

export const MAX_PROMPT_CHARS = 500;
