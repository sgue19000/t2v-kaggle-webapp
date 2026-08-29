export function friendlyClientError(raw: string): string {
  const text = raw.toLowerCase();
  if (!raw.trim()) return "Something went wrong. Try again.";
  if (text.includes("failed to fetch") || text.includes("networkerror") || text.includes("load failed")) {
    return "Colab backend offline or the tunnel disconnected. Re-run the tunnel cell and paste the new HTTPS URL.";
  }
  if (text.includes("timed out") || text.includes("timeout") || text.includes("aborted")) {
    return "Timed out waiting for Colab. Keep the notebook running and check the tunnel URL.";
  }
  if (text.includes("already generating") || text.includes("already running") || text.includes("409")) {
    return "The GPU is busy with another video. Wait until it finishes.";
  }
  if (text.includes("out of memory") || text.includes("oom")) {
    return "The GPU ran out of memory. Use 8 frames, 256x256, 10-15 steps, then restart the Colab runtime.";
  }
  if (text.includes("gpu unavailable") || text.includes("cuda")) {
    return "GPU unavailable. In Colab choose Runtime → Change runtime type → GPU and rerun.";
  }
  if (text.includes("prompt") || text.includes("frame") || text.includes("resolution") || text.includes("256")) {
    return raw;
  }
  if (text.includes("empty")) return "Set the Colab HTTPS tunnel URL in the API field first.";
  return raw.length > 240 ? "Generation failed. Keep the Colab notebook running and try again." : raw;
}
