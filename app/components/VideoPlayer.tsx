"use client";

export default function VideoPlayer({ src }: { src: string }) {
  if (!src) {
    return (
      <div className="flex aspect-square items-center justify-center rounded-3xl border border-dashed border-line bg-panel text-sm text-zinc-500">
        Preview appears here after generation
      </div>
    );
  }

  async function download(e: React.MouseEvent<HTMLAnchorElement>) {
    e.preventDefault();
    try {
      const res = await fetch(src, { cache: "no-store" });
      if (!res.ok) throw new Error("download failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "lumen-clip.mp4";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      window.open(src, "_blank");
    }
  }

  return (
    <div className="overflow-hidden rounded-3xl border border-line bg-black">
      <video src={src} controls playsInline className="aspect-square w-full bg-black object-contain" />
      <div className="flex justify-end border-t border-line bg-panel p-3">
        <a
          href={src}
          download="lumen-clip.mp4"
          onClick={download}
          className="rounded-xl bg-zinc-100 px-4 py-2 text-sm font-medium text-zinc-900"
        >
          Download MP4
        </a>
      </div>
    </div>
  );
}
