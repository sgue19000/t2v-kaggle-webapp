"use client";

const EXAMPLES = [
  "A golden retriever running through tall grass at sunrise, cinematic tracking shot",
  "Slow orbit around a mossy stone temple in fog, lanterns glowing",
  "Macro shot of coffee pouring into a glass, steam rising, studio light",
  "Night market in the rain, neon reflections, handheld camera",
];

export default function ExamplePrompts({
  onPick,
  disabled,
}: {
  onPick: (prompt: string) => void;
  disabled?: boolean;
}) {
  return (
    <div>
      <p className="mb-2 text-xs uppercase tracking-wide text-zinc-500">Example prompts</p>
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((prompt) => (
          <button
            key={prompt}
            type="button"
            disabled={disabled}
            onClick={() => onPick(prompt)}
            className="rounded-full border border-line bg-panel px-3 py-1.5 text-left text-xs text-zinc-300 disabled:opacity-50"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
