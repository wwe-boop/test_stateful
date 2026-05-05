import { Pause, Play, RadioTower, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { TraceEvent } from "../types";
import { buildDecodeTrace } from "../trace";
import { formatMs } from "./Timeline";

interface TextPlayerProps {
  events: TraceEvent[];
  text: string;
  audioUrl?: string;
  liveMs: number;
  live: boolean;
  source: string;
}

export function TextPlayer({ events, text, audioUrl, liveMs, live, source }: TextPlayerProps) {
  const model = useMemo(() => buildDecodeTrace(events, text), [events, text]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [cursorMs, setCursorMs] = useState(0);
  const [manualPlayback, setManualPlayback] = useState(false);
  const hasSeekableAudio = Boolean(audioUrl);
  const displayMs = hasSeekableAudio && (!live || manualPlayback) ? cursorMs : liveMs;
  const activeStep = stepAt(model.steps, displayMs);
  const maxMs = Math.max(model.audioDurationMs, 1);

  useEffect(() => {
    setPlaying(false);
    setCursorMs(0);
    setManualPlayback(false);
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, [audioUrl, events]);

  function togglePlayback() {
    const audio = audioRef.current;
    if (!audio || !hasSeekableAudio) {
      return;
    }
    if (audio.paused) {
      if (audio.ended || (Number.isFinite(audio.duration) && audio.currentTime >= audio.duration - 0.02)) {
        audio.currentTime = 0;
        setCursorMs(0);
      }
      setManualPlayback(true);
      void audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      audio.pause();
      setPlaying(false);
    }
  }

  function seek(value: number) {
    const audio = audioRef.current;
    const nextMs = Math.max(0, Math.min(value, maxMs));
    setManualPlayback(true);
    setCursorMs(nextMs);
    if (audio && hasSeekableAudio) {
      audio.currentTime = nextMs / 1000;
    }
  }

  function seekStep(startMs: number) {
    if (!hasSeekableAudio) {
      return;
    }
    seek(startMs);
  }

  return (
    <section className="panel text-player-panel">
      <div className="panel-head">
        <div>
          <h2>Text Player</h2>
          <p>one decode step is one audio chunk; token steps and PAD flush steps are shown as-is</p>
        </div>
        <div className="player-summary">
          <span><RadioTower size={14} /> {source}</span>
          <span>{model.textTokenCount} token steps</span>
          <span>{model.padStepCount} PAD flush steps</span>
          <span>{formatMs(model.stepMs)} / step</span>
        </div>
      </div>

      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={(event) => setCursorMs(event.currentTarget.currentTime * 1000)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={(event) => {
          event.currentTarget.currentTime = 0;
          setCursorMs(0);
          setPlaying(false);
        }}
      />

      <div className="text-player-transport">
        <button className="button primary" onClick={togglePlayback} disabled={!hasSeekableAudio}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={Math.round(maxMs)}
          value={Math.max(0, Math.min(Math.round(displayMs), Math.round(maxMs)))}
          onChange={(event) => seek(Number(event.target.value))}
          disabled={!hasSeekableAudio}
          aria-label="text player seek"
        />
        <span>{formatMs(displayMs)} / {formatMs(maxMs)}</span>
        <button className="button tiny" onClick={() => seek(0)} disabled={!hasSeekableAudio}>
          <RotateCcw size={14} />
          0
        </button>
      </div>

      {!hasSeekableAudio && (
        <div className="player-note">
          {live
            ? "Live stream is still being captured. Seekable playback appears when a WAV is available."
            : "No captured audio is attached to this trace. Timing and decode-step structure remain visible."}
        </div>
      )}

      <div className="text-token-transcript" aria-label="clickable token transcript">
        {model.steps.map((step) => {
          const played = step.endMs <= displayMs;
          const active = step.index === activeStep?.index;
          return (
            <button
              type="button"
              className={[
                "text-token",
                `text-token-${step.phase}`,
                played ? "text-token-played" : "",
                active ? "text-token-active" : ""
              ].join(" ")}
              key={`text-${step.key}`}
              onClick={() => seekStep(step.startMs)}
              disabled={!hasSeekableAudio}
              title={`#${step.index + 1} ${step.phase} ${formatMs(step.startMs)}-${formatMs(step.endMs)}`}
            >
              {step.label}
            </button>
          );
        })}
      </div>

      <div className="player-note">
        The PAD cells are not hidden latency. They are the engine flushing audio after text tokens have been consumed. Trace source: {model.source}.
      </div>
    </section>
  );
}

function stepAt(steps: ReturnType<typeof buildDecodeTrace>["steps"], ms: number) {
  return steps.find((step) => ms >= step.startMs && ms < step.endMs) ?? steps[steps.length - 1];
}
