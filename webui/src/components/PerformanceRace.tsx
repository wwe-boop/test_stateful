import { Play, RadioTower, Volume2 } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import type { RaceResult, RunResult } from "../types";
import { formatMs } from "./Timeline";

interface PerformanceRaceProps {
  race: RaceResult | null;
  busy: boolean;
  captureRunning: boolean;
  captureLogs: string[];
  onCollectRace: () => void;
  onRace: () => void;
  onRaceOfficial: () => void;
  onPlayAligned: (results: RunResult[]) => void;
  onPlayOne: (result: RunResult) => void;
}

const ORDER = [
  "official_pytorch_offline",
  "official_pytorch_streaming",
  "bare_engine_streaming",
  "triton_trt_streaming",
];

export function PerformanceRace({
  race,
  busy,
  captureRunning,
  captureLogs,
  onCollectRace,
  onRace,
  onRaceOfficial,
  onPlayAligned,
  onPlayOne,
}: PerformanceRaceProps) {
  const [clockMs, setClockMs] = useState(0);
  const animationRef = useRef<number>(0);
  const rows = useMemo(() => {
    return [...(race?.results ?? [])].sort((a, b) => ORDER.indexOf(a.backend) - ORDER.indexOf(b.backend));
  }, [race]);
  const hasPlayableAudio = rows.some((row) => Boolean(row.audio?.url));
  const maxMs = Math.max(
    1000,
    ...rows.map((row) => scheduledStartMs(row) + audioDurationMs(row)),
    ...rows.map((row) => row.metrics.total_ms ?? 0)
  );

  function playAligned() {
    cancelAnimationFrame(animationRef.current);
    setClockMs(0);
    onPlayAligned(rows.map(withPlaybackSchedule));
    const started = performance.now();
    const tick = () => {
      const elapsed = performance.now() - started;
      setClockMs(Math.min(elapsed, maxMs));
      if (elapsed < maxMs) {
        animationRef.current = requestAnimationFrame(tick);
      }
    };
    animationRef.current = requestAnimationFrame(tick);
  }

  return (
    <section className="panel performance-panel">
      <div className="panel-head">
        <div>
          <h2>Performance PK</h2>
          <p>same text, aligned request start, measured engine audio vs official public/full-wav and paper-reference replay</p>
        </div>
        <div className="concurrency-controls">
          <button className="button primary" onClick={onCollectRace} disabled={busy || captureRunning}>
            <RadioTower size={16} />
            {captureRunning ? "Collecting" : "Collect isolated PK"}
          </button>
          <button className="button" onClick={onRaceOfficial} disabled={busy || captureRunning}>
            Probe live endpoints
          </button>
          <button className="button" onClick={onRace} disabled={busy || captureRunning}>
            Engine+TRT quick
          </button>
          <button className="button" onClick={playAligned} disabled={!hasPlayableAudio}>
            <Play size={16} />
            Play aligned
          </button>
        </div>
      </div>

      <div className="race-player">
        {rows.map((row) => {
          const start = scheduledStartMs(row);
          const duration = audioDurationMs(row);
          const activeMs = Math.max(0, Math.min(duration, clockMs - start));
          const left = (start / maxMs) * 100;
          const width = (duration / maxMs) * 100;
          const fill = duration <= 0 ? 0 : (activeMs / duration) * 100;
          return (
            <div className={`race-row race-${row.backend}`} key={row.backend}>
              <div className="race-row-label">
                <strong>{row.label}</strong>
                <span>{sourceLabel(row.source)} · {audioLabel(row)} · {timingLabel(row)}</span>
                {row.warnings.slice(0, 2).map((warning) => (
                  <em key={warning}>{warning}</em>
                ))}
              </div>
              <div className="race-row-track">
                <div className="race-audio-window" style={{ left: `${left}%`, width: `${Math.max(1, width)}%` }}>
                  <div className="race-audio-fill" style={{ width: `${fill}%` }} />
                </div>
              </div>
              <button
                className="button tiny"
                onClick={() => onPlayOne(withPlaybackSchedule(row))}
                disabled={!row.audio?.url}
                title={row.audio?.url ? "play captured audio" : "no live audio captured for this row"}
              >
                <Volume2 size={14} />
                {row.audio?.url ? "Audio" : "No audio"}
              </button>
            </div>
          );
        })}
      </div>
      <div className="player-note">
        Collect isolated PK runs official PyTorch, bare engine and Triton sequentially on the demo host, then reloads the captured trace/audio. Official rows use TTFT approx = first generated code0 timestamp minus request start; captured full WAV is replayed from that approximate TTFT.
      </div>
      {captureLogs.length > 0 && (
        <div className="capture-log">
          {captureLogs.slice(-8).map((line, index) => (
            <span key={`${line}-${index}`}>{line}</span>
          ))}
        </div>
      )}
    </section>
  );
}

function scheduledStartMs(result: RunResult): number {
  return Number(result.audio?.scheduled_start_ms ?? playableStartMs(result) ?? 0);
}

function audioDurationMs(result: RunResult): number {
  return Number(result.metrics.audio_duration_ms ?? 1600);
}

function chunkTtftMs(result: RunResult): number | undefined {
  if (isOfficialHighLevel(result)) {
    return undefined;
  }
  return result.metrics.server_ttft_ms
    ?? result.metrics.triton_adapter_ttft_ms
    ?? result.metrics.engine_internal_ttft_ms
    ?? result.metrics.client_ttfb_ms;
}

function playableStartMs(result: RunResult): number | undefined {
  if (isOfficialHighLevel(result)) {
    return result.metrics.official_approx_ttft_ms
      ?? result.metrics.first_playable_ms;
  }
  return result.metrics.first_playable_ms
    ?? result.metrics.full_audio_ready_ms
    ?? chunkTtftMs(result);
}

function timingLabel(result: RunResult): string {
  const ready = result.metrics.full_audio_ready_ms ?? result.metrics.total_ms;
  const playable = playableStartMs(result);
  if (isOfficialHighLevel(result)) {
    return `TTFT approx ${formatMs(playable)} · full wav ready ${formatMs(ready)}`;
  }
  return `chunk TTFT ${formatMs(chunkTtftMs(result))} · stream playable ${formatMs(playable)} · ready ${formatMs(ready)}`;
}

function isOfficialHighLevel(result: RunResult): boolean {
  return result.source === "live_official_pytorch" || result.backend.startsWith("official_pytorch");
}

function withPlaybackSchedule(result: RunResult): RunResult {
  if (!result.audio?.url) {
    return result;
  }
  const scheduled = scheduledStartMs(result);
  if (result.audio.scheduled_start_ms === scheduled) {
    return result;
  }
  return {
    ...result,
    audio: {
      ...result.audio,
      scheduled_start_ms: scheduled,
    },
  };
}

function sourceLabel(source: string): string {
  if (source === "fixture") return "fixture timing";
  if (source === "live_triton") return "live Triton";
  if (source === "live_engine" || source === "live_engine_websocket") return "live bare engine";
  if (source === "live_official_pytorch") return "live official";
  return source;
}

function audioLabel(result: RunResult): string {
  if (!result.audio?.url) {
    return "timing only";
  }
  if (isOfficialHighLevel(result)) {
    return "captured full-wav replayed at TTFT approx";
  }
  if (result.audio.source) {
    return `${result.audio.source} audio`;
  }
  return "captured audio";
}
