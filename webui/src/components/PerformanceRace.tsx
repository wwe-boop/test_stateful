import { Play, RadioTower, Volume2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LlmPkRequest, LlmPkResult, RunResult, TraceEvent } from "../types";
import { formatMs } from "./Timeline";

interface LlmPkPanelProps {
  result: LlmPkResult | null;
  busy: boolean;
  defaults: LlmPkRequest;
  onRun: (request: LlmPkRequest) => void;
  onPlayAligned: (results: RunResult[]) => void;
  onPlayOne: (result: RunResult) => void;
}

const ROW_ORDER: Array<RunResult["backend"]> = ["triton_streaming", "triton_offline"];

const PRESETS: Array<{ label: string; ms: number; hint: string }> = [
  { label: "Local 7B", ms: 7, hint: "≈140 tok/s" },
  { label: "Claude Sonnet", ms: 12, hint: "≈80 tok/s" },
  { label: "GPT-4o", ms: 30, hint: "≈33 tok/s" },
  { label: "Reasoning", ms: 50, hint: "≈20 tok/s" },
];

export function PerformanceRace({
  result,
  busy,
  defaults,
  onRun,
  onPlayAligned,
  onPlayOne,
}: LlmPkPanelProps) {
  const [text, setText] = useState(defaults.text);
  const [speaker, setSpeaker] = useState(defaults.speaker);
  const [language, setLanguage] = useState(defaults.language);
  const [msPerToken, setMsPerToken] = useState(defaults.ms_per_token);
  const [clockMs, setClockMs] = useState(0);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    setText(defaults.text);
    setSpeaker(defaults.speaker);
    setLanguage(defaults.language);
    setMsPerToken(defaults.ms_per_token);
  }, [defaults.text, defaults.speaker, defaults.language, defaults.ms_per_token]);

  const rows = useMemo(() => {
    if (!result) return [] as RunResult[];
    return [...result.results].sort(
      (a, b) => ROW_ORDER.indexOf(a.backend) - ROW_ORDER.indexOf(b.backend),
    );
  }, [result]);

  const maxMs = useMemo(() => {
    const upper = Math.max(
      1000,
      ...rows.map((row) => scheduledStartMs(row) + audioDurationMs(row)),
      ...rows.map((row) => row.metrics.total_ms ?? 0),
      ...rows.map((row) => row.metrics.simulated_llm_complete_ms ?? 0),
    );
    return upper * 1.05;
  }, [rows]);

  const hasPlayableAudio = rows.some((row) => Boolean(row.audio?.url));

  const summary = useMemo(() => buildSummary(rows), [rows]);

  function handleRun() {
    onRun({
      text: text.trim() || defaults.text,
      speaker: speaker.trim() || defaults.speaker,
      language: language.trim() || defaults.language,
      ms_per_token: msPerToken,
    });
  }

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
          <h2>LLM PK · Streaming vs Offline TTS</h2>
          <p>
            Simulate an upstream LLM emitting tokens at your chosen rate, and
            watch streaming TTS start speaking before all the tokens arrive —
            while offline TTS waits for the last one.
          </p>
        </div>
      </div>

      <div className="llm-pk-controls">
        <textarea
          className="llm-pk-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={2}
          placeholder="enter text the simulated LLM will emit token-by-token"
        />
        <div className="llm-pk-row">
          <input
            value={speaker}
            onChange={(event) => setSpeaker(event.target.value)}
            aria-label="speaker"
            placeholder="speaker"
          />
          <input
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            aria-label="language"
            placeholder="language"
          />
          <div className="llm-pk-slider">
            <label>
              ms / token
              <input
                type="range"
                min={5}
                max={100}
                step={1}
                value={msPerToken}
                onChange={(event) => setMsPerToken(Number(event.target.value))}
              />
              <strong>{msPerToken}ms</strong>
            </label>
          </div>
          <button className="button primary" onClick={handleRun} disabled={busy}>
            <RadioTower size={16} />
            {busy ? "Running" : "Run PK"}
          </button>
          <button className="button" onClick={playAligned} disabled={!hasPlayableAudio}>
            <Play size={16} />
            Play aligned
          </button>
        </div>
        <div className="llm-pk-presets">
          <span>Presets:</span>
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              className={`button tiny ${msPerToken === preset.ms ? "active" : ""}`}
              onClick={() => setMsPerToken(preset.ms)}
              type="button"
            >
              {preset.label} {preset.ms}ms · {preset.hint}
            </button>
          ))}
        </div>
      </div>

      {summary && (
        <div className="llm-pk-summary">
          {summary}
        </div>
      )}

      <div className="race-player">
        {rows.map((row) => {
          const start = scheduledStartMs(row);
          const duration = audioDurationMs(row);
          const activeMs = Math.max(0, Math.min(duration, clockMs - start));
          const left = (start / maxMs) * 100;
          const width = (duration / maxMs) * 100;
          const fill = duration <= 0 ? 0 : (activeMs / duration) * 100;
          const llmCompleteLeft = row.metrics.simulated_llm_complete_ms !== undefined
            ? (row.metrics.simulated_llm_complete_ms / maxMs) * 100
            : null;
          const tokenTicks = row.events.filter((event) => event.type === "llm_token");
          return (
            <div className={`race-row race-${row.backend}`} key={row.backend}>
              <div className="race-row-label">
                <strong>{row.label}</strong>
                <span>{row.mode}</span>
                <span>{timingLabel(row)}</span>
                {row.warnings.slice(0, 2).map((warning) => (
                  <em key={warning}>{warning}</em>
                ))}
              </div>
              <div className="race-row-track">
                {tokenTicks.map((event, index) => (
                  <span
                    key={`${event.type}-${index}-${event.t_ms}`}
                    className="race-token-tick"
                    style={{ left: `${(event.t_ms / maxMs) * 100}%` }}
                    title={`${event.text} @ ${formatMs(event.t_ms)}`}
                  />
                ))}
                {llmCompleteLeft !== null && (
                  <span
                    className="race-llm-complete-marker"
                    style={{ left: `${llmCompleteLeft}%` }}
                    title={`Simulated LLM finished @ ${formatMs(row.metrics.simulated_llm_complete_ms)}`}
                  />
                )}
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
        Token ticks show when each simulated LLM token "arrives". The dashed
        marker is when the upstream finishes. The streaming row's audio
        window starts well before the marker; the offline row's only starts
        after it.
      </div>

      {rows.length === 0 && !busy && (
        <div className="player-note">
          Click <strong>Run PK</strong> to capture both runs against the bare
          engine. Cache is not pre-warmed; both runs start clean.
        </div>
      )}
    </section>
  );
}

function buildSummary(rows: RunResult[]): string | null {
  const streaming = rows.find((row) => row.backend === "triton_streaming");
  const offline = rows.find((row) => row.backend === "triton_offline");
  if (!streaming || !offline) return null;
  const streamingStart = streaming.metrics.first_playable_ms ?? streaming.metrics.client_ttfb_ms;
  const offlineStart = offline.metrics.first_playable_ms ?? offline.metrics.client_ttfb_ms;
  if (streamingStart === undefined || offlineStart === undefined) return null;
  const gap = offlineStart - streamingStart;
  if (gap <= 0) return null;
  return `Streaming starts speaking ${formatMs(gap)} before offline can — audio first byte ${formatMs(streamingStart)} vs ${formatMs(offlineStart)}.`;
}

function scheduledStartMs(result: RunResult): number {
  return Number(result.audio?.scheduled_start_ms ?? playableStartMs(result) ?? 0);
}

function audioDurationMs(result: RunResult): number {
  return Number(result.metrics.audio_duration_ms ?? 1600);
}

function playableStartMs(result: RunResult): number | undefined {
  return result.metrics.first_playable_ms ?? result.metrics.client_ttfb_ms;
}

function timingLabel(result: RunResult): string {
  const ttfb = result.metrics.client_ttfb_ms ?? result.metrics.first_playable_ms;
  const llmComplete = result.metrics.simulated_llm_complete_ms;
  const total = result.metrics.total_ms;
  return `first audio ${formatMs(ttfb)} · LLM done ${formatMs(llmComplete)} · total ${formatMs(total)}`;
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

export type { TraceEvent };
