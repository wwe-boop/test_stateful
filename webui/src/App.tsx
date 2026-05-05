import { Activity, AlertTriangle, Download, RadioTower, RotateCcw, Zap } from "lucide-react";
import type { MutableRefObject } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { getCapabilities, mediaUrl, runRace, startRaceCapture, wsUrl } from "./api";
import { PcmStreamPlayer } from "./audio/pcm-player";
import { wavBlobFromPcmF32 } from "./audio/wav";
import { ConcurrencyPanel } from "./components/ConcurrencyPanel";
import { PerformanceRace } from "./components/PerformanceRace";
import { TextPlayer } from "./components/TextPlayer";
import { formatMs } from "./components/Timeline";
import type { AudioFormat, Capabilities, DemoRequest, RaceCaptureEvent, RaceResult, TraceEvent } from "./types";
import "./styles.css";

const DEFAULT_AUDIO: AudioFormat = { encoding: "pcm_f32", sample_rate: 24000, channels: 1 };

export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [request, setRequest] = useState<DemoRequest>({
    text: "你好，这是千问3 TTS token级流式语音演示。",
    speaker: "Serena",
    language: "auto",
    cache_mode: "hit"
  });
  const [race, setRace] = useState<RaceResult | null>(null);
  const [liveEvents, setLiveEvents] = useState<TraceEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [captureRunning, setCaptureRunning] = useState(false);
  const [captureLogs, setCaptureLogs] = useState<string[]>([]);
  const [liveBusy, setLiveBusy] = useState(false);
  const [liveClockRunning, setLiveClockRunning] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [liveClockMs, setLiveClockMs] = useState(0);
  const [liveAudioUrl, setLiveAudioUrl] = useState<string | undefined>(undefined);
  const playerRef = useRef<PcmStreamPlayer | null>(null);
  const raceAudioRef = useRef<AudioContext | null>(null);
  const liveSocketRef = useRef<WebSocket | null>(null);
  const raceCaptureSocketRef = useRef<WebSocket | null>(null);
  const liveRunSeqRef = useRef(0);
  const liveAudioPartsRef = useRef<ArrayBuffer[]>([]);
  const liveAudioFormatRef = useRef<AudioFormat>(DEFAULT_AUDIO);
  const liveStartedAtRef = useRef<number | null>(null);
  const liveFirstAudioMsRef = useRef<number | null>(null);
  const liveAudioDurationMsRef = useRef(0);
  const livePlaybackEndMsRef = useRef<number | null>(null);

  useEffect(() => {
    getCapabilities()
      .then((caps) => {
        setCapabilities(caps);
        setRequest(caps.default_request);
        return runRace(caps.default_request, { useLiveTriton: false });
      })
      .then(applyRaceResult)
      .catch((error) => setWarnings([String(error)]));
  }, []);

  function applyRaceResult(result: RaceResult) {
    liveRunSeqRef.current += 1;
    liveSocketRef.current?.close();
    liveSocketRef.current = null;
    resetLiveProgress();
    void playerRef.current?.stop();
    setLiveBusy(false);
    setRace(result);
    setLiveEvents([]);
    clearLiveAudioUrl();
  }

  useEffect(() => {
    if (!liveClockRunning || liveStartedAtRef.current === null) {
      return;
    }
    let frame = 0;
    const tick = () => {
      if (liveStartedAtRef.current !== null) {
        const elapsedMs = performance.now() - liveStartedAtRef.current;
        const playbackEndMs = livePlaybackEndMsRef.current;
        if (playbackEndMs !== null && elapsedMs >= playbackEndMs) {
          setLiveClockMs(playbackEndMs);
          liveStartedAtRef.current = null;
          livePlaybackEndMsRef.current = null;
          setLiveClockRunning(false);
          setLiveBusy(false);
          return;
        }
        setLiveClockMs(elapsedMs);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [liveClockRunning]);

  function resetLiveProgress() {
    liveStartedAtRef.current = null;
    liveFirstAudioMsRef.current = null;
    liveAudioDurationMsRef.current = 0;
    livePlaybackEndMsRef.current = null;
    setLiveClockRunning(false);
  }

  function clearLiveAudioUrl() {
    if (liveAudioUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(liveAudioUrl);
    }
    setLiveAudioUrl(undefined);
    liveAudioPartsRef.current = [];
    liveAudioFormatRef.current = DEFAULT_AUDIO;
  }

  async function onRace() {
    setBusy(true);
    setWarnings([]);
    try {
      const result = await runRace(request, { useLiveTriton: true, useLiveEngine: true });
      applyRaceResult(result);
      setWarnings(result.warnings ?? []);
    } catch (error) {
      setWarnings([String(error)]);
    } finally {
      setBusy(false);
    }
  }

  async function onRaceOfficial() {
    setBusy(true);
    setWarnings([]);
    try {
      const result = await runRace(request, { useLiveTriton: true, useLiveEngine: true, liveBaselines: true });
      applyRaceResult(result);
      setWarnings(result.warnings ?? []);
    } catch (error) {
      setWarnings([String(error)]);
    } finally {
      setBusy(false);
    }
  }

  async function onCollectRace() {
    raceCaptureSocketRef.current?.close();
    setCaptureRunning(true);
    setCaptureLogs([]);
    setWarnings([]);
    try {
      const jobId = await startRaceCapture({
        ...request,
        triton_slots: capabilities?.concurrency?.triton_active_slot_limit ?? 128,
      });
      const socket = new WebSocket(wsUrl(`/api/v1/race-capture/${jobId}`));
      raceCaptureSocketRef.current = socket;
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as RaceCaptureEvent;
        if (message.type === "race_capture_started") {
          setCaptureLogs((current) => [...current, `$ ${message.command}`]);
        }
        if (message.type === "race_capture_log") {
          setCaptureLogs((current) => [...current, message.message]);
        }
        if (message.type === "race_capture_done") {
          applyRaceResult(message.race);
          setWarnings(message.race.warnings ?? []);
          setCaptureLogs((current) => [...current, "race capture complete"]);
          setCaptureRunning(false);
          socket.close();
        }
        if (message.type === "race_capture_error") {
          setWarnings((current) => [...current, message.message]);
          setCaptureLogs((current) => [...current, message.message]);
          setCaptureRunning(false);
          socket.close();
        }
      };
      socket.onerror = () => {
        setWarnings((current) => [...current, "race capture websocket failed"]);
        setCaptureLogs((current) => [...current, "race capture websocket failed"]);
        setCaptureRunning(false);
      };
      socket.onclose = () => {
        if (raceCaptureSocketRef.current === socket) {
          raceCaptureSocketRef.current = null;
        }
      };
    } catch (error) {
      setWarnings([String(error)]);
      setCaptureLogs((current) => [...current, String(error)]);
      setCaptureRunning(false);
    }
  }

  async function speakTrt() {
    const runSeq = liveRunSeqRef.current + 1;
    liveRunSeqRef.current = runSeq;
    liveSocketRef.current?.close();
    liveSocketRef.current = null;
    resetLiveProgress();
    setLiveBusy(true);
    setLiveEvents([]);
    setLiveClockMs(0);
    clearLiveAudioUrl();
    setWarnings([]);
    await playerRef.current?.stop();
    const player = new PcmStreamPlayer();
    playerRef.current = player;
    await player.start();

    const socket = new WebSocket(wsUrl("/api/v1/trt-live"));
    liveSocketRef.current = socket;
    socket.binaryType = "arraybuffer";
    let audioFormat = DEFAULT_AUDIO;
    socket.onopen = () => {
      if (runSeq !== liveRunSeqRef.current) {
        socket.close();
        return;
      }
      liveStartedAtRef.current = performance.now();
      setLiveClockRunning(true);
      socket.send(JSON.stringify({ type: "speak", ...request }));
    };
    socket.onmessage = (event) => {
      if (runSeq !== liveRunSeqRef.current) {
        return;
      }
      if (typeof event.data !== "string") {
        liveAudioPartsRef.current.push(event.data.slice(0));
        liveAudioDurationMsRef.current += audioDurationMs(event.data.byteLength, audioFormat);
        player.enqueue(event.data, audioFormat);
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type === "event") {
        const trace = message.event as TraceEvent;
        const maybeFormat = trace.meta?.audio_format as AudioFormat | undefined;
        if (maybeFormat) {
          audioFormat = maybeFormat;
          liveAudioFormatRef.current = maybeFormat;
        }
        if (trace.type === "first_audio_chunk") {
          liveFirstAudioMsRef.current = trace.t_ms;
        }
        setLiveEvents((current) => [...current, trace]);
        if (trace.type === "warning" && trace.meta?.message) {
          setWarnings((current) => [...current, String(trace.meta?.message)]);
        }
        if (trace.type === "done") {
          const playbackEndMs =
            liveFirstAudioMsRef.current === null
              ? trace.t_ms
              : Math.max(trace.t_ms, liveFirstAudioMsRef.current + liveAudioDurationMsRef.current);
          setLiveClockMs(trace.t_ms);
          livePlaybackEndMsRef.current = playbackEndMs;
          if (liveAudioPartsRef.current.length > 0) {
            const blob = wavBlobFromPcmF32(liveAudioPartsRef.current, liveAudioFormatRef.current.sample_rate || 24000);
            setLiveAudioUrl(URL.createObjectURL(blob));
          }
          if (playbackEndMs <= trace.t_ms + 16) {
            liveStartedAtRef.current = null;
            livePlaybackEndMsRef.current = null;
            setLiveClockRunning(false);
            setLiveBusy(false);
          }
          socket.close();
        }
      }
      if (message.type === "error") {
        setWarnings((current) => [...current, message.message]);
        resetLiveProgress();
        setLiveBusy(false);
      }
    };
    socket.onerror = () => {
      if (runSeq !== liveRunSeqRef.current) {
        return;
      }
      setWarnings((current) => [...current, "TRT live websocket failed"]);
      resetLiveProgress();
      setLiveBusy(false);
    };
    socket.onclose = () => {
      if (runSeq !== liveRunSeqRef.current) {
        return;
      }
      if (liveSocketRef.current === socket) {
        liveSocketRef.current = null;
      }
      if (livePlaybackEndMsRef.current !== null) {
        return;
      }
      resetLiveProgress();
      setLiveBusy(false);
    };
  }

  const liveFirstAudio = liveEvents.find((event) => event.type === "first_audio_chunk")?.t_ms;
  const tritonReady = capabilities?.backends.find((backend) => backend.id === "triton_trt_streaming")?.live_available;
  const officialReady = capabilities?.backends.some((backend) => backend.id.startsWith("official") && backend.live_available);
  const release = race?.release ?? capabilities?.release;
  const limitations = race?.limitations ?? capabilities?.limitations ?? [];
  const showingLiveTrace = liveBusy || liveEvents.length > 0;
  const tokenEvents = useMemo(
    () => (showingLiveTrace ? liveEvents : preferredTokenEvents(race, [])),
    [race, liveEvents, showingLiveTrace]
  );
  const tritonRaceResult = race?.results.find((result) => result.backend === "triton_trt_streaming");
  const raceHasLiveCapture = race?.results.some((result) => result.source !== "fixture") ?? false;
  const textPlayerAudioUrl = liveAudioUrl ?? (tritonRaceResult?.audio?.url ? mediaUrl(tritonRaceResult.audio.url) : undefined);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Qwen3-TTS Triton</h1>
          <p>PyTorch offline/online-text vs TensorRT token streaming</p>
        </div>
        <div className="headline-metrics">
          <div>
            <span>Cache-hit TTFT</span>
            <strong>{capabilities?.headline.single_stream_cache_hit_ttft_ms ?? 13}ms</strong>
          </div>
          <div>
            <span>128-stream avg</span>
            <strong>{capabilities?.headline.concurrent_128_avg_ttft_ms ?? 180}ms</strong>
          </div>
        </div>
      </header>

      <section className="release-notice">
        <div>
          <AlertTriangle size={17} />
          <strong>{release?.stage === "engineering_preview" ? "工程预览版" : "Preview"}</strong>
          <span>{release?.positioning ?? "当前演示用于展示 TensorRT token streaming 链路，不代表生产稳定性。"}</span>
        </div>
        {limitations.length > 0 && (
          <ul>
            {limitations.slice(0, 4).map((item) => <li key={item}>{item}</li>)}
          </ul>
        )}
      </section>

      <section className="control-surface">
        <textarea value={request.text} onChange={(event) => setRequest({ ...request, text: event.target.value })} />
        <div className="controls-row">
          <input value={request.speaker} onChange={(event) => setRequest({ ...request, speaker: event.target.value })} aria-label="speaker" />
          <input value={request.language} onChange={(event) => setRequest({ ...request, language: event.target.value })} aria-label="language" />
          <select value={request.cache_mode} onChange={(event) => setRequest({ ...request, cache_mode: event.target.value as DemoRequest["cache_mode"] })}>
            <option value="hit">Cache hit</option>
            <option value="miss">Cache miss</option>
            <option value="auto">Auto</option>
          </select>
          <button className="button primary" onClick={speakTrt} disabled={liveBusy}>
            <Zap size={16} />
            {liveBusy ? "Speaking" : "Speak TRT"}
          </button>
          <button className="button ghost" onClick={() => runRace(request, { useLiveTriton: false }).then(applyRaceResult)}>
            <RotateCcw size={16} />
            Trace
          </button>
        </div>
      </section>

      <section className="status-strip">
        <div className={`status-pill ${tritonReady ? "ready" : "muted"}`}>
          <RadioTower size={15} />
          Triton {tritonReady ? "live" : "fixture fallback"}
        </div>
        <div className={`status-pill ${officialReady ? "ready" : "muted"}`}>
          <RadioTower size={15} />
          Official API {officialReady ? "enabled" : "fixture"}
        </div>
        <div className="status-pill">
          <Activity size={15} />
          Live first audio {formatMs(liveFirstAudio)}
        </div>
        <div className="status-pill muted">
          {raceHasLiveCapture ? "Captured trace loaded" : race?.source_path ? "Fixture trace loaded" : "Live result pending"}
        </div>
        <button className="button tiny" onClick={() => downloadJson(race)}>
          <Download size={14} />
          JSON
        </button>
      </section>

      {warnings.length > 0 && (
        <section className="warning-band">
          {warnings.map((warning, index) => <span key={`${warning}-${index}`}>{warning}</span>)}
        </section>
      )}

      <TextPlayer
        events={tokenEvents}
        text={request.text}
        audioUrl={textPlayerAudioUrl}
        liveMs={showingLiveTrace ? liveClockMs : 0}
        live={liveBusy}
        source={showingLiveTrace ? "live TRT stream" : tritonRaceResult?.source ?? "trace"}
      />

      <PerformanceRace
        race={race}
        busy={busy}
        captureRunning={captureRunning}
        captureLogs={captureLogs}
        onCollectRace={onCollectRace}
        onRace={onRace}
        onRaceOfficial={onRaceOfficial}
        onPlayAligned={(items) => playAudioRace(items, true, raceAudioRef)}
        onPlayOne={(item) => playAudioRace([item], false, raceAudioRef)}
      />

      {race && (
        <section className="panel conditions-panel">
          <div className="panel-head">
            <div>
              <h2>Benchmark Conditions</h2>
              <p>hardware, cache, precision, replay mode</p>
            </div>
          </div>
          <div className="condition-grid">
            {Object.entries(race.benchmark_conditions).map(([key, value]) => (
              <div key={key}>
                <span>{key.replace(/_/g, " ")}</span>
                <strong>{String(value)}</strong>
              </div>
            ))}
          </div>
        </section>
      )}

      <ConcurrencyPanel request={request} />
    </main>
  );
}

function preferredTokenEvents(race: RaceResult | null, liveEvents: TraceEvent[]): TraceEvent[] {
  if (liveEvents.length > 0) {
    return liveEvents;
  }
  const triton = race?.results.find((result) => result.backend === "triton_trt_streaming");
  if (triton?.events.length) {
    return triton.events;
  }
  const engine = race?.results.find((result) => result.backend === "bare_engine_streaming");
  return engine?.events ?? [];
}

async function playAudioRace(
  results: Array<{ audio?: { url: string; scheduled_start_ms?: number }; label: string }>,
  measuredWait: boolean,
  contextRef: MutableRefObject<AudioContext | null>
): Promise<void> {
  if (contextRef.current) {
    await contextRef.current.close();
  }
  const playable = results.filter((result) => result.audio?.url);
  if (playable.length === 0) {
    return;
  }
  const context = new AudioContext();
  contextRef.current = context;
  await context.resume();
  const decoded = await Promise.all(
    playable.map(async (result) => {
      const response = await fetch(mediaUrl(result.audio!.url));
      if (!response.ok) {
        throw new Error(`audio fetch failed for ${result.label}: ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      return {
        result,
        audio: await context.decodeAudioData(buffer.slice(0)),
      };
    })
  );
  const baseTime = context.currentTime + 0.08;
  decoded.forEach(({ result, audio }) => {
    const source = context.createBufferSource();
    const gain = context.createGain();
    source.buffer = audio;
    gain.gain.value = playable.length > 1 ? 0.42 : 0.9;
    source.connect(gain);
    gain.connect(context.destination);
    const delaySec = measuredWait ? Math.max(0, result.audio?.scheduled_start_ms ?? 0) / 1000 : 0;
    source.start(baseTime + delaySec);
  });
}

function downloadJson(race: RaceResult | null): void {
  if (!race) return;
  const blob = new Blob([JSON.stringify(race, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "qwen3-tts-race-result.json";
  link.click();
  URL.revokeObjectURL(url);
}

function audioDurationMs(byteLength: number, format: AudioFormat): number {
  const sampleRate = format.sample_rate || 24000;
  const channels = format.channels || 1;
  const bytesPerSample = format.encoding === "pcm_s16le" ? 2 : 4;
  return (byteLength / (sampleRate * channels * bytesPerSample)) * 1000;
}
