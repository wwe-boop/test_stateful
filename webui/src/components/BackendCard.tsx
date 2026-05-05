import { Clock, Play, RadioTower, Timer, Volume2, Waves } from "lucide-react";
import type { ReactNode } from "react";
import type { RunResult } from "../types";
import { Timeline, formatMs } from "./Timeline";

interface BackendCardProps {
  result: RunResult;
  maxMs: number;
  onPlayNow?: (result: RunResult) => void;
  onPlayDelayed?: (result: RunResult) => void;
}

export function BackendCard({ result, maxMs, onPlayNow, onPlayDelayed }: BackendCardProps) {
  const officialHighLevel = result.source === "live_official_pytorch" || result.backend.startsWith("official_pytorch");
  const hasServerTtft = result.metrics.server_ttft_ms !== undefined && result.metrics.server_ttft_ms !== null;
  const hasTritonAdapterTtft = result.metrics.triton_adapter_ttft_ms !== undefined && result.metrics.triton_adapter_ttft_ms !== null;
  const metric = officialHighLevel
    ? result.metrics.official_approx_ttft_ms ?? result.metrics.first_playable_ms
    : result.metrics.server_ttft_ms
      ?? result.metrics.triton_adapter_ttft_ms
      ?? result.metrics.engine_internal_ttft_ms
      ?? result.metrics.client_ttfb_ms
      ?? result.metrics.first_playable_ms;
  const metricLabel = officialHighLevel
    ? "TTFT approx"
    : hasServerTtft
      ? "Server TTFT"
      : hasTritonAdapterTtft
        ? "Triton TTFT"
        : "Client TTFB";
  return (
    <section className={`backend-card backend-${result.backend}`}>
      <div className="backend-head">
        <div>
          <h3>{result.label}</h3>
          <p>{result.mode}</p>
        </div>
        <SourceBadge source={result.source} />
      </div>
      <div className="primary-metric">
        <Timer size={18} />
        <div>
          <span>{metricLabel}</span>
          <strong>{formatMs(metric)}</strong>
        </div>
      </div>
      <div className="metric-grid">
        <SmallMetric icon={<Volume2 size={15} />} label="Audible" value={formatMs(result.metrics.first_audible_ms)} />
        <SmallMetric
          icon={<Waves size={15} />}
          label={result.metrics.triton_adapter_ttft_ms ? "Client TTFB" : "Full Ready"}
          value={formatMs(result.metrics.triton_adapter_ttft_ms ? result.metrics.client_ttfb_ms : result.metrics.full_audio_ready_ms)}
        />
        <SmallMetric icon={<RadioTower size={15} />} label="Chunks" value={`${result.metrics.chunks ?? 0}`} />
      </div>
      <Timeline events={result.events} maxMs={maxMs} />
      <div className="card-actions">
        <button className="button tiny" onClick={() => onPlayDelayed?.(result)} disabled={!result.audio?.url}>
          <Clock size={14} />
          Wait
        </button>
        <button className="button tiny" onClick={() => onPlayNow?.(result)} disabled={!result.audio?.url}>
          <Play size={14} />
          Audio
        </button>
      </div>
    </section>
  );
}

function SourceBadge({ source }: { source: string }) {
  const label = source === "live_triton" ? "live" : source === "fixture" ? "trace" : source;
  return <span className={`source-badge source-${source}`}>{label}</span>;
}

function SmallMetric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="small-metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
