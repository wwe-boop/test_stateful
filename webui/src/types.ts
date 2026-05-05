export type BackendId =
  | "triton_streaming"
  | "triton_offline"
  | "triton_trt_streaming";

export interface TraceEvent {
  run_id: string;
  backend: BackendId;
  type: string;
  t_ms: number;
  server_t_ms?: number;
  stream_id?: string;
  text?: string;
  meta?: Record<string, unknown>;
}

export interface RunMetrics {
  first_playable_ms?: number;
  total_ms?: number;
  server_ttft_ms?: number;
  triton_adapter_ttft_ms?: number;
  engine_internal_ttft_ms?: number;
  client_ttfb_ms?: number;
  first_audible_ms?: number;
  full_audio_ready_ms?: number;
  audio_duration_ms?: number;
  simulated_llm_complete_ms?: number;
  chunks?: number;
  cache_hit?: boolean;
}

export interface RunResult {
  run_id: string;
  backend: BackendId;
  label: string;
  mode: string;
  source: string;
  metrics: RunMetrics;
  events: TraceEvent[];
  warnings: string[];
  audio_format: AudioFormat;
  audio?: AudioAsset;
}

export interface AudioFormat {
  encoding: "pcm_f32" | "pcm_s16le" | string;
  sample_rate: number;
  channels: number;
}

export interface AudioAsset {
  id?: string;
  url: string;
  encoding: "wav" | string;
  sample_rate?: number;
  scheduled_start_ms?: number;
  source?: string;
}

export interface DemoRequest {
  text: string;
  speaker: string;
  language: string;
}

export interface LlmPkRequest extends DemoRequest {
  ms_per_token: number;
}

export interface Capabilities {
  default_request: LlmPkRequest;
  backends: Array<{
    id: BackendId;
    label: string;
    streaming: boolean;
    live_available: boolean;
    endpoint?: string;
    model?: string;
    error?: string;
    runtime?: {
      max_batch_slots?: number;
      max_sessions?: number;
    };
  }>;
  concurrency?: {
    live_enabled: boolean;
    triton_active_slot_limit: number;
    triton_max_sessions: number;
  };
  release?: ReleaseMetadata;
  limitations?: string[];
  headline: {
    single_stream_cache_hit_ttft_ms: number;
    concurrent_128_avg_ttft_ms: number;
  };
}

export interface ReleaseMetadata {
  stage: string;
  positioning: string;
  recommended_variant: string;
  stable_paths: string[];
  experimental_paths: string[];
  planned_paths: string[];
}

export interface LlmPkResult {
  type: "llm_pk_result";
  request: LlmPkRequest;
  results: RunResult[];
  warnings: string[];
  release?: ReleaseMetadata;
  limitations?: string[];
}

export interface LaneUpdate {
  type: "lane_update";
  job_id: string;
  stream_id: string;
  status: "playing" | "error" | string;
  ttft_ms?: number;
  queued_by_slot_limit?: boolean;
  elapsed_ms?: number;
  error?: string;
  audio?: AudioAsset;
}

export interface ConcurrencySummary {
  type: "summary";
  job_id: string;
  source: string;
  concurrency: number;
  failed_streams: number;
  elapsed_ms: number;
  active_slot_limit?: number;
  queued_streams?: number;
  count: number;
  avg_ttft_ms?: number;
  active_avg_ttft_ms?: number;
  queued_avg_ttft_ms?: number;
  p50_ttft_ms?: number;
  p90_ttft_ms?: number;
  p99_ttft_ms?: number;
  max_ttft_ms?: number;
  throughput_audio_sec_per_sec?: number;
}
