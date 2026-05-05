import type { Capabilities, DemoRequest, RaceResult } from "./types";

const API_BASE = (import.meta.env.VITE_DEMO_API_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function mediaUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return apiUrl(path);
}

export function wsUrl(path: string): string {
  if (API_BASE.startsWith("http://")) {
    return `ws://${API_BASE.slice("http://".length)}${path}`;
  }
  if (API_BASE.startsWith("https://")) {
    return `wss://${API_BASE.slice("https://".length)}${path}`;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
  return (await response.json()) as T;
}

export function getCapabilities(): Promise<Capabilities> {
  return jsonFetch<Capabilities>("/api/v1/capabilities");
}

export function runRace(
  request: DemoRequest,
  options: { useLiveTriton: boolean; useLiveEngine?: boolean; liveBaselines?: boolean }
): Promise<RaceResult> {
  return jsonFetch<RaceResult>("/api/v1/race", {
    method: "POST",
    body: JSON.stringify({
      ...request,
      use_live_triton: options.useLiveTriton,
      use_live_engine: options.useLiveEngine ?? false,
      live_baselines: options.liveBaselines ?? false
    })
  });
}

export async function startRaceCapture(payload: DemoRequest & { triton_slots?: number; strict?: boolean }): Promise<string> {
  const response = await jsonFetch<{ job_id: string }>("/api/v1/race-capture", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  return response.job_id;
}

export async function startConcurrency(payload: DemoRequest & { concurrency: number; live: boolean }): Promise<string> {
  const response = await jsonFetch<{ job_id: string }>("/api/v1/concurrency", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  return response.job_id;
}
