import type { Capabilities, DemoRequest, LlmPkRequest, LlmPkResult } from "./types";

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

export function runLlmPk(request: LlmPkRequest): Promise<LlmPkResult> {
  return jsonFetch<LlmPkResult>("/api/v1/llm-pk", {
    method: "POST",
    body: JSON.stringify(request)
  });
}

export async function startConcurrency(payload: DemoRequest & { concurrency: number; live: boolean }): Promise<string> {
  const response = await jsonFetch<{ job_id: string }>("/api/v1/concurrency", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  return response.job_id;
}
