import type {
  Alert,
  AlertsQuery,
  ApiErrorShape,
  HealthResponse,
  Metrics,
  PipelineJobStatus,
  PipelineResult,
  PipelineRunRequest,
  ReplayEvents,
} from "./types";

export const API_BASE_URL = (import.meta.env["VITE_API_BASE_URL"] || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly endpoint: string,
    public readonly developerDetail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const init: RequestInit = {
    ...options,
    headers: { Accept: "application/json", ...options.headers },
  };
  if (signal) init.signal = signal;
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let body: ApiErrorShape = {};
    try { body = (await response.json()) as ApiErrorShape; } catch { /* non-JSON upstream */ }
    const fallback = response.status === 404 ? "Requested record was not found." : "The STORMTRACE engine rejected this request.";
    throw new ApiError(body.detail || fallback, response.status, path, body.detail);
  }
  return response.json() as Promise<T>;
}

const jsonPost = (body?: unknown): RequestInit => body === undefined
  ? { method: "POST" }
  : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/health", {}, signal),
  runPipeline: (input: PipelineRunRequest, signal?: AbortSignal) => request<PipelineJobStatus>("/pipeline/run", jsonPost(input), signal),
  getJob: (jobId: string, signal?: AbortSignal) => request<PipelineJobStatus>(`/pipeline/status/${encodeURIComponent(jobId)}`, {}, signal),
  runSync: (input: PipelineRunRequest, signal?: AbortSignal) => request<PipelineResult>("/pipeline/run_sync", jsonPost(input), signal),
  replayEvents: (signal?: AbortSignal) => request<ReplayEvents>("/replay/events", {}, signal),
  runReplay: (eventKey: string, signal?: AbortSignal) => request<PipelineJobStatus>(`/replay/${encodeURIComponent(eventKey)}`, jsonPost(), signal),
  alerts: (query: AlertsQuery = {}, signal?: AbortSignal) => {
    const params = new URLSearchParams();
    if (query.limit) params.set("limit", String(query.limit));
    if (query.severity) params.set("severity", query.severity);
    if (query.event_name) params.set("event_name", query.event_name);
    const suffix = params.size ? `?${params.toString()}` : "";
    return request<Alert[]>(`/alerts/latest${suffix}`, {}, signal);
  },
  alert: (alertId: number, signal?: AbortSignal) => request<Alert>(`/alerts/${alertId}`, {}, signal),
  metrics: (signal?: AbortSignal) => request<Metrics>("/metrics", {}, signal),
};