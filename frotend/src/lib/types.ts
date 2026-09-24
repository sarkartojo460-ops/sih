export type JobStatus = "queued" | "running" | "done" | "failed";
export type Severity = "low" | "moderate" | "severe";
export type OperatingMode = "live" | "replay";

export interface HealthResponse {
  status: string;
  version: string;
}

export interface PipelineRunRequest {
  event_name: string;
  seed: number | null;
}

export interface BoundingBox4D {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  z_min: number;
  z_max: number;
  t_start: number;
  t_end: number;
  n_points: number;
}

export interface PipelineResult {
  alert_id: number;
  centroid_row: number;
  centroid_col: number;
  peak_value: number;
  severity: Severity;
  confidence: number;
  radius_km: number;
  raw_score: number;
  physics_score: number;
  efi_peak: number;
  unet_peak: number;
  diffusion_peak: number;
  amplitude_gain_pct: number;
  bounding_box: BoundingBox4D;
  n_tracks: number;
  gnn_loss_history: number[];
  seed: number;
  elapsed_seconds: number;
  latitude?: number;
  longitude?: number;
}

export interface PipelineJobStatus {
  job_id: string;
  status: JobStatus;
  progress: string;
  result: PipelineResult | null;
  error: string | null;
}

export interface ReplayEvents {
  available_events: string[];
}

export interface Alert {
  id: number;
  event_name: string;
  centroid_row: number;
  centroid_col: number;
  peak_value: number;
  severity: Severity;
  confidence: number;
  radius_km: number;
  physics_score: number;
  efi_peak: number;
  unet_peak: number;
  diffusion_peak: number;
  amplitude_gain_pct: number;
  created_at: string | null;
  latitude?: number;
  longitude?: number;
}

export interface Metrics {
  total_alerts: number;
  by_severity: Record<Severity, number>;
  avg_physics_score: number;
  avg_diffusion_vs_unet_amplitude_gain_pct: number;
}

export interface AlertsQuery {
  limit?: number;
  severity?: Severity;
  event_name?: string;
}

export interface ApiErrorShape {
  detail?: string;
}

export interface JobRecord {
  id: string;
  mode: OperatingMode;
  eventName: string;
  startedAt: string;
  completedAt?: string;
  status: JobStatus;
  result?: PipelineResult;
}