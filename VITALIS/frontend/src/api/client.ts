const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface User {
  id: number;
  name: string;
  email: string;
  role: "NURSE" | "DOCTOR";
  is_active: boolean;
}

export interface Patient {
  id: number;
  patient_identifier: string;
  name: string;
  age: number | null;
  gender: string | null;
  race: string | null;
  insurance: string | null;
  admission_type: string | null;
  admission_location: string | null;
  icu_unit: string | null;
  bed_number: string | null;
  admission_diagnosis: string | null;
  height_cm: number | null;
  weight_kg: number | null;
  status: string;
  created_by: number | null;
  created_at: string;
  updated_at: string;
}

export interface DoctorOption {
  id: number;
  name: string;
  email: string;
}

export interface Assignment {
  id: number;
  patient_id: number;
  doctor_id: number;
  assigned_by: number;
  notes: string | null;
  assigned_at: string;
  unassigned_at: string | null;
}

export interface ShapFeature {
  feature: string;
  value: number | null;
  shap_value: number;
  abs_shap_value: number;
}

export interface RiskEvent {
  id: number;
  patient_id: number;
  stay_id: number;
  window_id: number;
  status: string;
  event_time: string | null;
  window_start: string | null;
  window_end: string | null;
  sequence_length: number | null;
  prediction: number | null;
  threshold: number | null;
  xgb_raw_probability: number | null;
  xgb_probability: number | null;
  lstm_raw_probability: number | null;
  lstm_probability: number | null;
  ensemble_probability: number | null;
  model_version: string | null;
  latency_ms: number | null;
  shap_status: string | null;
  top_features: ShapFeature[] | null;
  created_at: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface CreatePatientPayload {
  patient_identifier: string;
  name: string;
  age: number;
  gender: string;
  race?: string | null;
  insurance?: string | null;
  admission_type?: string | null;
  admission_location?: string | null;
  icu_unit?: string | null;
  bed_number?: string | null;
  admission_diagnosis?: string | null;
  height_cm?: number | null;
  weight_kg?: number | null;
}

function getToken(): string | null {
  return localStorage.getItem("vitalis_token");
}

function clearToken(): void {
  localStorage.removeItem("vitalis_token");
  localStorage.removeItem("vitalis_user");
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken();

  const headers = new Headers(options.headers);

  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers,
    });
  } catch (error) {
    console.error("VITALIS API connection error:", error);

    throw new Error(
      `Unable to connect to VITALIS backend at ${API_BASE_URL}. ` +
        "Make sure the FastAPI server is running.",
    );
  }

  if (response.status === 401) {
    clearToken();
    window.dispatchEvent(new Event("vitalis:logout"));
  }

  const contentType =
    response.headers.get("content-type") || "";

  let data: unknown;

  if (contentType.includes("application/json")) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}.`;

    if (
      typeof data === "object" &&
      data !== null &&
      "detail" in data
    ) {
      const detail = (data as { detail: unknown }).detail;

      if (typeof detail === "string") {
        message = detail;
      } else {
        message = JSON.stringify(detail);
      }
    } else if (
      typeof data === "string" &&
      data.trim()
    ) {
      message = data;
    }

    throw new Error(message);
  }

  return data as T;
}

export async function login(
  email: string,
  password: string,
): Promise<LoginResponse> {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({
      email: email.trim().toLowerCase(),
      password,
    }),
  });
}

export async function signup(payload: {
  name: string;
  email: string;
  password: string;
  confirm_password?: string;
}): Promise<unknown> {
  return request("/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getMe(): Promise<User> {
  return request<User>("/auth/me");
}

export async function getPatients(): Promise<Patient[]> {
  const result = await request<Patient[]>("/patients");

  return Array.isArray(result) ? result : [];
}

export async function getPatient(
  patientId: number,
): Promise<Patient> {
  return request<Patient>(`/patients/${patientId}`);
}

export async function createPatient(
  payload: CreatePatientPayload,
): Promise<Patient> {
  return request<Patient>("/patients", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getDoctors(): Promise<DoctorOption[]> {
  const result = await request<DoctorOption[]>(
    "/patients/doctors",
  );

  return Array.isArray(result) ? result : [];
}

export async function assignDoctor(
  patientId: number,
  doctorId: number,
  notes?: string,
): Promise<Assignment> {
  return request<Assignment>(
    `/patients/${patientId}/assign`,
    {
      method: "POST",
      body: JSON.stringify({
        doctor_id: doctorId,
        notes: notes?.trim() || null,
      }),
    },
  );
}

export async function getAssignments(
  patientId: number,
): Promise<Assignment[]> {
  const result = await request<Assignment[]>(
    `/patients/${patientId}/assignments`,
  );

  return Array.isArray(result) ? result : [];
}

export async function dischargePatient(
  patientId: number,
): Promise<Patient> {
  return request<Patient>(
    `/patients/${patientId}/discharge`,
    {
      method: "POST",
    },
  );
}

export async function getLatestRisk(
  patientId: number,
): Promise<RiskEvent> {
  return request<RiskEvent>(
    `/patients/${patientId}/risk/latest`,
  );
}

export async function getRiskHistory(
  patientId: number,
  limit = 100,
): Promise<RiskEvent[]> {
  return request<RiskEvent[]>(
    `/patients/${patientId}/risk/history?limit=${limit}`,
  );
}

export async function getRiskExplanation(
  patientId: number,
): Promise<RiskEvent> {
  return request<RiskEvent>(
    `/patients/${patientId}/risk/explanation`,
  );
}

export function saveAuth(
  token: string,
  user: User,
): void {
  localStorage.setItem(
    "vitalis_token",
    token,
  );

  localStorage.setItem(
    "vitalis_user",
    JSON.stringify(user),
  );
}

export function loadStoredUser(): User | null {
  const raw =
    localStorage.getItem("vitalis_user");

  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as User;
  } catch {
    return null;
  }
}

export function logout(): void {
  clearToken();
}

export { API_BASE_URL };