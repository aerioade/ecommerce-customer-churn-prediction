const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"

export type CustomerFeatures = {
  Tenure: number | null
  WarehouseToHome: number | null
  NumberOfDeviceRegistered: number
  PreferedOrderCat: string
  SatisfactionScore: number
  MaritalStatus: string
  NumberOfAddress: number
  Complain: number
  DaySinceLastOrder: number | null
  CashbackAmount: number
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail ?? "Tidak dapat terhubung ke ChurnGuard API.")
  }
  return response.json() as Promise<T>
}

export const api = {
  dashboardSummary: () => request<any>("/api/dashboard-summary"),
  modelAnalytics: () => request<any>("/api/model-analytics"),
  edaInsights: () => request<any>("/api/eda-insights"),
  featureImportance: () => request<any>("/api/feature-importance"),
  predictIndividual: (customer: CustomerFeatures) => request<any>("/api/predict/individual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  }),
  predictBatch: (file: File) => {
    const form = new FormData()
    form.append("file", file)
    return request<any>("/api/predict/batch", { method: "POST", body: form })
  },
}
