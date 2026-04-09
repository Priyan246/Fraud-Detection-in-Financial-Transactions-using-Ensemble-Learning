import type { Transaction, PredictResponse, HealthResponse, MetadataResponse, ModelType } from '@/types/api';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => res.statusText);
    throw new Error(`API error ${res.status}: ${errorText}`);
  }

  return res.json() as Promise<T>;
}

export const apiService = {
  predict(transactions: Transaction[], model: ModelType = 'ensemble', returnAll = false): Promise<PredictResponse> {
    return request<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({ transactions, model, return_all: returnAll }),
    });
  },

  health(): Promise<HealthResponse> {
    return request<HealthResponse>('/health');
  },

  metadata(): Promise<MetadataResponse> {
    return request<MetadataResponse>('/metadata');
  },

  resetCardHistory(): Promise<{ status: string }> {
    return request<{ status: string }>('/reset_card_history', { method: 'POST' });
  },
};
