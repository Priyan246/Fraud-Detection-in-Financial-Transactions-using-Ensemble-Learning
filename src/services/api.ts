import type { Transaction, PredictResponse, HealthStatus, ModelMetadata } from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      let errMsg = error.detail;
      if (Array.isArray(errMsg)) {
        errMsg = errMsg.map((e: any) => `${e.loc?.slice(1).join('.')}: ${e.msg}`).join(', ');
      } else if (typeof errMsg === 'object') {
        errMsg = JSON.stringify(errMsg);
      }
      throw new Error(errMsg || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async predict(transactions: Transaction[], model: string = 'ensemble', returnAll: boolean = false): Promise<PredictResponse> {
    return this.fetch<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({
        transactions,
        model,
        return_all: returnAll,
      }),
    });
  }

  async getHealth(): Promise<HealthStatus> {
    return this.fetch<HealthStatus>('/health');
  }

  async getMetadata(): Promise<ModelMetadata> {
    return this.fetch<ModelMetadata>('/metadata');
  }

  async resetCardHistory(): Promise<{ status: string }> {
    return this.fetch<{ status: string }>('/reset_card_history', {
      method: 'POST',
    });
  }
}

export const apiService = new ApiService();
