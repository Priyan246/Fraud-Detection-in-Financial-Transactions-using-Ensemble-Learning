export type ModelType = 'ensemble' | 'LightGBM' | 'XGBoost' | 'CatBoost';

export interface Transaction {
  trans_date_trans_time: string;
  cc_num: string;
  merchant: string;
  category: string;
  amt: number;
  gender: string;
  city: string;
  state: string;
  zip: string;
  lat: number;
  long: number;
  city_pop: number;
  job: string;
  dob: string;
  unix_time: number;
  merch_lat: number;
  merch_long: number;
}

export interface PredictionResult {
  fraud_probability: number;
  is_fraud_predicted: boolean;
  status: 'Safe' | 'Uncertain' | 'Fraud';
  threshold: number;
  model_used: string;
  lgb_score?: number;
  xgb_score?: number;
  cb_score?: number;
}

export interface PredictResponse {
  predictions: PredictionResult[];
  n_transactions: number;
}

export interface HealthResponse {
  status: string;
  n_features: number;
  models_loaded: string[];
  card_registry_size: number;
}

export interface MetadataResponse {
  thresholds: Record<string, number>;
  cat_feature_indices: number[];
}
