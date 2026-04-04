import { motion } from 'framer-motion';
import { 
  AlertTriangle, 
  CheckCircle, 
  TrendingUp, 
  BarChart3, 
  Shield,
  ArrowLeft
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import type { PredictionResult } from '@/types/api';

interface ResultsDisplayProps {
  results: PredictionResult[];
  onNewAnalysis: () => void;
}

const getRiskLevel = (probability: number) => {
  if (probability >= 0.7) return { level: 'High Risk', bgColor: 'bg-foreground/15' };
  if (probability >= 0.4) return { level: 'Medium Risk', bgColor: 'bg-muted' };
  return { level: 'Low Risk', bgColor: 'bg-muted' };
};

const ModelScoreBar = ({ label, score }: { label: string; score: number }) => (
  <div className="space-y-1.5">
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground">{(score * 100).toFixed(1)}%</span>
    </div>
    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${score * 100}%` }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="h-full bg-foreground rounded-full"
      />
    </div>
  </div>
);

const ResultCard = ({ result, index }: { result: PredictionResult; index: number }) => {
  const risk = getRiskLevel(result.fraud_probability);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
    >
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-medium text-foreground">
              Result #{index + 1}
            </CardTitle>
            <div className={`px-2.5 py-1 rounded-md ${risk.bgColor} text-xs font-medium flex items-center gap-1.5 text-foreground`}>
              {result.is_fraud_predicted ? (
                <AlertTriangle className="w-3.5 h-3.5" />
              ) : (
                <CheckCircle className="w-3.5 h-3.5" />
              )}
              {risk.level}
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-5">
          {/* Main Probability */}
          <div className="text-center space-y-3">
            <div className="relative inline-flex items-center justify-center">
              <svg className="w-28 h-28 transform -rotate-90">
                <circle
                  cx="56"
                  cy="56"
                  r="48"
                  fill="none"
                  stroke="hsl(var(--muted))"
                  strokeWidth="10"
                />
                <motion.circle
                  cx="56"
                  cy="56"
                  r="48"
                  fill="none"
                  stroke="hsl(var(--foreground))"
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeDasharray={`${2 * Math.PI * 48}`}
                  initial={{ strokeDashoffset: `${2 * Math.PI * 48}` }}
                  animate={{ strokeDashoffset: `${2 * Math.PI * 48 * (1 - result.fraud_probability)}` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-2xl font-semibold text-foreground">
                  {(result.fraud_probability * 100).toFixed(0)}%
                </span>
                <span className="text-xs text-muted-foreground">Fraud</span>
              </div>
            </div>
            
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Safe</span>
                <span>Fraudulent</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.fraud_probability * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className="h-full bg-foreground rounded-full"
                />
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="pt-3 border-t border-border space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Model</span>
              <span className="font-medium text-foreground">{result.model_used}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Threshold</span>
              <span className="font-medium text-foreground">{(result.threshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Individual Model Scores */}
          {(result.lgb_score !== undefined || result.xgb_score !== undefined || result.cb_score !== undefined) && (
            <div className="pt-3 border-t border-border space-y-3">
              <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-muted-foreground" />
                Model Scores
              </h4>
              {result.lgb_score !== undefined && (
                <ModelScoreBar label="LightGBM" score={result.lgb_score} />
              )}
              {result.xgb_score !== undefined && (
                <ModelScoreBar label="XGBoost" score={result.xgb_score} />
              )}
              {result.cb_score !== undefined && (
                <ModelScoreBar label="CatBoost" score={result.cb_score} />
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default function ResultsDisplay({ results, onNewAnalysis }: ResultsDisplayProps) {
  const avgProbability = results.reduce((sum, r) => sum + r.fraud_probability, 0) / results.length;
  const fraudCount = results.filter(r => r.is_fraud_predicted).length;
  const safeCount = results.length - fraudCount;

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-3"
      >
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="p-2 bg-muted rounded-md">
              <TrendingUp className="w-4 h-4 text-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Avg. Probability</p>
              <p className="text-lg font-semibold text-foreground">{(avgProbability * 100).toFixed(1)}%</p>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="p-2 bg-foreground/15 rounded-md">
              <AlertTriangle className="w-4 h-4 text-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Flagged</p>
              <p className="text-lg font-semibold text-foreground">{fraudCount}</p>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-card border-border">
          <CardContent className="p-4 flex items-center gap-3">
            <div className="p-2 bg-muted rounded-md">
              <Shield className="w-4 h-4 text-foreground" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Safe</p>
              <p className="text-lg font-semibold text-foreground">{safeCount}</p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Individual Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {results.map((result, index) => (
          <ResultCard key={index} result={result} index={index} />
        ))}
      </div>

      {/* Back Button */}
      <div className="pt-4">
        <Button
          variant="outline"
          onClick={onNewAnalysis}
          className="h-9 border-border text-foreground hover:bg-muted"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          New Analysis
        </Button>
      </div>
    </div>
  );
}
