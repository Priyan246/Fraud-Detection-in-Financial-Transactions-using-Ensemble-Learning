import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  ChevronDown,
  ChevronUp,
  Plus,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { PredictionResult } from '@/types/api';

interface ResultsDisplayProps {
  results: PredictionResult[];
  onNewAnalysis: () => void;
}

const statusConfig = {
  Safe: {
    icon: ShieldCheck,
    color: 'text-green-500',
    bg: 'bg-green-500/10',
    border: 'border-green-500/30',
    label: 'Safe',
    barColor: 'bg-green-500',
  },
  Uncertain: {
    icon: ShieldAlert,
    color: 'text-yellow-500',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    label: 'Uncertain',
    barColor: 'bg-yellow-500',
  },
  Fraud: {
    icon: ShieldX,
    color: 'text-red-500',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    label: 'Fraud Detected',
    barColor: 'bg-red-500',
  },
} as const;

type StatusKey = keyof typeof statusConfig;

/** Normalise whatever the API returns to a 0–1 float */
function normProb(raw: number): number {
  if (raw > 1) return raw / 100;
  return raw;
}

/** Derive a display status from the result, with safe fallbacks */
function deriveStatus(result: PredictionResult): StatusKey {
  if (result.status && result.status in statusConfig) {
    return result.status as StatusKey;
  }
  if (result.is_fraud_predicted) return 'Fraud';
  const p = normProb(result.fraud_probability);
  if (p >= 0.9) return 'Fraud';
  if (p >= 0.7) return 'Uncertain';
  return 'Safe';
}

function ProbabilityBar({ prob, rawProb, status }: { prob: number; rawProb: number; status: StatusKey }) {
  const pct = Math.min(100, Math.max(0, prob * 100));
  const { barColor } = statusConfig[status];

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className="text-xs text-muted-foreground">Fraud probability</span>
        <span className="text-sm font-mono font-semibold text-foreground">{rawProb}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${barColor}`}
          initial={{ width: '0%' }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}

function ModelScore({ label, score }: { label: string; score: number }) {
  const pct = Math.min(100, normProb(score) * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-foreground">{score}</span>
      </div>
      <div className="h-1 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-foreground/40"
          style={{ width: `${pct}%`, transition: 'width 0.5s ease' }}
        />
      </div>
    </div>
  );
}

function ResultCard({ result, index }: { result: PredictionResult; index: number }) {
  const [expanded, setExpanded] = useState(false);

  const status = deriveStatus(result);
  const prob = normProb(result.fraud_probability);
  const cfg = statusConfig[status];
  const Icon = cfg.icon;

  const hasBreakdown =
    result.lgb_score !== undefined ||
    result.xgb_score !== undefined ||
    result.cb_score !== undefined;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07, duration: 0.3 }}
      className={`rounded-xl border p-4 space-y-3 ${cfg.bg} ${cfg.border}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className={`p-1.5 rounded-lg ${cfg.bg}`}>
            <Icon className={`w-4 h-4 ${cfg.color}`} />
          </div>
          <div>
            <p className={`text-sm font-semibold ${cfg.color}`}>{cfg.label}</p>
            <p className="text-xs text-muted-foreground">Transaction #{index + 1}</p>
          </div>
        </div>
        <div className="text-right shrink-0">
          <p className="text-xs text-muted-foreground">Model</p>
          <p className="text-xs font-mono font-medium text-foreground">{result.model_used}</p>
        </div>
      </div>

      {/* THE BAR — always rendered, probability always shown */}
      <ProbabilityBar prob={prob} rawProb={result.fraud_probability} status={status} />

      {/* Footer */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Threshold: {result.threshold}</span>
        {hasBreakdown && (
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 hover:text-foreground transition-colors"
          >
            Per-model scores
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        )}
      </div>

      {/* Expandable breakdown */}
      {expanded && hasBreakdown && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="border-t border-border/50 pt-3 space-y-2"
        >
          {result.lgb_score !== undefined && <ModelScore label="LightGBM" score={result.lgb_score} />}
          {result.xgb_score !== undefined && <ModelScore label="XGBoost" score={result.xgb_score} />}
          {result.cb_score !== undefined && <ModelScore label="CatBoost" score={result.cb_score} />}
        </motion.div>
      )}
    </motion.div>
  );
}

export default function ResultsDisplay({ results, onNewAnalysis }: ResultsDisplayProps) {
  const fraudCount   = results.filter(r => deriveStatus(r) === 'Fraud').length;
  const uncertainCount = results.filter(r => deriveStatus(r) === 'Uncertain').length;
  const safeCount    = results.filter(r => deriveStatus(r) === 'Safe').length;

  return (
    <div className="space-y-6">
      {/* Summary strip */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Safe',      count: safeCount,      color: 'text-green-500',  bg: 'bg-green-500/10'  },
          { label: 'Uncertain', count: uncertainCount, color: 'text-yellow-500', bg: 'bg-yellow-500/10' },
          { label: 'Fraud',     count: fraudCount,     color: 'text-red-500',    bg: 'bg-red-500/10'    },
        ].map(({ label, count, color, bg }) => (
          <div key={label} className={`rounded-xl p-4 text-center ${bg}`}>
            <p className={`text-2xl font-bold ${color}`}>{count}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
          </div>
        ))}
      </div>

      {/* Result cards */}
      <div className="space-y-3">
        {results.map((result, i) => (
          <ResultCard key={i} result={result} index={i} />
        ))}
      </div>

      <Button onClick={onNewAnalysis} variant="outline" className="w-full border-border">
        <Plus className="w-4 h-4 mr-2" />
        Analyze New Transactions
      </Button>
    </div>
  );
}
