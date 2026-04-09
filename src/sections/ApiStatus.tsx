import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, Trash2, Cpu, Database, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { apiService } from '@/services/api';
import type { HealthResponse } from '@/types/api';

interface ApiStatusProps {
  onResetHistory: () => void;
}

export default function ApiStatus({ onResetHistory }: ApiStatusProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchHealth() {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.health();
      setHealth(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not reach API');
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    setResetting(true);
    try {
      await apiService.resetCardHistory();
      toast.success('Card history has been reset');
      onResetHistory();
      fetchHealth();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Reset failed');
    } finally {
      setResetting(false);
    }
  }

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-foreground">API Status</h2>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchHealth}
          disabled={loading}
          className="border-border text-xs"
        >
          <RefreshCw className={`w-3 h-3 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-500">
          {error}
        </div>
      ) : health ? (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-3"
        >
          {/* Status Badge */}
          <div className="flex items-center gap-2 rounded-xl border border-green-500/30 bg-green-500/10 p-4">
            <Activity className="w-4 h-4 text-green-500" />
            <span className="text-sm font-medium text-green-500 capitalize">{health.status}</span>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-border bg-card p-4 space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Cpu className="w-3.5 h-3.5" />
                <span className="text-xs">Features</span>
              </div>
              <p className="text-xl font-semibold text-foreground">{health.n_features}</p>
            </div>
            <div className="rounded-xl border border-border bg-card p-4 space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Database className="w-3.5 h-3.5" />
                <span className="text-xs">Card Registry</span>
              </div>
              <p className="text-xl font-semibold text-foreground">{health.card_registry_size}</p>
            </div>
          </div>

          {/* Models */}
          <div className="rounded-xl border border-border bg-card p-4 space-y-2">
            <p className="text-xs text-muted-foreground">Loaded models</p>
            <div className="flex flex-wrap gap-2">
              {health.models_loaded.map(m => (
                <span
                  key={m}
                  className="px-2.5 py-1 rounded-md bg-foreground/10 text-foreground text-xs font-mono"
                >
                  {m}
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      ) : (
        <div className="rounded-xl border border-border bg-card p-8 text-center text-muted-foreground text-sm">
          Loading...
        </div>
      )}

      {/* Reset Card History */}
      <div className="rounded-xl border border-border bg-card p-4 space-y-3">
        <div>
          <p className="text-sm font-medium text-foreground">Reset Card History</p>
          <p className="text-xs text-muted-foreground mt-0.5">
            Clears in-memory rolling features for all cards. Use when starting a fresh test session.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleReset}
          disabled={resetting}
          className="border-red-500/30 text-red-500 hover:bg-red-500/10 text-xs"
        >
          <Trash2 className="w-3 h-3 mr-1.5" />
          {resetting ? 'Resetting…' : 'Reset History'}
        </Button>
      </div>
    </div>
  );
}
