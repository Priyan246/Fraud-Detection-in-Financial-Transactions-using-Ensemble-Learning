import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Server, 
  Activity, 
  Database, 
  RefreshCw, 
  XCircle,
  Cpu,
  Layers,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { apiService } from '@/services/api';
import type { HealthStatus, ModelMetadata } from '@/types/api';

interface ApiStatusProps {
  onResetHistory: () => void;
}

const StatusIndicator = ({ isOnline }: { isOnline: boolean }) => (
  <motion.div
    initial={{ scale: 0 }}
    animate={{ scale: 1 }}
    className={`w-2.5 h-2.5 rounded-full ${isOnline ? 'bg-foreground' : 'bg-muted-foreground'}`}
  >
    {isOnline && (
      <motion.div
        className="w-full h-full rounded-full bg-foreground"
        animate={{ scale: [1, 1.5, 1], opacity: [1, 0, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
    )}
  </motion.div>
);

const ModelBadge = ({ name }: { name: string }) => (
  <div className="flex items-center gap-1.5 px-2.5 py-1 bg-muted rounded-md text-xs font-medium text-foreground">
    <Cpu className="w-3.5 h-3.5" />
    {name}
  </div>
);

export default function ApiStatus({ onResetHistory }: ApiStatusProps) {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      const [healthData, metadataData] = await Promise.all([
        apiService.getHealth(),
        apiService.getMetadata(),
      ]);
      setHealth(healthData);
      setMetadata(metadataData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to API');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      setResetting(true);
      await apiService.resetCardHistory();
      onResetHistory();
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset history');
    } finally {
      setResetting(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const isOnline = health?.status === 'ok';

  return (
    <div className="space-y-4">
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base font-medium text-foreground flex items-center gap-2">
              <Server className="w-4 h-4 text-muted-foreground" />
              API Status
            </CardTitle>
            <div className="flex items-center gap-2">
              <StatusIndicator isOnline={isOnline} />
              <span className={`text-xs font-medium text-foreground`}>
                {isOnline ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center justify-center py-8"
              >
                <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
              </motion.div>
            ) : error ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-3 py-8"
              >
                <XCircle className="w-10 h-10 text-muted-foreground" />
                <p className="text-sm text-muted-foreground text-center">{error}</p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={fetchStatus}
                  className="border-border text-foreground hover:bg-muted"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Retry
                </Button>
              </motion.div>
            ) : health && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-5"
              >
                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-muted/50 rounded-md p-3 text-center">
                    <Layers className="w-4 h-4 mx-auto mb-1.5 text-muted-foreground" />
                    <p className="text-xl font-semibold text-foreground">{health.n_features}</p>
                    <p className="text-xs text-muted-foreground">Features</p>
                  </div>
                  <div className="bg-muted/50 rounded-md p-3 text-center">
                    <Database className="w-4 h-4 mx-auto mb-1.5 text-muted-foreground" />
                    <p className="text-xl font-semibold text-foreground">{health.card_registry_size}</p>
                    <p className="text-xs text-muted-foreground">Registry</p>
                  </div>
                  <div className="bg-muted/50 rounded-md p-3 text-center">
                    <Zap className="w-4 h-4 mx-auto mb-1.5 text-muted-foreground" />
                    <p className="text-xl font-semibold text-foreground">{health.models_loaded.length}</p>
                    <p className="text-xs text-muted-foreground">Models</p>
                  </div>
                  <div className="bg-muted/50 rounded-md p-3 text-center">
                    <Activity className="w-4 h-4 mx-auto mb-1.5 text-muted-foreground" />
                    <p className="text-xl font-semibold text-foreground">Active</p>
                    <p className="text-xs text-muted-foreground">Status</p>
                  </div>
                </div>

                {/* Loaded Models */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-foreground">Loaded Models</h4>
                  <div className="flex flex-wrap gap-2">
                    {health.models_loaded.map((model, index) => (
                      <motion.div
                        key={model}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: index * 0.05 }}
                      >
                        <ModelBadge name={model} />
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Thresholds */}
                {metadata && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-foreground">Thresholds</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {Object.entries(metadata.thresholds).map(([model, threshold]) => (
                        <div key={model} className="bg-muted/50 rounded-md p-2.5">
                          <p className="text-xs text-muted-foreground mb-0.5">{model}</p>
                          <p className="text-base font-semibold text-foreground">{(threshold * 100).toFixed(0)}%</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="pt-3 border-t border-border flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={fetchStatus}
                    className="border-border text-foreground hover:bg-muted"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleReset}
                    disabled={resetting}
                    className="border-border text-foreground hover:bg-muted"
                  >
                    {resetting ? (
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Database className="w-4 h-4 mr-2" />
                    )}
                    Reset History
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </div>
  );
}
