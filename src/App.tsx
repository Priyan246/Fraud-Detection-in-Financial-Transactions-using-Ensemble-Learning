import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'sonner';
import {
  Shield,
  Settings,
  BarChart3,
  CreditCard,
  Moon,
  Sun
} from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import TransactionForm from '@/sections/TransactionForm';
import ResultsDisplay from '@/sections/ResultsDisplay';
import ApiStatus from '@/sections/ApiStatus';
import BackgroundAnimation from '@/components/BackgroundAnimation';
import { apiService } from '@/services/api';
import type { Transaction, PredictionResult, ModelType } from '@/types/api - Copy';

function App() {
  const [results, setResults] = useState<PredictionResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('detect');
  const [isDark, setIsDark] = useState(true);

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  const handleSubmit = async (transactions: Transaction[], model: ModelType, returnAll: boolean) => {
    try {
      setIsLoading(true);
      const response = await apiService.predict(transactions, model, returnAll);
      setResults(response.predictions);
      setActiveTab('results');

      const fraudCount = response.predictions.filter(r => r.is_fraud_predicted).length;
      if (fraudCount > 0) {
        toast.warning(`${fraudCount} transaction(s) flagged as potentially fraudulent`);
      } else {
        toast.success('All transactions appear safe');
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to analyze transactions');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetHistory = () => {
    toast.success('Card history has been reset');
  };

  return (
    <div className={`min-h-screen bg-background text-foreground transition-colors duration-300 ${isDark ? 'dark' : ''}`}>
      <Toaster
        position="top-right"
        theme={isDark ? 'dark' : 'light'}
        toastOptions={{
          style: {
            background: isDark ? 'hsl(0 0% 8%)' : 'hsl(0 0% 100%)',
            color: isDark ? 'hsl(0 0% 100%)' : 'hsl(0 0% 0%)',
            border: `1px solid ${isDark ? 'hsl(0 0% 25%)' : 'hsl(0 0% 80%)'}`,
          },
        }}
      />

      <BackgroundAnimation />

      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/95 backdrop-blur border-b border-border">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-foreground rounded-md flex items-center justify-center">
                <Shield className="w-4 h-4 text-background" />
              </div>
              <h1 className="text-lg font-semibold text-foreground">Fraud Detection</h1>
            </div>

            <Button
              variant="outline"
              size="icon"
              onClick={toggleTheme}
              className="rounded-full border-border"
            >
              {isDark ? (
                <Sun className="w-4 h-4" />
              ) : (
                <Moon className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-card border border-border p-1 rounded-lg w-full justify-start">
            <TabsTrigger
              value="detect"
              className="data-[state=active]:bg-foreground data-[state=active]:text-background rounded-md px-4 py-2 text-sm text-foreground"
            >
              <CreditCard className="w-4 h-4 mr-2" />
              Detect
            </TabsTrigger>
            <TabsTrigger
              value="results"
              disabled={!results}
              className="data-[state=active]:bg-foreground data-[state=active]:text-background rounded-md px-4 py-2 text-sm text-foreground"
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              Results
              {results && (
                <span className="ml-2 px-1.5 py-0.5 bg-muted rounded text-xs text-foreground">
                  {results.length}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger
              value="status"
              className="data-[state=active]:bg-foreground data-[state=active]:text-background rounded-md px-4 py-2 text-sm text-foreground"
            >
              <Settings className="w-4 h-4 mr-2" />
              Status
            </TabsTrigger>
          </TabsList>

          <AnimatePresence mode="wait">
            <TabsContent value="detect" className="space-y-6 mt-6">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <TransactionForm onSubmit={handleSubmit} isLoading={isLoading} />
              </motion.div>
            </TabsContent>

            <TabsContent value="results" className="space-y-6 mt-6">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {results ? (
                  <ResultsDisplay results={results} onNewAnalysis={() => setActiveTab('detect')} />
                ) : (
                  <div className="text-center py-20 text-muted-foreground">
                    <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No results yet. Submit a transaction to analyze.</p>
                  </div>
                )}
              </motion.div>
            </TabsContent>

            <TabsContent value="status" className="space-y-6 mt-6">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <ApiStatus onResetHistory={handleResetHistory} />
              </motion.div>
            </TabsContent>
          </AnimatePresence>
        </Tabs>
      </main>
    </div>
  );
}

export default App;
