import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CreditCard, 
  Store, 
  MapPin, 
  Briefcase,
  Clock,
  Navigation,
  Send,
  Loader2,
  Plus,
  Trash2,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent } from '@/components/ui/card';
import type { Transaction, TransactionFormData, ModelType } from '@/types/api';

interface TransactionFormProps {
  onSubmit: (transactions: Transaction[], model: ModelType, returnAll: boolean) => void;
  isLoading: boolean;
}

const initialFormData: TransactionFormData = {
  trans_date_trans_time: new Date().toISOString().slice(0, 16),
  cc_num: '',
  merchant: '',
  category: 'grocery_pos',
  amt: '',
  gender: 'M',
  city: '',
  state: '',
  zip: '',
  lat: '',
  long: '',
  city_pop: '',
  job: '',
  dob: '1985-04-12',
  unix_time: Math.floor(Date.now() / 1000).toString(),
  merch_lat: '',
  merch_long: '',
};

const categories = [
  'grocery_pos', 'gas_transport', 'home', 'shopping_pos', 
  'kids_pets', 'entertainment', 'food_dining', 'personal_care',
  'health_fitness', 'misc_pos', 'misc_net', 'grocery_net',
  'shopping_net', 'travel', 'utilities'
];

const states = [
  'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
];

const FormSection = ({ 
  title, 
  icon: Icon, 
  children,
  defaultOpen = false
}: { 
  title: string; 
  icon: any; 
  children: React.ReactNode;
  defaultOpen?: boolean;
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  return (
    <div className="border border-border rounded-lg overflow-hidden bg-card">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3 bg-muted hover:bg-muted/80 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const FormField = ({ 
  label, 
  children, 
  required = false 
}: { 
  label: string; 
  children: React.ReactNode;
  required?: boolean;
}) => (
  <div className="space-y-1.5">
    <Label className="text-xs text-muted-foreground flex items-center gap-1">
      {label}
      {required && <span className="text-foreground">*</span>}
    </Label>
    {children}
  </div>
);

export default function TransactionForm({ onSubmit, isLoading }: TransactionFormProps) {
  const [transactions, setTransactions] = useState<TransactionFormData[]>([{ ...initialFormData }]);
  const [model, setModel] = useState<ModelType>('ensemble');
  const [returnAll, setReturnAll] = useState(false);

  const addTransaction = () => {
    setTransactions([...transactions, { ...initialFormData, cc_num: transactions[0]?.cc_num || '' }]);
  };

  const removeTransaction = (index: number) => {
    if (transactions.length > 1) {
      setTransactions(transactions.filter((_, i) => i !== index));
    }
  };

  const updateTransaction = (index: number, field: keyof TransactionFormData, value: string) => {
    const updated = [...transactions];
    updated[index] = { ...updated[index], [field]: value };
    setTransactions(updated);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const formattedTransactions: Transaction[] = transactions.map(t => ({
      trans_date_trans_time: t.trans_date_trans_time,
      cc_num: t.cc_num,
      merchant: t.merchant,
      category: t.category,
      amt: parseFloat(t.amt) || 0,
      gender: t.gender,
      city: t.city,
      state: t.state,
      zip: parseInt(t.zip) || 0,
      lat: parseFloat(t.lat) || 0,
      long: parseFloat(t.long) || 0,
      city_pop: parseInt(t.city_pop) || 0,
      job: t.job,
      dob: t.dob,
      unix_time: parseInt(t.unix_time) || Math.floor(Date.now() / 1000),
      merch_lat: parseFloat(t.merch_lat) || 0,
      merch_long: parseFloat(t.merch_long) || 0,
    }));
    onSubmit(formattedTransactions, model, returnAll);
  };

  const fillSampleData = (index: number) => {
    const sampleData: Partial<TransactionFormData> = {
      cc_num: '4111111111111111',
      merchant: 'fraud_Sample Merchant Inc',
      category: 'grocery_pos',
      amt: '42.50',
      city: 'Springfield',
      state: 'IL',
      zip: '62701',
      lat: '39.7817',
      long: '-89.6501',
      city_pop: '116250',
      job: 'Teacher',
      merch_lat: '39.80',
      merch_long: '-89.66',
    };
    const updated = [...transactions];
    updated[index] = { ...updated[index], ...sampleData };
    setTransactions(updated);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {transactions.map((transaction, index) => (
        <Card key={index} className="bg-card border-border">
          <CardContent className="p-4 space-y-4">
            {transactions.length > 1 && (
              <div className="flex items-center justify-between pb-3 border-b border-border">
                <span className="text-sm font-medium text-foreground">Transaction #{index + 1}</span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => removeTransaction(index)}
                  className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            )}

            <FormSection title="Card Information" icon={CreditCard} defaultOpen={true}>
              <FormField label="Card Number" required>
                <Input
                  value={transaction.cc_num}
                  onChange={(e) => updateTransaction(index, 'cc_num', e.target.value)}
                  placeholder="4111111111111111"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Amount ($)" required>
                <Input
                  type="number"
                  step="0.01"
                  value={transaction.amt}
                  onChange={(e) => updateTransaction(index, 'amt', e.target.value)}
                  placeholder="42.50"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Gender">
                <Select 
                  value={transaction.gender} 
                  onValueChange={(v) => updateTransaction(index, 'gender', v as 'M' | 'F')}
                >
                  <SelectTrigger className="h-9 bg-background border-border text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="M" className="text-foreground">Male</SelectItem>
                    <SelectItem value="F" className="text-foreground">Female</SelectItem>
                  </SelectContent>
                </Select>
              </FormField>
              <FormField label="Date of Birth">
                <Input
                  type="date"
                  value={transaction.dob}
                  onChange={(e) => updateTransaction(index, 'dob', e.target.value)}
                  className="h-9 bg-background border-border text-foreground"
                />
              </FormField>
            </FormSection>

            <FormSection title="Merchant Information" icon={Store}>
              <FormField label="Merchant Name" required>
                <Input
                  value={transaction.merchant}
                  onChange={(e) => updateTransaction(index, 'merchant', e.target.value)}
                  placeholder="fraud_Sample Merchant Inc"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Category">
                <Select 
                  value={transaction.category} 
                  onValueChange={(v) => updateTransaction(index, 'category', v)}
                >
                  <SelectTrigger className="h-9 bg-background border-border text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    {categories.map(cat => (
                      <SelectItem key={cat} value={cat} className="text-foreground">{cat}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FormField>
              <FormField label="Transaction Time">
                <Input
                  type="datetime-local"
                  value={transaction.trans_date_trans_time}
                  onChange={(e) => updateTransaction(index, 'trans_date_trans_time', e.target.value)}
                  className="h-9 bg-background border-border text-foreground"
                />
              </FormField>
            </FormSection>

            <FormSection title="Customer Location" icon={MapPin}>
              <FormField label="City">
                <Input
                  value={transaction.city}
                  onChange={(e) => updateTransaction(index, 'city', e.target.value)}
                  placeholder="Springfield"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="State">
                <Select 
                  value={transaction.state} 
                  onValueChange={(v) => updateTransaction(index, 'state', v)}
                >
                  <SelectTrigger className="h-9 bg-background border-border text-foreground">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    {states.map(s => (
                      <SelectItem key={s} value={s} className="text-foreground">{s}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FormField>
              <FormField label="ZIP Code">
                <Input
                  value={transaction.zip}
                  onChange={(e) => updateTransaction(index, 'zip', e.target.value)}
                  placeholder="62701"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Latitude">
                <Input
                  type="number"
                  step="0.0001"
                  value={transaction.lat}
                  onChange={(e) => updateTransaction(index, 'lat', e.target.value)}
                  placeholder="39.7817"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Longitude">
                <Input
                  type="number"
                  step="0.0001"
                  value={transaction.long}
                  onChange={(e) => updateTransaction(index, 'long', e.target.value)}
                  placeholder="-89.6501"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="City Population">
                <Input
                  type="number"
                  value={transaction.city_pop}
                  onChange={(e) => updateTransaction(index, 'city_pop', e.target.value)}
                  placeholder="116250"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
            </FormSection>

            <FormSection title="Merchant Location" icon={Navigation}>
              <FormField label="Merchant Latitude">
                <Input
                  type="number"
                  step="0.0001"
                  value={transaction.merch_lat}
                  onChange={(e) => updateTransaction(index, 'merch_lat', e.target.value)}
                  placeholder="39.80"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Merchant Longitude">
                <Input
                  type="number"
                  step="0.0001"
                  value={transaction.merch_long}
                  onChange={(e) => updateTransaction(index, 'merch_long', e.target.value)}
                  placeholder="-89.66"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
            </FormSection>

            <FormSection title="Additional Info" icon={Briefcase}>
              <FormField label="Job">
                <Input
                  value={transaction.job}
                  onChange={(e) => updateTransaction(index, 'job', e.target.value)}
                  placeholder="Teacher"
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
              <FormField label="Unix Timestamp">
                <Input
                  type="number"
                  value={transaction.unix_time}
                  onChange={(e) => updateTransaction(index, 'unix_time', e.target.value)}
                  placeholder={Math.floor(Date.now() / 1000).toString()}
                  className="h-9 bg-background border-border text-foreground placeholder:text-muted-foreground"
                />
              </FormField>
            </FormSection>

            <div className="flex justify-end">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fillSampleData(index)}
                className="border-border text-foreground hover:bg-muted"
              >
                Fill Sample Data
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}

      {/* Model Selection */}
      <Card className="bg-card border-border">
        <CardContent className="p-4 space-y-4">
          <div className="flex items-center gap-2 text-sm font-medium text-foreground">
            <Clock className="w-4 h-4 text-muted-foreground" />
            Model Configuration
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label className="text-xs text-muted-foreground">Prediction Model</Label>
              <Select value={model} onValueChange={(v) => setModel(v as ModelType)}>
                <SelectTrigger className="h-9 bg-background border-border text-foreground">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-card border-border">
                  <SelectItem value="ensemble" className="text-foreground">Ensemble (Recommended)</SelectItem>
                  <SelectItem value="LightGBM" className="text-foreground">LightGBM</SelectItem>
                  <SelectItem value="XGBoost" className="text-foreground">XGBoost</SelectItem>
                  <SelectItem value="CatBoost" className="text-foreground">CatBoost</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-3 cursor-pointer">
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={returnAll}
                    onChange={(e) => setReturnAll(e.target.checked)}
                    className="sr-only"
                  />
                  <div className={`w-10 h-5 rounded-full transition-colors ${returnAll ? 'bg-foreground' : 'bg-muted'}`}>
                    <div className={`w-4 h-4 rounded-full bg-background shadow transition-transform ${returnAll ? 'translate-x-5' : 'translate-x-0.5'} mt-0.5`} />
                  </div>
                </div>
                <span className="text-sm text-muted-foreground">
                  Show all model scores
                </span>
              </label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3">
        <Button
          type="button"
          variant="outline"
          onClick={addTransaction}
          className="h-9 border-border text-foreground hover:bg-muted"
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Transaction
        </Button>
        
        <Button
          type="submit"
          disabled={isLoading}
          className="h-9 bg-foreground text-background hover:bg-foreground/90"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Send className="w-4 h-4 mr-2" />
              Detect Fraud
            </>
          )}
        </Button>
      </div>
    </form>
  );
}
