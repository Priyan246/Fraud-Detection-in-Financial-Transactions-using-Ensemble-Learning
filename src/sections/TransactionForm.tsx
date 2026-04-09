import { useState } from 'react';
import { Plus, Trash2, Send, ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { Transaction, ModelType } from '@/types/api';

interface TransactionFormProps {
  onSubmit: (transactions: Transaction[], model: ModelType, returnAll: boolean) => void;
  isLoading: boolean;
}

const CATEGORIES = [
  'grocery_pos', 'entertainment', 'gas_transport', 'misc_net', 'grocery_net',
  'shopping_net', 'shopping_pos', 'misc_pos', 'food_dining', 'personal_care',
  'health_fitness', 'travel', 'kids_pets', 'home',
];

const US_STATES = [
  'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
  'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
  'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
  'VA','WA','WV','WI','WY',
];

function makeBlankTransaction(): Transaction {
  const now = new Date();
  return {
    trans_date_trans_time: now.toISOString().slice(0, 19).replace('T', ' '),
    cc_num: '',
    merchant: '',
    category: 'shopping_pos',
    amt: 0,
    gender: 'M',
    city: '',
    state: 'CA',
    zip: '',
    lat: 0,
    long: 0,
    city_pop: 50000,
    job: '',
    dob: '1990-01-01',
    unix_time: Math.floor(now.getTime() / 1000),
    merch_lat: 0,
    merch_long: 0,
  };
}

function TxField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

function TransactionRow({
  tx,
  index,
  onChange,
  onRemove,
  canRemove,
}: {
  tx: Transaction;
  index: number;
  onChange: (updated: Transaction) => void;
  onRemove: () => void;
  canRemove: boolean;
}) {
  const [expanded, setExpanded] = useState(index === 0);

  function set<K extends keyof Transaction>(key: K, value: Transaction[K]) {
    onChange({ ...tx, [key]: value });
  }

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-muted/50 transition-colors"
      >
        <span className="text-sm font-medium text-foreground">
          Transaction #{index + 1}
          {tx.amt > 0 && (
            <span className="ml-2 text-xs text-muted-foreground font-normal">
              ${tx.amt} · {tx.merchant || 'unnamed'}
            </span>
          )}
        </span>
        <div className="flex items-center gap-2">
          {canRemove && (
            <span
              role="button"
              onClick={(e) => { e.stopPropagation(); onRemove(); }}
              className="p-1 rounded hover:bg-red-500/10 text-muted-foreground hover:text-red-500 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </span>
          )}
          {expanded ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 border-t border-border">
          <div className="grid grid-cols-2 gap-3 mt-3">
            <TxField label="Date & Time">
              <Input
                type="datetime-local"
                value={tx.trans_date_trans_time.replace(' ', 'T')}
                onChange={e => {
                  const val = e.target.value.replace('T', ' ');
                  const unix = Math.floor(new Date(e.target.value).getTime() / 1000);
                  onChange({ ...tx, trans_date_trans_time: val, unix_time: unix });
                }}
                className="text-xs h-8"
              />
            </TxField>

            <TxField label="Amount ($)">
              <Input
                type="number"
                min={0}
                step={0.01}
                value={tx.amt}
                onChange={e => set('amt', parseFloat(e.target.value) || 0)}
                className="text-xs h-8"
                placeholder="0.00"
              />
            </TxField>

            <TxField label="Card Number">
              <Input
                value={tx.cc_num}
                onChange={e => set('cc_num', e.target.value)}
                className="text-xs h-8"
                placeholder="1234567890123456"
              />
            </TxField>

            <TxField label="Merchant">
              <Input
                value={tx.merchant}
                onChange={e => set('merchant', e.target.value)}
                className="text-xs h-8"
                placeholder="fraud_Store Name"
              />
            </TxField>

            <TxField label="Category">
              <Select value={tx.category} onValueChange={v => set('category', v)}>
                <SelectTrigger className="text-xs h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {CATEGORIES.map(c => (
                    <SelectItem key={c} value={c} className="text-xs">{c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </TxField>

            <TxField label="Gender">
              <Select value={tx.gender} onValueChange={v => set('gender', v)}>
                <SelectTrigger className="text-xs h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="M" className="text-xs">Male</SelectItem>
                  <SelectItem value="F" className="text-xs">Female</SelectItem>
                </SelectContent>
              </Select>
            </TxField>

            <TxField label="City">
              <Input
                value={tx.city}
                onChange={e => set('city', e.target.value)}
                className="text-xs h-8"
                placeholder="San Francisco"
              />
            </TxField>

            <TxField label="State">
              <Select value={tx.state} onValueChange={v => set('state', v)}>
                <SelectTrigger className="text-xs h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {US_STATES.map(s => (
                    <SelectItem key={s} value={s} className="text-xs">{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </TxField>

            <TxField label="ZIP">
              <Input
                value={tx.zip}
                onChange={e => set('zip', e.target.value)}
                className="text-xs h-8"
                placeholder="94103"
              />
            </TxField>

            <TxField label="City Population">
              <Input
                type="number"
                value={tx.city_pop}
                onChange={e => set('city_pop', parseInt(e.target.value) || 0)}
                className="text-xs h-8"
              />
            </TxField>

            <TxField label="Date of Birth">
              <Input
                type="date"
                value={tx.dob}
                onChange={e => set('dob', e.target.value)}
                className="text-xs h-8"
              />
            </TxField>

            <TxField label="Job">
              <Input
                value={tx.job}
                onChange={e => set('job', e.target.value)}
                className="text-xs h-8"
                placeholder="Software Engineer"
              />
            </TxField>
          </div>

          {/* Location section */}
          <p className="text-xs font-medium text-muted-foreground mt-4 mb-2">Cardholder Location</p>
          <div className="grid grid-cols-2 gap-3">
            <TxField label="Latitude">
              <Input
                type="number"
                step="any"
                value={tx.lat}
                onChange={e => set('lat', parseFloat(e.target.value) || 0)}
                className="text-xs h-8"
              />
            </TxField>
            <TxField label="Longitude">
              <Input
                type="number"
                step="any"
                value={tx.long}
                onChange={e => set('long', parseFloat(e.target.value) || 0)}
                className="text-xs h-8"
              />
            </TxField>
          </div>

          <p className="text-xs font-medium text-muted-foreground mt-4 mb-2">Merchant Location</p>
          <div className="grid grid-cols-2 gap-3">
            <TxField label="Merchant Latitude">
              <Input
                type="number"
                step="any"
                value={tx.merch_lat}
                onChange={e => set('merch_lat', parseFloat(e.target.value) || 0)}
                className="text-xs h-8"
              />
            </TxField>
            <TxField label="Merchant Longitude">
              <Input
                type="number"
                step="any"
                value={tx.merch_long}
                onChange={e => set('merch_long', parseFloat(e.target.value) || 0)}
                className="text-xs h-8"
              />
            </TxField>
          </div>
        </div>
      )}
    </div>
  );
}

export default function TransactionForm({ onSubmit, isLoading }: TransactionFormProps) {
  const [transactions, setTransactions] = useState<Transaction[]>([makeBlankTransaction()]);
  const [model, setModel] = useState<ModelType>('ensemble');
  const [returnAll, setReturnAll] = useState(false);

  function addTransaction() {
    setTransactions(prev => [...prev, makeBlankTransaction()]);
  }

  function removeTransaction(index: number) {
    setTransactions(prev => prev.filter((_, i) => i !== index));
  }

  function updateTransaction(index: number, updated: Transaction) {
    setTransactions(prev => prev.map((t, i) => (i === index ? updated : t)));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit(transactions, model, returnAll);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Model selector */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card p-4">
        <div className="flex-1 min-w-[160px] space-y-1.5">
          <Label className="text-xs text-muted-foreground">Model</Label>
          <Select value={model} onValueChange={v => setModel(v as ModelType)}>
            <SelectTrigger className="text-xs h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ensemble" className="text-xs">Ensemble (all models)</SelectItem>
              <SelectItem value="LightGBM" className="text-xs">LightGBM</SelectItem>
              <SelectItem value="XGBoost" className="text-xs">XGBoost</SelectItem>
              <SelectItem value="CatBoost" className="text-xs">CatBoost</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <label className="flex items-center gap-2 cursor-pointer select-none text-xs text-muted-foreground mt-4">
          <input
            type="checkbox"
            checked={returnAll}
            onChange={e => setReturnAll(e.target.checked)}
            className="rounded"
          />
          Return all model scores
        </label>
      </div>

      {/* Transaction rows */}
      <div className="space-y-3">
        {transactions.map((tx, i) => (
          <TransactionRow
            key={i}
            tx={tx}
            index={i}
            onChange={updated => updateTransaction(i, updated)}
            onRemove={() => removeTransaction(i)}
            canRemove={transactions.length > 1}
          />
        ))}
      </div>

      <div className="flex gap-3">
        <Button
          type="button"
          variant="outline"
          onClick={addTransaction}
          className="flex-1 border-border text-sm"
        >
          <Plus className="w-4 h-4 mr-2" />
          Add Transaction
        </Button>

        <Button
          type="submit"
          disabled={isLoading}
          className="flex-1 bg-foreground text-background hover:bg-foreground/90 text-sm"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Analyzing…
            </>
          ) : (
            <>
              <Send className="w-4 h-4 mr-2" />
              Analyze {transactions.length > 1 ? `${transactions.length} Transactions` : 'Transaction'}
            </>
          )}
        </Button>
      </div>
    </form>
  );
}
