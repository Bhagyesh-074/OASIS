import React from 'react';
import { Cpu, DollarSign, Clock, Hash, CheckCircle2, AlertTriangle } from 'lucide-react';
import { RunDetail } from '../types';

interface BudgetGaugesProps {
  run: RunDetail;
}

export const BudgetGauges: React.FC<BudgetGaugesProps> = ({ run }) => {
  if (!run.budget) {
    return (
      <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 text-sm text-slate-400">
        Budget limits are unavailable for this run.
      </div>
    );
  }
  const budget = run.budget;
  const totals = run.totals;
  const compliance = run.compliance;

  const tokenPct = budget.max_tokens > 0 ? Math.min(100, (totals.tokens / budget.max_tokens) * 100) : 0;
  const costPct = budget.max_cost_usd > 0 ? Math.min(100, (totals.cost_usd / budget.max_cost_usd) * 100) : 0;
  const wallPct = budget.max_wall_seconds > 0 ? Math.min(100, (totals.wall_seconds / budget.max_wall_seconds) * 100) : 0;
  const callsPct = budget.max_calls > 0 ? Math.min(100, (totals.calls / budget.max_calls) * 100) : 0;

  // Identify binding constraint
  const percentages = [
    { name: 'Tokens', pct: tokenPct },
    { name: 'Cost', pct: costPct },
    { name: 'Wall Time', pct: wallPct },
    { name: 'Calls', pct: callsPct },
  ];
  const maxPct = Math.max(...percentages.map((p) => p.pct));
  const bindingConstraint = percentages.find((p) => p.pct === maxPct && maxPct > 0)?.name;

  const getProgressColor = (pct: number) => {
    if (pct >= 100) return 'bg-rose-500 shadow-rose-500/50 shadow-sm';
    if (pct >= 80) return 'bg-amber-400 shadow-amber-400/50 shadow-sm';
    return 'bg-sky-400';
  };

  return (
    <section className="rounded-2xl border border-white/[0.07] bg-gradient-to-br from-[#111a29] to-[#0d131e] p-5 shadow-[0_20px_60px_-50px_rgba(16,185,129,0.25)] sm:p-6">
      <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold tracking-tight text-white sm:text-lg">
              Resource guardrails
            </h2>
            <span className="rounded border border-white/[0.07] bg-white/[0.03] px-2 py-1 text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500">
              4 dimensions
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            Observed consumption against this run&apos;s configured limits
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {bindingConstraint && (
            <div className="rounded-lg border border-amber-300/10 bg-amber-300/[0.05] px-2.5 py-1.5 text-[10px] text-slate-400">
              Highest use <span className="ml-1 font-semibold text-amber-200">{bindingConstraint}</span>
            </div>
          )}
          {compliance.all ? (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-300/15 bg-emerald-300/[0.06] px-2.5 py-1.5 text-[10px] font-semibold text-emerald-200">
              <CheckCircle2 className="h-3.5 w-3.5" /> Within limits
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-rose-300/15 bg-rose-300/[0.06] px-2.5 py-1.5 text-[10px] font-semibold text-rose-200">
              <AlertTriangle className="h-3.5 w-3.5" /> Limit exceeded
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {/* Dimension 1: Tokens */}
        <div className="flex flex-col justify-between rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-sky-400" /> Tokens
            </span>
            <span className="text-xs font-mono">{tokenPct.toFixed(1)}%</span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">
            {totals.tokens.toLocaleString()}{' '}
            <span className="text-xs font-normal text-slate-500">/ {budget.max_tokens.toLocaleString()}</span>
          </div>
          <div
            className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-800"
            role="progressbar"
            aria-label="Token budget used"
            aria-valuenow={Math.round(tokenPct)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(tokenPct)}`} style={{ width: `${tokenPct}%` }} />
          </div>
          <div className="mt-2 text-[10px] text-slate-500">
            {totals.supervision_tokens.toLocaleString()} supervision tokens · {(totals.supervision_share * 100).toFixed(1)}%
          </div>
        </div>

        {/* Dimension 2: Cost USD */}
        <div className="flex flex-col justify-between rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider flex items-center gap-1.5">
              <DollarSign className="w-4 h-4 text-emerald-400" /> Cost (USD)
            </span>
            <span className="text-xs font-mono">{costPct.toFixed(1)}%</span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">
            ${totals.cost_usd.toFixed(4)}{' '}
            <span className="text-xs font-normal text-slate-500">/ ${budget.max_cost_usd.toFixed(2)}</span>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-800" role="progressbar" aria-label="Cost budget used" aria-valuenow={Math.round(costPct)} aria-valuemin={0} aria-valuemax={100}>
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(costPct)}`} style={{ width: `${costPct}%` }} />
          </div>
          <div className={`mt-2 text-[10px] ${compliance.cost ? 'text-slate-500' : 'text-rose-300'}`}>
            {compliance.cost ? 'Within configured ceiling' : 'Over configured ceiling'}
          </div>
        </div>

        {/* Dimension 3: Wall Clock Time */}
        <div className="flex flex-col justify-between rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider flex items-center gap-1.5">
              <Clock className="w-4 h-4 text-indigo-400" /> Wall Time
            </span>
            <span className="text-xs font-mono">{wallPct.toFixed(1)}%</span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">
            {totals.wall_seconds.toFixed(1)}s{' '}
            <span className="text-xs font-normal text-slate-500">/ {budget.max_wall_seconds}s</span>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-800" role="progressbar" aria-label="Wall time budget used" aria-valuenow={Math.round(wallPct)} aria-valuemin={0} aria-valuemax={100}>
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(wallPct)}`} style={{ width: `${wallPct}%` }} />
          </div>
          <div className={`mt-2 text-[10px] ${compliance.wall ? 'text-slate-500' : 'text-rose-300'}`}>
            {compliance.wall ? 'Within configured limit' : 'Over configured limit'}
          </div>
        </div>

        {/* Dimension 4: Call Count */}
        <div className="flex flex-col justify-between rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider flex items-center gap-1.5">
              <Hash className="w-4 h-4 text-amber-400" /> LLM Calls
            </span>
            <span className="text-xs font-mono">{callsPct.toFixed(1)}%</span>
          </div>
          <div className="text-xl font-bold font-mono text-slate-100">
            {totals.calls}{' '}
            <span className="text-xs font-normal text-slate-500">/ {budget.max_calls}</span>
          </div>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-800" role="progressbar" aria-label="Call budget used" aria-valuenow={Math.round(callsPct)} aria-valuemin={0} aria-valuemax={100}>
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(callsPct)}`} style={{ width: `${callsPct}%` }} />
          </div>
          <div className="mt-2 text-[10px] text-slate-500">
            {run.replacements} executed handoffs · {run.replacements_denied} denied
          </div>
        </div>
      </div>
    </section>
  );
};
