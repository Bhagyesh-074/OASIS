import React from 'react';
import { Cpu, DollarSign, Clock, Hash, CheckCircle2, AlertTriangle } from 'lucide-react';
import { RunDetail } from '../types';

interface BudgetGaugesProps {
  run: RunDetail;
}

export const BudgetGauges: React.FC<BudgetGaugesProps> = ({ run }) => {
  const budget = run.budget || {
    max_tokens: 30000,
    max_cost_usd: 0.15,
    max_wall_seconds: 60,
    max_calls: 20,
  };
  const totals = run.totals || {
    tokens: 0,
    cost_usd: 0,
    wall_seconds: 0,
    calls: 0,
    supervision_tokens: 0,
    supervision_share: 0,
  };
  const compliance = run.compliance || {
    tokens: true,
    cost: true,
    wall: true,
    calls: true,
    all: true,
  };

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
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <span>4D Real-Time Budget Enforcement (RBE)</span>
            {compliance.all ? (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800">
                <CheckCircle2 className="w-3 h-3" /> Compliant
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-rose-950 text-rose-400 border border-rose-800">
                <AlertTriangle className="w-3 h-3" /> Limit Exceeded
              </span>
            )}
          </h2>
          <p className="text-xs text-slate-400">Strictly enforced four-dimensional admission control</p>
        </div>
        {bindingConstraint && (
          <div className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">
            Binding: <span className="text-amber-400 font-semibold">{bindingConstraint}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Dimension 1: Tokens */}
        <div className="bg-slate-900/60 p-4 rounded-lg border border-slate-800 flex flex-col justify-between">
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
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-3">
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(tokenPct)}`} style={{ width: `${tokenPct}%` }} />
          </div>
          <div className="text-[11px] text-slate-500 mt-2 font-mono">
            Supervision: {((totals.supervision_share || 0) * 100).toFixed(1)}%
          </div>
        </div>

        {/* Dimension 2: Cost USD */}
        <div className="bg-slate-900/60 p-4 rounded-lg border border-slate-800 flex flex-col justify-between">
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
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-3">
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(costPct)}`} style={{ width: `${costPct}%` }} />
          </div>
          <div className="text-[11px] text-slate-500 mt-2 font-mono">
            Status: {compliance.cost ? 'Within budget' : 'Over budget'}
          </div>
        </div>

        {/* Dimension 3: Wall Clock Time */}
        <div className="bg-slate-900/60 p-4 rounded-lg border border-slate-800 flex flex-col justify-between">
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
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-3">
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(wallPct)}`} style={{ width: `${wallPct}%` }} />
          </div>
          <div className="text-[11px] text-slate-500 mt-2 font-mono">
            Secs: {totals.wall_seconds.toFixed(1)}s
          </div>
        </div>

        {/* Dimension 4: Call Count */}
        <div className="bg-slate-900/60 p-4 rounded-lg border border-slate-800 flex flex-col justify-between">
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
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden mt-3">
            <div className={`h-full rounded-full transition-all duration-300 ${getProgressColor(callsPct)}`} style={{ width: `${callsPct}%` }} />
          </div>
          <div className="text-[11px] text-slate-500 mt-2 font-mono">
            Handoffs: {run.replacements} swap(s)
          </div>
        </div>
      </div>
    </div>
  );
};
