import React from 'react';
import { Shield, Activity, DollarSign, Square } from 'lucide-react';
import { RunDetail, SpendStatus } from '../types';

interface HeaderProps {
  run: RunDetail | null;
  spend: SpendStatus | null;
  runIdInput: string;
  setRunIdInput: (id: string) => void;
  onLoadRun: (id: string) => void;
  onCancelRun: () => void;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  run,
  spend,
  runIdInput,
  setRunIdInput,
  onLoadRun,
  onCancelRun,
  isLoading,
}) => {
  const getStatusColor = (status?: string) => {
    switch (status) {
      case 'running':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse';
      case 'completed':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'halted_budget':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'failed':
      case 'cancelled':
        return 'bg-red-500/20 text-red-300 border-red-500/40';
      case 'queued':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40';
      default:
        return 'bg-slate-700/50 text-slate-300 border-slate-600';
    }
  };

  const spendPercent = spend
    ? Math.min(100, (spend.cumulative_spend_usd / spend.spend_ceiling_usd) * 100)
    : 0;

  return (
    <header className="border-b border-slate-800 bg-[#0F172A]/90 backdrop-blur sticky top-0 z-50 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand & System Status */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-xl tracking-tight text-white">OASIS</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-sky-950 text-sky-400 border border-sky-800 font-mono">
                Supervisor v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">Dynamic Supervisory Layer for Multi-Agent LLMs</p>
          </div>
        </div>

        {/* Global Spend Ceiling Guardrail (NFR-3) */}
        {spend && (
          <div className="flex items-center gap-4 bg-slate-900/80 px-4 py-2 rounded-lg border border-slate-800">
            <DollarSign className="w-5 h-5 text-emerald-400" />
            <div className="flex flex-col">
              <div className="flex justify-between text-xs gap-4 font-mono">
                <span className="text-slate-400">Global Spend:</span>
                <span className="font-semibold text-slate-200">
                  ${spend.cumulative_spend_usd.toFixed(2)} / ${spend.spend_ceiling_usd.toFixed(2)}
                </span>
              </div>
              <div className="w-40 h-1.5 bg-slate-800 rounded-full mt-1.5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    spendPercent > 90 ? 'bg-rose-500' : spendPercent > 70 ? 'bg-amber-400' : 'bg-emerald-400'
                  }`}
                  style={{ width: `${spendPercent}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Run Selector & Controller */}
        <div className="flex items-center gap-2">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (runIdInput.trim()) onLoadRun(runIdInput.trim());
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              placeholder="Run ID (e.g. 01J...)"
              value={runIdInput}
              onChange={(e) => setRunIdInput(e.target.value)}
              className="px-3 py-1.5 bg-slate-900 text-sm font-mono border border-slate-700 rounded-lg focus:outline-none focus:border-sky-500 text-slate-200 w-44 md:w-56"
            />
            <button
              type="submit"
              disabled={isLoading || !runIdInput.trim()}
              className="px-3 py-1.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1.5 shadow-sm"
            >
              <Activity className="w-4 h-4" />
              <span>Load</span>
            </button>
          </form>

          {run && run.status === 'running' && (
            <button
              onClick={onCancelRun}
              className="px-3 py-1.5 bg-rose-600/80 hover:bg-rose-600 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-1.5 border border-rose-500"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Cancel</span>
            </button>
          )}

          {run && (
            <span
              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold uppercase tracking-wider border ${getStatusColor(
                run.status
              )}`}
            >
              {run.status.replace('_', ' ')}
            </span>
          )}
        </div>
      </div>
    </header>
  );
};
