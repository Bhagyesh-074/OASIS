import React from 'react';
import { ArrowUpRight, DollarSign, LoaderCircle, Shield, Square } from 'lucide-react';
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

const statusStyles: Record<RunDetail['status'], string> = {
  queued: 'border-blue-400/20 bg-blue-400/10 text-blue-200',
  running: 'border-amber-400/20 bg-amber-400/10 text-amber-200',
  completed: 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200',
  halted_budget: 'border-rose-400/20 bg-rose-400/10 text-rose-200',
  failed: 'border-rose-400/20 bg-rose-400/10 text-rose-200',
  cancelled: 'border-slate-400/20 bg-slate-400/10 text-slate-200',
};

export const Header: React.FC<HeaderProps> = ({
  run,
  spend,
  runIdInput,
  setRunIdInput,
  onLoadRun,
  onCancelRun,
  isLoading,
}) => {
  const spendPercent = spend
    ? Math.min(
        100,
        spend.ceiling_usd > 0 ? (spend.cumulative_spend_usd / spend.ceiling_usd) * 100 : 0,
      )
    : 0;

  return (
    <header className="sticky top-0 z-50 border-b border-white/[0.07] bg-[#090e18]/90 px-4 py-3 backdrop-blur-xl sm:px-6">
      <div className="mx-auto flex max-w-[1500px] flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-sky-300/15 bg-gradient-to-br from-sky-400/20 to-indigo-500/10 shadow-[0_0_30px_-12px_rgba(56,189,248,0.65)]">
              <Shield className="h-5 w-5 text-sky-200" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-semibold tracking-[0.18em] text-white">OASIS</span>
                <span className="rounded border border-white/[0.08] bg-white/[0.04] px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-[0.12em] text-slate-400">
                  Control room
                </span>
              </div>
              <p className="hidden truncate text-[11px] text-slate-500 sm:block">
                Multi-agent execution observability
              </p>
            </div>
          </div>
          {run && (
            <span
              className={`shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider ${statusStyles[run.status]}`}
            >
              {run.status.replace('_', ' ')}
            </span>
          )}
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center lg:justify-end">
          {spend && (
            <div className="min-w-[220px] rounded-xl border border-white/[0.07] bg-white/[0.03] px-3 py-2">
              <div className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                  <DollarSign className="h-3.5 w-3.5 text-emerald-300" />
                  Global spend
                </span>
                <span className="font-mono text-xs font-medium text-slate-200">
                  ${spend.cumulative_spend_usd.toFixed(2)}{' '}
                  <span className="text-slate-600">/</span> ${spend.ceiling_usd.toFixed(2)}
                </span>
              </div>
              <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    spendPercent > 90
                      ? 'bg-rose-400'
                      : spendPercent > 70
                        ? 'bg-amber-300'
                        : 'bg-emerald-300'
                  }`}
                  style={{ width: `${spendPercent}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[9px] text-slate-600">
                <span>{spendPercent.toFixed(1)}% of soft ceiling</span>
                <span>Hard stop ${spend.kill_ceiling_usd.toFixed(2)}</span>
              </div>
            </div>
          )}

          {run && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (runIdInput.trim()) onLoadRun(runIdInput.trim());
              }}
              className="flex min-w-0 items-center gap-2"
            >
              <input
                type="text"
                aria-label="Run ID"
                placeholder="Paste a run ID…"
                value={runIdInput}
                onChange={(event) => setRunIdInput(event.target.value)}
                className="min-w-0 flex-1 rounded-lg border border-white/[0.09] bg-slate-950/70 px-3 py-2 text-xs font-mono text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-sky-400/50 focus:ring-2 focus:ring-sky-400/10 sm:w-48 sm:flex-none"
              />
              <button
                type="submit"
                disabled={isLoading || !runIdInput.trim()}
                className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-sky-300/20 bg-sky-400/10 px-3 py-2 text-xs font-semibold text-sky-100 transition hover:border-sky-300/40 hover:bg-sky-400/15 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {isLoading ? (
                  <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <ArrowUpRight className="h-3.5 w-3.5" />
                )}
                <span>{isLoading ? 'Loading' : 'Open run'}</span>
              </button>
            </form>
          )}

          {run?.status === 'running' && (
            <button
              onClick={onCancelRun}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-xs font-semibold text-rose-200 transition hover:bg-rose-400/20"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              Stop run
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
