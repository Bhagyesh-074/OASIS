import React from 'react';
import { Activity, Clock3, Coins, Cpu, Hash, Layers3 } from 'lucide-react';
import { RunDetail } from '../types';

interface RunOverviewProps {
  run: RunDetail;
}

const statusStyles: Record<RunDetail['status'], string> = {
  queued: 'border-blue-400/20 bg-blue-400/10 text-blue-200',
  running: 'border-amber-400/20 bg-amber-400/10 text-amber-200',
  completed: 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200',
  halted_budget: 'border-rose-400/20 bg-rose-400/10 text-rose-200',
  failed: 'border-rose-400/20 bg-rose-400/10 text-rose-200',
  cancelled: 'border-slate-400/20 bg-slate-400/10 text-slate-200',
};

const Metric: React.FC<{
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
  color: string;
}> = ({ label, value, detail, icon, color }) => (
  <div className="min-w-0 rounded-xl border border-white/[0.07] bg-slate-950/35 p-4">
    <div className="flex items-center justify-between gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</span>
      <span className={color}>{icon}</span>
    </div>
    <div className="mt-3 truncate text-2xl font-semibold tracking-tight text-slate-100">{value}</div>
    <div className="mt-1 truncate text-xs text-slate-500">{detail}</div>
  </div>
);

export const RunOverview: React.FC<RunOverviewProps> = ({ run }) => (
  <section className="relative overflow-hidden rounded-2xl border border-sky-300/10 bg-gradient-to-br from-[#14243a] via-[#111a2b] to-[#101522] p-5 shadow-[0_24px_70px_-45px_rgba(14,165,233,0.35)] sm:p-6">
    <div className="pointer-events-none absolute -right-20 -top-32 h-72 w-72 rounded-full bg-sky-400/[0.07] blur-3xl" />
    <div className="relative">
      <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-300">
              <Activity className="h-3.5 w-3.5" />
              Run overview
            </span>
            <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] ${statusStyles[run.status]}`}>
              {run.status.replace('_', ' ')}
            </span>
          </div>
          <h1 className="mt-3 text-xl font-semibold tracking-tight text-white sm:text-2xl">
            {run.task_statement || 'Execution details'}
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
            {run.status_detail || 'Inspect the run configuration, resource use, and audit events below.'}
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            {[
              ['RUN', run.run_id],
              ['TASK', run.task_id],
              ['ARM', run.arm],
              ['FRAMEWORK', run.framework],
              ['SEED', run.seed === undefined ? undefined : String(run.seed)],
            ]
              .filter((entry): entry is [string, string] => Boolean(entry[1]))
              .map(([label, value]) => (
                <span key={label} className="inline-flex max-w-full items-center gap-2 rounded-lg border border-white/[0.07] bg-slate-950/45 px-2.5 py-1.5 text-[11px]">
                  <span className="shrink-0 text-[9px] font-semibold tracking-[0.12em] text-slate-500">{label}</span>
                  <span className="truncate font-mono text-slate-300">{value}</span>
                </span>
              ))}
          </div>
        </div>

        {run.quality?.composite != null && (
          <div className="flex shrink-0 items-center gap-3 rounded-xl border border-indigo-300/10 bg-indigo-400/[0.06] px-4 py-3 lg:min-w-40">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-indigo-400/10 text-indigo-300">
              <Layers3 className="h-5 w-5" />
            </div>
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-indigo-200/60">Quality score</div>
              <div className="mt-0.5 font-mono text-xl font-semibold text-indigo-100">{run.quality.composite.toFixed(3)}</div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric
          label="Provider cost"
          value={`$${run.totals.cost_usd.toFixed(4)}`}
          detail={run.budget ? `of $${run.budget.max_cost_usd.toFixed(2)} allowed` : 'Run total'}
          icon={<Coins className="h-4 w-4" />}
          color="text-emerald-300"
        />
        <Metric
          label="Tokens"
          value={run.totals.tokens.toLocaleString()}
          detail={run.budget ? `of ${run.budget.max_tokens.toLocaleString()} allowed` : 'Run total'}
          icon={<Cpu className="h-4 w-4" />}
          color="text-sky-300"
        />
        <Metric
          label="Provider calls"
          value={run.totals.calls.toLocaleString()}
          detail={run.budget ? `of ${run.budget.max_calls} allowed` : 'Run total'}
          icon={<Hash className="h-4 w-4" />}
          color="text-amber-300"
        />
        <Metric
          label="Elapsed time"
          value={`${run.totals.wall_seconds.toFixed(1)}s`}
          detail={run.budget ? `of ${run.budget.max_wall_seconds}s allowed` : 'Wall-clock time'}
          icon={<Clock3 className="h-4 w-4" />}
          color="text-indigo-300"
        />
      </div>
    </div>
  </section>
);
