import React from 'react';
import { BarChart3 } from 'lucide-react';
import { BenchmarkSummary } from '../types';

interface BenchmarkViewProps {
  summary: BenchmarkSummary | null;
  isLoading: boolean;
  error: string | null;
}

const armNames: Record<string, string> = {
  full: 'OASIS Full',
  rbe_only: 'RBE Only',
  rtpm_only: 'RTPM Only',
  monitor_only: 'Monitor Only',
  unconstrained: 'Unconstrained Baseline',
  vanilla_fixed: 'Vanilla Fixed',
  captainagent: 'CaptainAgent',
};

const formatNumber = (value: number | null, digits: number): string =>
  value === null ? '—' : value.toFixed(digits);

const formatInterval = (interval: [number | null, number | null], digits: number): string =>
  `[${formatNumber(interval[0], digits)}, ${formatNumber(interval[1], digits)}]`;

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({ summary, isLoading, error }) => {
  const arms = Object.entries(summary?.arms ?? {});
  const completedRuns = arms.reduce((total, [, result]) => total + result.total_runs, 0);
  const qualityLeader = arms
    .filter(([, result]) => result.quality.median !== null)
    .sort((left, right) => (right[1].quality.median ?? 0) - (left[1].quality.median ?? 0))[0];
  const lowestCost = [...arms].sort((left, right) => left[1].cost_usd.median - right[1].cost_usd.median)[0];

  return (
    <section className="flex flex-col gap-5 rounded-2xl border border-white/[0.07] bg-gradient-to-br from-[#111a29] to-[#0d131e] p-5 shadow-[0_20px_60px_-50px_rgba(99,102,241,0.3)] sm:p-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-indigo-200/70">
            <BarChart3 className="h-4 w-4" />
            Evaluation matrix
          </div>
          <h2 className="mt-2 text-xl font-semibold tracking-tight text-white">Benchmark analysis</h2>
          <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">
            Aggregates use persisted run data only. Confidence intervals and paired comparisons remain blank when scores are unavailable.
          </p>
        </div>
        {summary && (
          <div className="rounded-lg border border-white/[0.07] bg-slate-950/40 px-3 py-2 text-[10px] text-slate-500">
            Reference arm <span className="ml-1 font-mono font-medium text-slate-300">{summary.reference_arm}</span>
          </div>
        )}
      </div>

      {error ? (
        <div role="alert" className="rounded-xl border border-rose-400/15 bg-rose-400/[0.06] p-4 text-sm text-rose-200">
          {error}
        </div>
      ) : isLoading ? (
        <div role="status" className="rounded-xl border border-white/[0.06] bg-slate-950/25 py-12 text-center text-sm text-slate-400">
          Loading benchmark results…
        </div>
      ) : arms.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/[0.09] bg-slate-950/20 px-5 py-10 text-center">
          <div className="mx-auto grid h-11 w-11 place-items-center rounded-xl border border-indigo-300/10 bg-indigo-300/[0.06] text-indigo-200/80">
            <BarChart3 className="h-5 w-5" />
          </div>
          <h3 className="mt-3 text-sm font-semibold text-slate-200">No benchmark runs to compare yet</h3>
          <p className="mx-auto mt-1 max-w-md text-xs leading-5 text-slate-500">
            Once completed runs are recorded, this view will summarize each arm&apos;s quality, cost, token use, and budget compliance.
          </p>
        </div>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
              <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">Experimental arms</div>
              <div className="mt-2 text-2xl font-semibold text-white">{arms.length}</div>
              <div className="mt-1 text-[10px] text-slate-600">with recorded run summaries</div>
            </div>
            <div className="rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
              <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">Runs summarized</div>
              <div className="mt-2 text-2xl font-semibold text-white">{completedRuns}</div>
              <div className="mt-1 text-[10px] text-slate-600">across all visible arms</div>
            </div>
            <div className="rounded-xl border border-white/[0.06] bg-slate-950/35 p-4">
              <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
                {qualityLeader ? 'Highest median quality' : 'Lowest median cost'}
              </div>
              <div className="mt-2 truncate text-lg font-semibold text-white">
                {qualityLeader
                  ? armNames[qualityLeader[0]] ?? qualityLeader[0]
                  : lowestCost
                    ? armNames[lowestCost[0]] ?? lowestCost[0]
                    : '—'}
              </div>
              <div className="mt-1 text-[10px] text-slate-600">
                {qualityLeader
                  ? `Median ${qualityLeader[1].quality.median?.toFixed(3)}`
                  : lowestCost
                    ? `Median $${lowestCost[1].cost_usd.median.toFixed(4)}`
                    : 'No aggregate available'}
              </div>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-white/[0.06]">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[980px] text-left text-xs">
                <thead className="bg-slate-950/65 text-[9px] uppercase tracking-[0.13em] text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Arm</th>
                    <th className="px-3 py-3">Runs</th>
                    <th className="px-3 py-3">Quality median · 95% CI</th>
                    <th className="px-3 py-3">Cost median · 95% CI</th>
                    <th className="px-3 py-3">Tokens median</th>
                    <th className="px-3 py-3">Budget breach</th>
                    <th className="px-3 py-3">Holm-adjusted p</th>
                    <th className="px-3 py-3">Effect size</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.05]">
                  {arms.map(([arm, result]) => (
                    <tr
                      key={arm}
                      className={`transition hover:bg-white/[0.025] ${
                        arm === summary?.reference_arm ? 'bg-sky-300/[0.025]' : ''
                      }`}
                    >
                      <td className="px-4 py-3.5">
                        <div className="font-medium text-slate-200">{armNames[arm] ?? arm}</div>
                        {arm === summary?.reference_arm && (
                          <div className="mt-1 text-[9px] uppercase tracking-wider text-sky-300/70">Reference</div>
                        )}
                      </td>
                      <td className="px-3 py-3.5 font-mono text-slate-400">{result.total_runs}</td>
                      <td className="px-3 py-3.5">
                        <div className="font-mono text-sky-200">{formatNumber(result.quality.median, 3)}</div>
                        <div className="mt-1 text-[10px] text-slate-600">{formatInterval(result.quality.ci_95, 3)}</div>
                      </td>
                      <td className="px-3 py-3.5">
                        <div className="font-mono text-emerald-200">${result.cost_usd.median.toFixed(4)}</div>
                        <div className="mt-1 text-[10px] text-slate-600">{formatInterval(result.cost_usd.ci_95, 4)}</div>
                      </td>
                      <td className="px-3 py-3.5 font-mono text-slate-300">{result.tokens.median.toLocaleString()}</td>
                      <td className="px-3 py-3.5">
                        <span className={`rounded-md px-2 py-1 font-mono ${
                          result.compliance_rate >= 0.99
                            ? 'bg-emerald-300/[0.07] text-emerald-200'
                            : 'bg-amber-300/[0.07] text-amber-200'
                        }`}>
                          {((1 - result.compliance_rate) * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-3 py-3.5 font-mono text-slate-400">{formatNumber(result.holm_adjusted_p_value, 4)}</td>
                      <td className="px-3 py-3.5 font-mono text-slate-400">{formatNumber(result.effect_size, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </section>
  );
};
