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

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col gap-5">
      <div>
        <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-sky-400" />
          <span>Benchmark Results</span>
        </h2>
        <p className="text-xs text-slate-400">
          Statistics are calculated from completed benchmark runs. Reference arm:{' '}
          <span className="font-mono">{summary?.reference_arm ?? '—'}</span>
        </p>
      </div>

      {error ? (
        <div role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-4 text-sm text-rose-300">
          {error}
        </div>
      ) : isLoading ? (
        <div role="status" className="py-8 text-center text-sm text-slate-400">
          Loading benchmark results…
        </div>
      ) : arms.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-6 text-center text-sm text-slate-400">
          No completed benchmark results are available.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm font-mono">
            <thead className="text-xs uppercase bg-slate-900/80 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Arm</th>
                <th className="py-2.5 px-3">Runs</th>
                <th className="py-2.5 px-3">Quality Median [95% CI]</th>
                <th className="py-2.5 px-3">Cost Median [95% CI]</th>
                <th className="py-2.5 px-3">Tokens</th>
                <th className="py-2.5 px-3">Budget Breach Rate</th>
                <th className="py-2.5 px-3">Adjusted p-value</th>
                <th className="py-2.5 px-3">Effect Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {arms.map(([arm, result]) => (
                <tr key={arm} className="hover:bg-slate-900/40 transition-colors">
                  <td className="py-3 px-3 text-slate-100">{armNames[arm] ?? arm}</td>
                  <td className="py-3 px-3 text-slate-300">{result.total_runs}</td>
                  <td className="py-3 px-3 text-sky-300">
                    {formatNumber(result.quality.median, 3)}{' '}
                    <span className="text-xs text-slate-500 font-normal">
                      {formatInterval(result.quality.ci_95, 3)}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-emerald-300">
                    ${result.cost_usd.median.toFixed(4)}{' '}
                    <span className="text-xs text-slate-500 font-normal">
                      {formatInterval(result.cost_usd.ci_95, 4)}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-300">{result.tokens.median.toLocaleString()}</td>
                  <td className="py-3 px-3 text-slate-300">
                    {((1 - result.compliance_rate) * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-slate-400">
                    {formatNumber(result.holm_adjusted_p_value, 4)}
                  </td>
                  <td className="py-3 px-3 text-slate-400">
                    {formatNumber(result.effect_size, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
