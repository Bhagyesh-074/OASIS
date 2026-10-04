import React from 'react';
import { Layers, CheckCircle2, AlertCircle } from 'lucide-react';
import { ScoreRecord } from '../types';

interface LayeredScoresProps {
  scores: ScoreRecord[];
  finalCompositeScore?: number;
}

export const LayeredScores: React.FC<LayeredScoresProps> = ({ scores, finalCompositeScore }) => {
  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            <span>Layered Agent Performance Scores (RTPM)</span>
          </h2>
          <p className="text-xs text-slate-400">Separate evaluation layers: L1 Embed, L2 Judge, L3 Verifier (FR-15)</p>
        </div>
        {finalCompositeScore !== undefined && (
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-wider text-slate-400 font-medium">Final Composite</div>
            <div className="text-2xl font-bold font-mono text-sky-400">{finalCompositeScore.toFixed(3)}</div>
          </div>
        )}
      </div>

      {scores.length === 0 ? (
        <div className="bg-slate-900/40 rounded-lg p-6 text-center text-slate-500 text-sm border border-slate-800/60">
          No layered scores recorded for this run yet.
        </div>
      ) : (
        <div className="space-y-3">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm font-mono">
              <thead className="text-xs uppercase bg-slate-900/80 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Output / Score ID</th>
                  <th className="py-2.5 px-3">Layer</th>
                  <th className="py-2.5 px-3">Composite</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {scores.map((sc, idx) => {
                  const isPassing = sc.composite >= 0.7;
                  return (
                    <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-2.5 px-3 text-xs text-slate-400 font-mono">
                        {sc.output_id ? sc.output_id.substring(0, 10) + '...' : sc.score_id.substring(0, 10) + '...'}
                      </td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`text-xs px-2 py-0.5 rounded font-semibold ${
                            sc.layer === 'l1_embed'
                              ? 'bg-blue-950 text-blue-300 border border-blue-800'
                              : sc.layer === 'l2_judge'
                              ? 'bg-purple-950 text-purple-300 border border-purple-800'
                              : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          }`}
                        >
                          {sc.layer.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-slate-200 font-bold">
                        {sc.composite.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3">
                        {isPassing ? (
                          <span className="inline-flex items-center gap-1 text-xs text-emerald-400">
                            <CheckCircle2 className="w-3.5 h-3.5" /> Passing
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs text-amber-400">
                            <AlertCircle className="w-3.5 h-3.5" /> Flagged
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
