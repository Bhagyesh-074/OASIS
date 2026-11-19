import React from 'react';
import { RefreshCw, ArrowRight, XCircle, CheckCircle, ShieldAlert } from 'lucide-react';
import { ReplacementEvent } from '../types';

interface ReplacementTimelineProps {
  replacements: ReplacementEvent[];
}

export const ReplacementTimeline: React.FC<ReplacementTimelineProps> = ({ replacements }) => {
  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <RefreshCw className="w-5 h-5 text-amber-400" />
            <span>Replacement & Swap Audit Trail</span>
          </h2>
          <p className="text-xs text-slate-400">Dynamic agent substitutions with state preservation and budget checks</p>
        </div>
        <div className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300">
          Total Swaps: <span className="font-semibold text-white">{replacements.length}</span>
        </div>
      </div>

      {replacements.length === 0 ? (
        <div className="bg-slate-900/40 rounded-lg p-6 text-center text-slate-500 text-sm border border-slate-800/60">
          No replacement events occurred during this run.
        </div>
      ) : (
        <div className="space-y-3">
          {replacements.map((ev, idx) => {
            const isExecuted = ev.outcome === 'executed';
            return (
              <div
                key={idx}
                className={`p-4 rounded-lg border text-sm flex flex-col gap-2 ${
                  isExecuted
                    ? 'bg-slate-900/80 border-slate-700'
                    : 'bg-rose-950/20 border-rose-800/60'
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 font-mono">
                    <span className="text-slate-300 font-semibold">{ev.incumbent_agent || 'Incumbent'}</span>
                    <ArrowRight className="w-4 h-4 text-slate-500" />
                    <span className="text-sky-300 font-semibold">{ev.replacement_candidate || 'Candidate'}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    {isExecuted ? (
                      <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800 font-medium">
                        <CheckCircle className="w-3.5 h-3.5" /> Executed
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-xs px-2.5 py-0.5 rounded-full bg-rose-950 text-rose-400 border border-rose-800 font-medium">
                        <XCircle className="w-3.5 h-3.5" /> Denied ({ev.outcome})
                      </span>
                    )}
                    <span className="text-[11px] text-slate-500 font-mono">
                      {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : ''}
                    </span>
                  </div>
                </div>

                {ev.trigger_reason && (
                  <div className="text-xs text-amber-300/90 font-mono flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
                    <span>Trigger: {ev.trigger_reason}</span>
                  </div>
                )}

                <div className="text-xs text-slate-400 bg-slate-950/60 p-2.5 rounded border border-slate-900 leading-relaxed font-mono">
                  <span className="text-slate-500">Justification: </span>
                  {ev.justification}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
