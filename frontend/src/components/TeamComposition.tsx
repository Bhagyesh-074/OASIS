import React from 'react';
import { Users, Network, Terminal, Tag, FileText } from 'lucide-react';
import { RunDetail } from '../types';

interface TeamCompositionProps {
  run: RunDetail;
}

export const TeamComposition: React.FC<TeamCompositionProps> = ({ run }) => {
  const teamConfig = run.config || (run as any).team_config;
  const roles = teamConfig?.roles || [];
  const origin = teamConfig?.origin || run.arm || 'tce';

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col gap-5">
      {/* Header & Task metadata */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <Users className="w-5 h-5 text-sky-400" />
            <span>Team Composition & Topology</span>
          </h2>
          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-sky-300 font-mono flex items-center gap-1.5">
              <Network className="w-3.5 h-3.5" />
              {teamConfig?.topology || 'hub_and_spoke'}
            </span>
            <span className="text-xs px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-purple-300 font-mono flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5" />
              Origin: {origin}
            </span>
            <span className="text-xs px-2.5 py-1 rounded bg-slate-900 border border-slate-700 text-amber-300 font-mono">
              Tier: {run.complexity_tier || 'moderate'}
            </span>
          </div>
        </div>

        {run.task_statement && (
          <div className="bg-slate-900/60 p-3.5 rounded-lg border border-slate-800/80 text-sm text-slate-300 font-mono mt-3 flex items-start gap-2.5">
            <FileText className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
            <div className="line-clamp-2">{run.task_statement}</div>
          </div>
        )}
      </div>

      {/* Agents / Roles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {roles.map((role: any, idx: number) => (
          <div
            key={idx}
            className="bg-slate-900/70 border border-slate-800 rounded-lg p-4 flex flex-col justify-between hover:border-slate-700 transition-colors"
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-slate-100 text-sm">{role.role_name}</span>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                  T={role.temperature}
                </span>
              </div>
              <div className="text-xs font-mono text-sky-400 truncate mb-3">
                {role.model}
              </div>
            </div>

            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1 flex items-center gap-1">
                <Terminal className="w-3 h-3" /> Tools
              </div>
              <div className="flex flex-wrap gap-1">
                {role.tools && role.tools.length > 0 ? (
                  role.tools.map((t: string, tIdx: number) => (
                    <span
                      key={tIdx}
                      className="text-[10px] px-2 py-0.5 rounded bg-slate-800/80 text-slate-300 border border-slate-700 font-mono"
                    >
                      {t}
                    </span>
                  ))
                ) : (
                  <span className="text-[10px] text-slate-500 italic">No tools</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
