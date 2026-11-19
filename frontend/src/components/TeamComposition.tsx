import React from 'react';
import { Users, Network, Terminal, Tag, Cpu, Coins, Hash } from 'lucide-react';
import { RunDetail } from '../types';

interface TeamCompositionProps {
  run: RunDetail;
}

export const TeamComposition: React.FC<TeamCompositionProps> = ({ run }) => {
  const teamConfig = run.config;
  const roles = teamConfig?.roles || [];
  const origin = teamConfig?.origin || run.arm || 'unknown';

  return (
    <section className="flex flex-col gap-5 rounded-2xl border border-white/[0.07] bg-gradient-to-br from-[#111a29] to-[#0d131e] p-5 shadow-[0_20px_60px_-50px_rgba(56,189,248,0.28)] sm:p-6">
      {/* Header & Task metadata */}
      <div>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <Users className="h-5 w-5 text-sky-300" />
            <span>Team configuration</span>
          </h2>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.07] bg-slate-950/45 px-2.5 py-1.5 text-[10px] font-mono text-sky-200">
              <Network className="w-3.5 h-3.5" />
              {teamConfig?.topology || 'Topology unavailable'}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-white/[0.07] bg-slate-950/45 px-2.5 py-1.5 text-[10px] font-mono text-indigo-200">
              <Tag className="w-3.5 h-3.5" />
              {origin}
            </span>
            <span className="rounded-lg border border-white/[0.07] bg-slate-950/45 px-2.5 py-1.5 text-[10px] font-mono text-slate-300">
              {teamConfig ? `${teamConfig.team_size} ${teamConfig.team_size === 1 ? 'agent' : 'agents'}` : 'Team size unavailable'}
            </span>
          </div>
        </div>

      </div>

      {/* Agents / Roles Grid */}
      {roles.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/[0.09] bg-slate-950/20 px-4 py-8 text-center text-sm text-slate-500">
          No role configuration was recorded for this run.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {roles.map((role, idx) => (
          <div
            key={`${role.role}-${idx}`}
            className="group flex flex-col justify-between rounded-xl border border-white/[0.07] bg-slate-950/35 p-4 transition hover:border-sky-300/20 hover:bg-slate-950/55"
          >
            <div>
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="truncate text-sm font-semibold text-slate-100">{role.role}</span>
                <span className="shrink-0 rounded-md border border-white/[0.07] bg-white/[0.03] px-2 py-1 text-[10px] font-mono text-slate-400">
                  temp {role.temperature}
                </span>
              </div>
              <div className="truncate text-[11px] font-mono text-sky-200/80" title={role.model}>
                {role.model}
              </div>
            </div>

            <div className="mt-4 border-t border-white/[0.06] pt-3">
              <div className="mb-3 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                <Terminal className="h-3 w-3" /> {role.template_id}
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <div className="flex items-center gap-1 text-[9px] text-slate-600"><Cpu className="h-3 w-3" /> Est. tokens</div>
                  <div className="mt-1 text-[11px] font-mono text-slate-300">{role.est_tokens.toLocaleString()}</div>
                </div>
                <div>
                  <div className="flex items-center gap-1 text-[9px] text-slate-600"><Coins className="h-3 w-3" /> Est. cost</div>
                  <div className="mt-1 text-[11px] font-mono text-slate-300">${role.est_cost_usd.toFixed(3)}</div>
                </div>
                <div>
                  <div className="flex items-center gap-1 text-[9px] text-slate-600"><Hash className="h-3 w-3" /> Calls</div>
                  <div className="mt-1 text-[11px] font-mono text-slate-300">{role.est_calls}</div>
                </div>
              </div>
            </div>
          </div>
        ))}
        </div>
      )}
    </section>
  );
};
