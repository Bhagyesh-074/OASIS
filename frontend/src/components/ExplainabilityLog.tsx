import React, { useState } from 'react';
import { Eye, Search, ChevronDown, ChevronRight } from 'lucide-react';
import { DecisionRecord } from '../types';

interface ExplainabilityLogProps {
  decisions: DecisionRecord[];
}

export const ExplainabilityLog: React.FC<ExplainabilityLogProps> = ({ decisions }) => {
  const [filterComponent, setFilterComponent] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const components = Array.from(new Set(decisions.map((d) => d.component))).sort();

  const filteredDecisions = decisions.filter((d) => {
    const matchesComp = filterComponent === 'all' || d.component === filterComponent;
    const matchesQuery =
      d.justification.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.decision.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.component.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesComp && matchesQuery;
  });

  const getDecisionBadge = (component: string) => {
    switch (component.toLowerCase()) {
      case 'tce':
        return 'bg-blue-950 text-blue-300 border-blue-800';
      case 'rbe':
        return 'bg-emerald-950 text-emerald-300 border-emerald-800';
      case 'rtpm':
        return 'bg-purple-950 text-purple-300 border-purple-800';
      case 'replacement':
        return 'bg-amber-950 text-amber-300 border-amber-800';
      case 'cm':
        return 'bg-cyan-950 text-cyan-300 border-cyan-800';
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <div className="bg-[#131B2E] border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
            <Eye className="w-5 h-5 text-sky-400" />
            <span>Explainability & Decision Audit Log (FR-24)</span>
          </h2>
          <p className="text-xs text-slate-400">Complete provenance and human-readable justifications for all decisions</p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-2.5 top-2.5" />
            <input
              type="text"
              placeholder="Search decisions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 bg-slate-900 text-xs font-mono border border-slate-700 rounded-lg focus:outline-none focus:border-sky-500 text-slate-200 w-44"
            />
          </div>

          <div className="relative">
            <select
              value={filterComponent}
              onChange={(e) => setFilterComponent(e.target.value)}
              className="px-2.5 py-1.5 bg-slate-900 text-xs font-mono border border-slate-700 rounded-lg focus:outline-none focus:border-sky-500 text-slate-200"
            >
              <option value="all">All Modules</option>
              {components.map((c) => (
                <option key={c} value={c}>
                  {c.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Decision Items */}
      {filteredDecisions.length === 0 ? (
        <div className="bg-slate-900/40 rounded-lg p-6 text-center text-slate-500 text-sm border border-slate-800/60">
          No decisions match the current query or filter.
        </div>
      ) : (
        <div className="space-y-2.5 max-h-[480px] overflow-y-auto pr-1">
          {filteredDecisions.map((item) => {
            const isExpanded = expandedId === item.decision_id;
            return (
              <div
                key={item.decision_id}
                className="bg-slate-900/70 border border-slate-800 hover:border-slate-700 rounded-lg p-3.5 transition-colors"
              >
                <div
                  className="flex items-start justify-between gap-3 cursor-pointer select-none"
                  onClick={() => setExpandedId(isExpanded ? null : item.decision_id)}
                >
                  <div className="flex items-start gap-2.5">
                    <span className="mt-0.5 text-slate-400">
                      {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                    </span>
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold border ${getDecisionBadge(item.component)}`}>
                          {item.component.toUpperCase()}
                        </span>
                        <span className="text-xs font-semibold text-slate-200 font-mono">
                          {item.decision}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 font-sans leading-relaxed">
                        {item.justification}
                      </p>
                    </div>
                  </div>

                  <span className="text-[11px] text-slate-500 font-mono shrink-0">
                    {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : ''}
                  </span>
                </div>

                {isExpanded && item.inputs && Object.keys(item.inputs).length > 0 && (
                  <div className="mt-3 pt-3 border-t border-slate-800 font-mono text-[11px] bg-slate-950/60 p-2.5 rounded">
                    <div className="text-slate-400 font-semibold mb-1">Decision Inputs:</div>
                    <pre className="text-slate-300 overflow-x-auto">
                      {JSON.stringify(item.inputs, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
