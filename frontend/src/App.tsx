import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { BudgetGauges } from './components/BudgetGauges';
import { TeamComposition } from './components/TeamComposition';
import { LayeredScores } from './components/LayeredScores';
import { ReplacementTimeline } from './components/ReplacementTimeline';
import { ExplainabilityLog } from './components/ExplainabilityLog';
import { BenchmarkView } from './components/BenchmarkView';
import { RunDetail, SpendStatus, DecisionRecord, ReplacementEvent, ScoreRecord } from './types';
import { fetchRun, fetchDecisions, fetchSpend, cancelRun, subscribeToRunEvents } from './api';
import { Activity, BarChart2 } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'run' | 'benchmark'>('run');
  const [runIdInput, setRunIdInput] = useState<string>('01JRUN00000000000000000001');
  const [currentRunId, setCurrentRunId] = useState<string>('01JRUN00000000000000000001');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isLiveConnected, setIsLiveConnected] = useState<boolean>(false);

  // Core state
  const [run, setRun] = useState<RunDetail | null>(null);
  const [spend, setSpend] = useState<SpendStatus | null>(null);
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [replacements, setReplacements] = useState<ReplacementEvent[]>([]);
  const [scores, setScores] = useState<ScoreRecord[]>([]);

  // Default demo run data for offline viva demonstration
  const loadDemoData = useCallback((runId: string) => {
    const demoRun: RunDetail = {
      run_id: runId,
      benchmark_task_id: 'task_qa_042',
      status: 'completed',
      status_detail: 'Execution finished successfully under 4D budget constraints',
      task_statement: 'Synthesize comparative clinical trial outcomes for SGLT2 inhibitors vs GLP-1 agonists in heart failure patients.',
      complexity_tier: 'moderate',
      arm: 'full',
      seed: 1,
      framework: 'langgraph',
      config: {
        config_id: 'cfg_01J999',
        team_size: 3,
        origin: 'tce',
        topology: 'hub_and_spoke',
        roles: [
          { role_name: 'clinical_lead', model: 'gpt-4o-2024-11-20', temperature: 0.1, tools: ['search_pubmed', 'extract_tables'] },
          { role_name: 'statistician', model: 'gpt-4o-mini-2024-07-18', temperature: 0.0, tools: ['calc_ci', 'verify_pvals'] },
          { role_name: 'synthesizer', model: 'gpt-4o-mini-2024-07-18', temperature: 0.2, tools: ['format_markdown'] },
        ],
        justification: 'TCE predicted high domain complexity requiring specialized verifier and clinical lead roles.',
      },
      totals: {
        tokens: 14820,
        cost_usd: 0.0485,
        wall_seconds: 12.45,
        calls: 8,
        supervision_tokens: 1850,
        supervision_share: 0.125,
      },
      compliance: {
        tokens: true,
        cost: true,
        wall: true,
        calls: true,
        all: true,
      },
      quality: {
        composite: 0.892,
      },
      budget: {
        max_tokens: 25000,
        max_cost_usd: 0.12,
        max_wall_seconds: 60,
        max_calls: 15,
      },
      replacements: 1,
      replacements_denied: 0,
      final_output_ref: 'artifacts/01JRUN.md',
      created_at: new Date(Date.now() - 360000).toISOString(),
    };

    const demoSpend: SpendStatus = {
      cumulative_spend_usd: 42.15,
      spend_ceiling_usd: 250.0,
      kill_multiplier: 1.2,
      hard_ceiling_usd: 300.0,
      ceiling_breached: false,
    };

    const demoDecisions: DecisionRecord[] = [
      {
        decision_id: 'dec_01',
        run_id: runId,
        component: 'tce',
        decision: 'Team Sized to 3 Agents',
        inputs: { vocabulary_density: 0.74, sentence_depth: 4.2 },
        justification: 'High syntactic and vocabulary density prompted 3-role allocation: clinical_lead, statistician, synthesizer.',
        timestamp: new Date(Date.now() - 350000).toISOString(),
      },
      {
        decision_id: 'dec_02',
        run_id: runId,
        component: 'rbe',
        decision: 'Admission Approved (Call 1/15)',
        inputs: { tokens: 1200, cost: 0.0035 },
        justification: 'Projected consumption of 1,200 tokens is within the 25,000 budget ceiling.',
        timestamp: new Date(Date.now() - 320000).toISOString(),
      },
      {
        decision_id: 'dec_03',
        run_id: runId,
        component: 'rtpm',
        decision: 'L1 Cosine Flag Raised',
        inputs: { cosine_similarity: 0.61, threshold: 0.75 },
        justification: 'Output embedding similarity to task prompt dropped below threshold (0.61 < 0.75); triggering L2 judge rubric.',
        timestamp: new Date(Date.now() - 250000).toISOString(),
      },
      {
        decision_id: 'dec_04',
        run_id: runId,
        component: 'replacement',
        decision: 'Agent Swapped: synthesizer -> synthesizer_precise',
        inputs: { previous_agent: 'synthesizer', candidate: 'synthesizer_precise' },
        justification: 'Sustained low adherence in synthesizer. Swapped to precise template from ATL with context handoff.',
        timestamp: new Date(Date.now() - 210000).toISOString(),
      },
    ];

    const demoReplacements: ReplacementEvent[] = [
      {
        event_id: 'rep_01',
        run_id: runId,
        incumbent_agent: 'synthesizer',
        replacement_candidate: 'synthesizer_precise',
        trigger_reason: 'Two consecutive outputs below composite threshold 0.70',
        outcome: 'executed',
        justification: 'Agent Template Library (ATL) match found; handoff cost $0.004 within remaining margin.',
        timestamp: new Date(Date.now() - 210000).toISOString(),
      },
    ];

    const demoScores: ScoreRecord[] = [
      {
        score_id: 'sc_01',
        output_id: 'out_001',
        layer: 'l1_embed',
        composite: 0.88,
        timestamp: new Date(Date.now() - 300000).toISOString(),
      },
      {
        score_id: 'sc_02',
        output_id: 'out_002',
        layer: 'l1_embed',
        composite: 0.61,
        timestamp: new Date(Date.now() - 250000).toISOString(),
      },
      {
        score_id: 'sc_03',
        output_id: 'out_002',
        layer: 'l2_judge',
        composite: 0.64,
        timestamp: new Date(Date.now() - 245000).toISOString(),
      },
      {
        score_id: 'sc_04',
        output_id: 'out_003',
        layer: 'l3_verifier',
        composite: 0.95,
        timestamp: new Date(Date.now() - 190000).toISOString(),
      },
    ];

    setRun(demoRun);
    setSpend(demoSpend);
    setDecisions(demoDecisions);
    setReplacements(demoReplacements);
    setScores(demoScores);
  }, []);

  const loadRunData = useCallback(async (runId: string) => {
    setIsLoading(true);
    try {
      const [runData, decisionsData, spendData] = await Promise.all([
        fetchRun(runId).catch(() => null),
        fetchDecisions(runId).catch(() => []),
        fetchSpend().catch(() => null),
      ]);

      if (runData) {
        setRun(runData);
        setDecisions(decisionsData);
        if (spendData) setSpend(spendData);
      } else {
        // Fallback to demo data
        loadDemoData(runId);
      }
    } catch (err) {
      console.warn('API unavailable or run not found; loading viva demonstration mode.');
      loadDemoData(runId);
    } finally {
      setIsLoading(false);
    }
  }, [loadDemoData]);

  // Initial load
  useEffect(() => {
    loadRunData(currentRunId);
  }, [currentRunId, loadRunData]);

  // SSE Subscription for live monitoring (FR-25)
  useEffect(() => {
    if (!currentRunId) return;

    let unsubscribe: (() => void) | null = null;
    try {
      unsubscribe = subscribeToRunEvents(
        currentRunId,
        (type, data) => {
          setIsLiveConnected(true);
          if (type === 'budget_event') {
            setDecisions((prev) => [
              {
                decision_id: data.event_id || `dec_${Date.now()}`,
                run_id: currentRunId,
                component: 'rbe',
                decision: `Budget Action: ${data.action} (${data.dimension})`,
                inputs: data,
                justification: data.justification || 'Enforcing budget limit',
                timestamp: new Date().toISOString(),
              },
              ...prev,
            ]);
          } else if (type === 'replacement') {
            setReplacements((prev) => [
              {
                event_id: data.event_id || `rep_${Date.now()}`,
                run_id: currentRunId,
                outcome: data.outcome || 'executed',
                justification: data.justification || 'Dynamic agent substitution',
                timestamp: new Date().toISOString(),
              },
              ...prev,
            ]);
          } else if (type === 'score_computed') {
            setScores((prev) => [
              {
                score_id: data.score_id || `sc_${Date.now()}`,
                layer: data.layer || 'l1_embed',
                composite: data.composite || 0,
                timestamp: new Date().toISOString(),
              },
              ...prev,
            ]);
          } else if (type === 'run_completed') {
            setRun((prev) => (prev ? { ...prev, status: data.status } : null));
          }
        },
        () => {
          setIsLiveConnected(false);
        }
      );
    } catch (err) {
      setIsLiveConnected(false);
    }

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [currentRunId]);

  const handleCancelRun = async () => {
    if (!currentRunId) return;
    try {
      await cancelRun(currentRunId);
      setRun((prev) => (prev ? { ...prev, status: 'cancelled' } : null));
    } catch (err) {
      console.error('Cancel failed', err);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col font-sans">
      <Header
        run={run}
        spend={spend}
        runIdInput={runIdInput}
        setRunIdInput={setRunIdInput}
        onLoadRun={(id) => {
          setCurrentRunId(id);
          loadRunData(id);
        }}
        onCancelRun={handleCancelRun}
        isLoading={isLoading}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6 flex flex-col gap-6">
        {/* Navigation Tabs */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveTab('run')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 ${
                activeTab === 'run'
                  ? 'bg-sky-500/10 text-sky-400 border border-sky-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Active Run Monitor</span>
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 ${
                activeTab === 'benchmark'
                  ? 'bg-sky-500/10 text-sky-400 border border-sky-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart2 className="w-4 h-4" />
              <span>Ablation Matrix & Statistics</span>
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span
              className={`h-2 w-2 rounded-full ${
                isLiveConnected ? 'bg-emerald-400 animate-ping' : 'bg-slate-600'
              }`}
            />
            <span>{isLiveConnected ? 'SSE Stream: Connected' : 'SSE Stream: Replay / Offline'}</span>
          </div>
        </div>

        {activeTab === 'run' ? (
          <>
            {run && (
              <>
                {/* 4D Budget Enforcement Gauges (FR-5) */}
                <BudgetGauges run={run} />

                {/* Team Composition & Task Metadata */}
                <TeamComposition run={run} />

                {/* Middle Grid: Layered Scores & Replacement Timeline */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <LayeredScores scores={scores} finalCompositeScore={run.quality?.composite || undefined} />
                  <ReplacementTimeline replacements={replacements} />
                </div>

                {/* Explainability & Audit Log (FR-24) */}
                <ExplainabilityLog decisions={decisions} />
              </>
            )}
          </>
        ) : (
          <BenchmarkView />
        )}
      </main>

      <footer className="border-t border-slate-900 bg-[#0B0F19] px-6 py-4 text-center text-xs text-slate-500 font-mono">
        OASIS Dynamic Supervisory Layer • Strict 4D Budget Enforcement • Provable Replay Purity
      </footer>
    </div>
  );
};

export default App;
