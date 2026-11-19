import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Header } from './components/Header';
import { BudgetGauges } from './components/BudgetGauges';
import { TeamComposition } from './components/TeamComposition';
import { LayeredScores } from './components/LayeredScores';
import { ReplacementTimeline } from './components/ReplacementTimeline';
import { ExplainabilityLog } from './components/ExplainabilityLog';
import { BenchmarkView } from './components/BenchmarkView';
import {
  BenchmarkSummary,
  DecisionRecord,
  ReplacementEvent,
  RunDetail,
  RunEventData,
  ScoreRecord,
  SpendStatus,
} from './types';
import {
  cancelRun,
  fetchDecisions,
  fetchMetricsSummary,
  fetchRun,
  fetchSpend,
  subscribeToRunEvents,
} from './api';
import { Activity, BarChart2 } from 'lucide-react';

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'run' | 'benchmark'>('run');
  const [runIdInput, setRunIdInput] = useState('');
  const [currentRunId, setCurrentRunId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLiveConnected, setIsLiveConnected] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [spendError, setSpendError] = useState<string | null>(null);
  const [benchmarkSummary, setBenchmarkSummary] = useState<BenchmarkSummary | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);

  const [run, setRun] = useState<RunDetail | null>(null);
  const [spend, setSpend] = useState<SpendStatus | null>(null);
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [replacements, setReplacements] = useState<ReplacementEvent[]>([]);
  const [scores, setScores] = useState<ScoreRecord[]>([]);
  const loadSequence = useRef(0);

  const loadRunData = useCallback(async (runId: string) => {
    if (!runId) return;
    const sequence = ++loadSequence.current;
    setIsLoading(true);
    setRunError(null);
    setSpendError(null);
    setRun(null);
    setDecisions([]);
    setReplacements([]);
    setScores([]);
    const [runResult, decisionsResult, spendResult] = await Promise.allSettled([
      fetchRun(runId),
      fetchDecisions(runId),
      fetchSpend(),
    ]);
    if (sequence !== loadSequence.current) return;

    if (runResult.status === 'fulfilled') {
      setRun(runResult.value);
    } else {
      setRun(null);
      setRunError(errorMessage(runResult.reason));
    }

    if (decisionsResult.status === 'fulfilled') {
      setDecisions(decisionsResult.value);
    } else {
      setDecisions([]);
      setRunError((previous) =>
        [previous, `Loading decisions: ${errorMessage(decisionsResult.reason)}`]
          .filter(Boolean)
          .join(' '),
      );
    }

    if (spendResult.status === 'fulfilled') {
      setSpend(spendResult.value);
    } else {
      setSpend(null);
      setSpendError(errorMessage(spendResult.reason));
    }

    setIsLoading(false);
  }, []);

  useEffect(() => {
    if (currentRunId) {
      void loadRunData(currentRunId);
    } else {
      fetchSpend()
        .then(setSpend)
        .catch((error: unknown) => setSpendError(errorMessage(error)));
    }
  }, [currentRunId, loadRunData]);

  useEffect(() => {
    if (!currentRunId) return;

    return subscribeToRunEvents(
      currentRunId,
      (type, data: RunEventData) => {
        setIsLiveConnected(true);
        if (type === 'budget_event') {
          setDecisions((previous) => [
            {
              decision_id: data.event_id || `dec_${Date.now()}`,
              run_id: currentRunId,
              component: 'rbe',
              decision: `Budget Action: ${data.action ?? 'unknown'} (${data.dimension ?? 'unknown'})`,
              inputs: data,
              justification: data.justification || 'Budget event recorded.',
              ts: new Date().toISOString(),
            },
            ...previous,
          ]);
        } else if (type === 'replacement' || type === 'replacement_denied') {
          setReplacements((previous) => [
            {
              event_id: data.event_id || `rep_${Date.now()}`,
              run_id: currentRunId,
              incumbent_agent: data.incumbent_agent,
              replacement_candidate: data.replacement_candidate,
              trigger_reason: data.trigger_reason,
              outcome: data.outcome || (type === 'replacement_denied' ? 'denied_adapter' : 'executed'),
              justification: data.justification || 'No justification was provided.',
              timestamp: new Date().toISOString(),
            },
            ...previous,
          ]);
        } else if (type === 'score_computed' && data.layer) {
          const layer = data.layer;
          setScores((previous) => [
            {
              score_id: data.score_id || `sc_${Date.now()}`,
              output_id: data.output_id,
              layer,
              composite: data.composite ?? null,
              timestamp: new Date().toISOString(),
            },
            ...previous,
          ]);
        } else if (type === 'run_completed') {
          setIsLiveConnected(false);
          if (data.status) {
            setRun((previous) => (previous ? { ...previous, status: data.status! } : null));
          }
          void loadRunData(currentRunId);
        }
      },
      (error) => {
        setIsLiveConnected(false);
        setRunError(`Run event stream: ${error.message}`);
      },
      () => setIsLiveConnected(true),
    );
  }, [currentRunId, loadRunData]);

  useEffect(() => {
    if (activeTab !== 'benchmark') return;

    let active = true;
    setBenchmarkLoading(true);
    setBenchmarkError(null);
    fetchMetricsSummary()
      .then((summary) => {
        if (active) setBenchmarkSummary(summary);
      })
      .catch((error: unknown) => {
        if (active) setBenchmarkError(errorMessage(error));
      })
      .finally(() => {
        if (active) setBenchmarkLoading(false);
      });

    return () => {
      active = false;
    };
  }, [activeTab]);

  const handleCancelRun = async () => {
    if (!currentRunId) return;
    setRunError(null);
    try {
      await cancelRun(currentRunId);
      setRun((previous) => (previous ? { ...previous, status: 'cancelled' } : null));
    } catch (error) {
      setRunError(`Cancelling run: ${errorMessage(error)}`);
    }
  };

  const handleLoadRun = (runId: string) => {
    setRunIdInput(runId);
    if (runId === currentRunId) {
      void loadRunData(runId);
    } else {
      setCurrentRunId(runId);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col font-sans">
      <Header
        run={run}
        spend={spend}
        runIdInput={runIdInput}
        setRunIdInput={setRunIdInput}
        onLoadRun={handleLoadRun}
        onCancelRun={handleCancelRun}
        isLoading={isLoading}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6 flex flex-col gap-6">
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
              <span>Ablation Matrix &amp; Statistics</span>
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className={`h-2 w-2 rounded-full ${isLiveConnected ? 'bg-emerald-400 animate-ping' : 'bg-slate-600'}`} />
            <span>{isLiveConnected ? 'Run event stream: Connected' : 'Run event stream: Disconnected'}</span>
          </div>
        </div>

        {activeTab === 'run' ? (
          <>
            {runError && (
              <div role="alert" className="rounded-lg border border-rose-800 bg-rose-950/30 p-4 text-sm text-rose-300">
                {runError}
              </div>
            )}
            {spendError && (
              <div role="alert" className="rounded-lg border border-amber-800 bg-amber-950/30 p-4 text-sm text-amber-300">
                Global spend unavailable: {spendError}
              </div>
            )}
            {run ? (
              <>
                <BudgetGauges run={run} />
                <TeamComposition run={run} />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <LayeredScores
                    scores={scores}
                    finalCompositeScore={run.quality?.composite ?? undefined}
                  />
                  <ReplacementTimeline replacements={replacements} />
                </div>
                <ExplainabilityLog decisions={decisions} />
              </>
            ) : (
              <div className="rounded-xl border border-slate-800 bg-[#131B2E] p-8 text-center text-sm text-slate-400">
                {isLoading ? 'Loading run…' : 'Enter a run ID above to load its live status and audit data.'}
              </div>
            )}
          </>
        ) : (
          <BenchmarkView
            summary={benchmarkSummary}
            isLoading={benchmarkLoading}
            error={benchmarkError}
          />
        )}
      </main>

      <footer className="border-t border-slate-900 bg-[#0B0F19] px-6 py-4 text-center text-xs text-slate-500 font-mono">
        OASIS Dynamic Supervisory Layer • Four-Dimension Budget Monitoring • Replay-Aware Evaluation
      </footer>
    </div>
  );
};

export default App;
