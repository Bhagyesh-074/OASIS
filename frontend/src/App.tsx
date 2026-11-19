import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Header } from './components/Header';
import { BudgetGauges } from './components/BudgetGauges';
import { TeamComposition } from './components/TeamComposition';
import { LayeredScores } from './components/LayeredScores';
import { ReplacementTimeline } from './components/ReplacementTimeline';
import { ExplainabilityLog } from './components/ExplainabilityLog';
import { BenchmarkView } from './components/BenchmarkView';
import { RunOverview } from './components/RunOverview';
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
import { Activity, BarChart2, Radio, Sparkles } from 'lucide-react';

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

  const loadRunData = useCallback(async (runId: string, preserveLiveEvents = false) => {
    if (!runId) return;
    const sequence = ++loadSequence.current;
    setIsLoading(true);
    setRunError(null);
    setSpendError(null);
    if (!preserveLiveEvents) setRun(null);
    setDecisions([]);
    if (!preserveLiveEvents) {
      setReplacements([]);
      setScores([]);
    }
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
      void loadRunData(currentRunId, true);
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
    <div className="relative min-h-screen overflow-hidden bg-[#080c14] font-sans text-slate-100">
      <div className="pointer-events-none fixed inset-0 -z-0 bg-[radial-gradient(ellipse_at_50%_-20%,rgba(14,165,233,0.12),transparent_50%),radial-gradient(ellipse_at_100%_50%,rgba(99,102,241,0.06),transparent_40%)]" />
      <Header
        run={run}
        spend={spend}
        runIdInput={runIdInput}
        setRunIdInput={setRunIdInput}
        onLoadRun={handleLoadRun}
        onCancelRun={handleCancelRun}
        isLoading={isLoading}
      />

      <main className="relative z-10 mx-auto flex w-full max-w-[1500px] flex-1 flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
        <section className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
          <div>
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-sky-300/70">
              <Sparkles className="h-3.5 w-3.5" />
              Observability workspace
            </div>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
              Run intelligence, at a glance.
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Follow resource consumption, team configuration, and decision provenance across your agent runs.
            </p>
          </div>
          <div
            className={`flex w-fit items-center gap-2 rounded-full border px-3 py-2 text-[11px] font-medium ${
              isLiveConnected
                ? 'border-emerald-300/15 bg-emerald-300/[0.06] text-emerald-200'
                : 'border-white/[0.07] bg-white/[0.025] text-slate-400'
            }`}
            aria-live="polite"
          >
            <Radio className={`h-3.5 w-3.5 ${isLiveConnected ? 'text-emerald-300' : 'text-slate-500'}`} />
            {isLiveConnected ? 'Live event stream connected' : 'Waiting for a run stream'}
          </div>
        </section>

        <div className="flex flex-col justify-between gap-3 border-b border-white/[0.07] pb-3 sm:flex-row sm:items-center">
          <div className="flex items-center gap-1 rounded-xl border border-white/[0.06] bg-white/[0.025] p-1">
            <button
              onClick={() => setActiveTab('run')}
              className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                activeTab === 'run'
                  ? 'border border-sky-300/10 bg-sky-300/[0.08] text-sky-100 shadow-sm'
                  : 'border border-transparent text-slate-500 hover:text-slate-200'
              }`}
            >
              <Activity className="h-3.5 w-3.5" />
              <span>Run monitor</span>
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition ${
                activeTab === 'benchmark'
                  ? 'border border-sky-300/10 bg-sky-300/[0.08] text-sky-100 shadow-sm'
                  : 'border border-transparent text-slate-500 hover:text-slate-200'
              }`}
            >
              <BarChart2 className="h-3.5 w-3.5" />
              <span>Benchmark analysis</span>
            </button>
          </div>

        </div>

        {activeTab === 'run' ? (
          <>
            {runError && (
              <div role="alert" className="rounded-xl border border-rose-400/15 bg-rose-400/[0.06] p-4 text-sm text-rose-200">
                {runError}
              </div>
            )}
            {spendError && (
              <div role="alert" className="rounded-xl border border-amber-400/15 bg-amber-400/[0.05] p-4 text-sm text-amber-200">
                Global spend unavailable: {spendError}
              </div>
            )}
            {run ? (
              <>
                <RunOverview run={run} />
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
              <div className="relative overflow-hidden rounded-2xl border border-white/[0.07] bg-gradient-to-br from-[#111b2c] via-[#0f1725] to-[#0d121d] p-6 sm:p-8">
                <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-indigo-400/[0.07] blur-3xl" />
                <div className="relative grid gap-8 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
                  <div>
                    <div className="inline-flex items-center gap-2 rounded-full border border-sky-300/10 bg-sky-300/[0.06] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.15em] text-sky-200/80">
                      <Activity className="h-3.5 w-3.5" />
                      Run workspace
                    </div>
                    <h2 className="mt-4 max-w-xl text-2xl font-semibold tracking-tight text-white sm:text-3xl">
                      Your agents are doing the work. See how.
                    </h2>
                    <p className="mt-3 max-w-xl text-sm leading-6 text-slate-400">
                      Load an execution by its run ID to inspect its task, team setup, four budget dimensions,
                      quality scores, and the decisions made along the way.
                    </p>
                    <form
                      onSubmit={(event) => {
                        event.preventDefault();
                        if (runIdInput.trim()) handleLoadRun(runIdInput.trim());
                      }}
                      className="mt-5 flex max-w-lg flex-col gap-2 sm:flex-row"
                    >
                      <input
                        aria-label="Run ID to inspect"
                        value={runIdInput}
                        onChange={(event) => setRunIdInput(event.target.value)}
                        placeholder="Paste a run ID to get started"
                        className="min-w-0 flex-1 rounded-lg border border-white/[0.09] bg-slate-950/70 px-3.5 py-3 text-xs font-mono text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-sky-400/50 focus:ring-2 focus:ring-sky-400/10"
                      />
                      <button
                        type="submit"
                        disabled={isLoading || !runIdInput.trim()}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-sky-300/20 bg-sky-400/10 px-4 py-3 text-xs font-semibold text-sky-100 transition hover:bg-sky-400/15 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {isLoading ? <Radio className="h-4 w-4 animate-pulse" /> : <Activity className="h-4 w-4" />}
                        {isLoading ? 'Loading run' : 'Inspect run'}
                      </button>
                    </form>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
                    {[
                      ['01', 'Execution context', 'Task, arm, framework, and selected team'],
                      ['02', 'Resource guardrails', 'Tokens, cost, wall time, and call limits'],
                      ['03', 'Decision audit', 'Scores, replacements, and justifications'],
                    ].map(([step, title, detail]) => (
                      <div key={step} className="flex items-start gap-3 rounded-xl border border-white/[0.06] bg-slate-950/25 p-3.5">
                        <span className="font-mono text-xs text-sky-300/70">{step}</span>
                        <div>
                          <div className="text-xs font-semibold text-slate-200">{title}</div>
                          <div className="mt-1 text-[11px] leading-5 text-slate-500">{detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
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

      <footer className="relative z-10 mt-8 border-t border-white/[0.06] px-4 py-4 text-center text-[10px] uppercase tracking-[0.14em] text-slate-600 sm:px-6">
        OASIS <span className="mx-2 text-slate-800">/</span> Execution observability
        <span className="mx-2 text-slate-800">/</span> Four-dimension budget monitoring
      </footer>
    </div>
  );
};

export default App;
