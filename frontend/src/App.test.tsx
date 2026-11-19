import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import App from './App';

const runId = '01JRUN00000000000000000001';

const runResponse = {
  run_id: runId,
  task_id: 'task_42',
  task_statement: 'Synthesize recent results and report the key differences.',
  status: 'completed',
  arm: 'full',
  config: {
    config_id: 'cfg_01J999',
    team_size: 1,
    origin: 'estimator',
    topology: 'hub_and_spoke',
    roles: [
      {
        role: 'researcher',
        template_id: 'fast_cheap',
        model: 'gpt-4o-mini',
        temperature: 0.1,
        est_tokens: 1000,
        est_cost_usd: 0.001,
        est_wall_seconds: 5,
        est_calls: 1,
      },
    ],
    justification: 'Configured for the requested task.',
  },
  budget: {
    max_tokens: 25000,
    max_cost_usd: 0.12,
    max_wall_seconds: 60,
    max_calls: 15,
  },
  totals: {
    tokens: 14820,
    cost_usd: 0.0485,
    wall_seconds: 12.45,
    calls: 8,
    supervision_tokens: 1850,
    supervision_share: 0.125,
  },
  compliance: { tokens: true, cost: true, wall: true, calls: true, all: true },
  quality: { composite: null },
  replacements: 0,
  replacements_denied: 0,
};

const benchmarkSummary = {
  reference_arm: 'vanilla_fixed',
  arms: {
    full: {
      total_runs: 2,
      quality: { median: null, ci_95: [null, null] },
      cost_usd: { median: 0.05, ci_95: [0.04, 0.06] },
      tokens: { median: 12000, ci_95: [10000, 14000] },
      compliance_rate: 1,
      wilcoxon_p_value: null,
      effect_size: null,
      holm_adjusted_p_value: null,
    },
    vanilla_fixed: {
      total_runs: 2,
      quality: { median: null, ci_95: [null, null] },
      cost_usd: { median: 0.08, ci_95: [0.07, 0.09] },
      tokens: { median: 15000, ci_95: [14000, 16000] },
      compliance_rate: 0.5,
      wilcoxon_p_value: null,
      effect_size: null,
      holm_adjusted_p_value: null,
    },
  },
};

describe('OASIS React Dashboard (FR-25)', () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/events')) {
          return Promise.resolve(
            new Response(
              new ReadableStream({
                start(controller) {
                  controller.enqueue(new TextEncoder().encode('event: heartbeat\ndata: {}\n\n'));
                  controller.close();
                },
              }),
            ),
          );
        }
        if (url.includes('/v1/spend')) {
          return Promise.resolve(
            Response.json({
              cumulative_spend_usd: 15.2,
              ceiling_usd: 250,
              kill_multiplier: 1.2,
              kill_ceiling_usd: 300,
              ceiling_exceeded: false,
            }),
          );
        }
        if (url.includes('/decisions')) {
          return Promise.resolve(
            Response.json({
              run_id: runId,
              decisions: [
                {
                  decision_id: 'dec_01',
                  component: 'tce',
                  decision: 'Team Sized',
                  justification: 'Configuration recorded by the API.',
                  ts: new Date().toISOString(),
                },
              ],
            }),
          );
        }
        if (url.includes('/v1/metrics/summary')) {
          return Promise.resolve(Response.json(benchmarkSummary));
        }
        if (url.includes('/v1/runs/')) {
          return Promise.resolve(Response.json(runResponse));
        }
        return Promise.reject(new Error(`Unknown endpoint: ${url}`));
      }),
    );
  });

  it('loads the requested run and displays API-backed budget and decision data', async () => {
    render(<App />);

    expect(screen.getByText('OASIS')).toBeInTheDocument();
    fireEvent.change(screen.getByRole('textbox', { name: 'Run ID to inspect' }), { target: { value: runId } });
    fireEvent.click(screen.getByRole('button', { name: /Inspect run/ }));

    expect(await screen.findByText('Run overview')).toBeInTheDocument();
    expect(screen.getByText('Resource guardrails')).toBeInTheDocument();
    expect(screen.getAllByText(/25,000/).length).toBeGreaterThan(0);
    expect(screen.getAllByText('Synthesize recent results and report the key differences.').length).toBeGreaterThan(0);
    expect(await screen.findByText(/Explainability & Decision Audit Log/i)).toBeInTheDocument();
    expect(screen.getByText('researcher')).toBeInTheDocument();
  });

  it('displays API benchmark summaries instead of hardcoded benchmark results', async () => {
    render(<App />);
    fireEvent.click(screen.getByText('Benchmark analysis'));

    expect(await screen.findByText('Experimental arms')).toBeInTheDocument();
    expect(screen.getAllByText('OASIS Full').length).toBeGreaterThan(0);
    expect(screen.getByText('Vanilla Fixed')).toBeInTheDocument();
    expect(screen.getByText('50.0%')).toBeInTheDocument();
    expect(screen.queryByText('0.882')).not.toBeInTheDocument();
  });

  it('shows run load errors without substituting demo data', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/v1/runs/') && !url.endsWith('/events')) {
          return Promise.resolve(new Response('not found', { status: 404, statusText: 'Not Found' }));
        }
        if (url.includes('/decisions')) {
          return Promise.resolve(new Response('not found', { status: 404 }));
        }
        if (url.includes('/v1/spend')) {
          return Promise.resolve(
            Response.json({
              cumulative_spend_usd: 0,
              ceiling_usd: 250,
              kill_multiplier: 1.2,
              kill_ceiling_usd: 300,
              ceiling_exceeded: false,
            }),
          );
        }
        return Promise.resolve(new Response('', { status: 200 }));
      }),
    );
    render(<App />);
    fireEvent.change(screen.getByRole('textbox', { name: 'Run ID to inspect' }), { target: { value: runId } });
    fireEvent.click(screen.getByRole('button', { name: /Inspect run/ }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/Fetching run failed \(404/);
    await waitFor(() => expect(screen.queryByText('Run overview')).not.toBeInTheDocument());
  });
});
