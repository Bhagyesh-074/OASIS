import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from './App';

describe('OASIS React Dashboard (FR-25)', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/v1/spend')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              cumulative_spend_usd: 15.2,
              spend_ceiling_usd: 250.0,
              kill_multiplier: 1.2,
              hard_ceiling_usd: 300.0,
              ceiling_breached: false,
            }),
          });
        }
        if (url.includes('/decisions')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              run_id: '01JRUN00000000000000000001',
              decisions: [
                {
                  decision_id: 'dec_01',
                  component: 'tce',
                  decision: 'Team Sized to 3 Agents',
                  justification: 'High syntactic complexity.',
                  timestamp: new Date().toISOString(),
                },
              ],
            }),
          });
        }
        if (url.includes('/v1/runs/')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              run_id: '01JRUN00000000000000000001',
              status: 'completed',
              task_statement: 'Synthesize comparative clinical trial outcomes for SGLT2 inhibitors.',
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
                  { role_name: 'clinical_lead', model: 'gpt-4o-2024-11-20', temperature: 0.1, tools: ['search'] },
                  { role_name: 'statistician', model: 'gpt-4o-mini-2024-07-18', temperature: 0.0, tools: ['calc'] },
                ],
                justification: 'High domain complexity.',
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
            }),
          });
        }
        return Promise.reject(new Error(`Unknown endpoint: ${url}`));
      })
    );
  });

  it('renders the dashboard with header, spend ceiling, and 4D budget indicators', async () => {
    render(<App />);

    // Brand and header
    expect(screen.getByText('OASIS')).toBeInTheDocument();
    expect(screen.getByText('Dynamic Supervisory Layer for Multi-Agent LLMs')).toBeInTheDocument();

    // 4D Budget Enforcement panel
    expect(await screen.findByText(/4D Real-Time Budget Enforcement/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Tokens/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Cost/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Wall Time/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Calls/i).length).toBeGreaterThan(0);

    // Explainability Log
    expect(await screen.findByText(/Explainability & Decision Audit Log/i)).toBeInTheDocument();
  });

  it('switches to the ablation matrix tab and displays statistical comparisons', async () => {
    render(<App />);

    const benchmarkTabBtn = screen.getByText('Ablation Matrix & Statistics');
    fireEvent.click(benchmarkTabBtn);

    await waitFor(() => {
      expect(screen.getByText('8-Configuration Ablation Matrix Summary')).toBeInTheDocument();
      expect(screen.getByText(/OASIS Full \(Ours\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Unconstrained Baseline/i)).toBeInTheDocument();
    });
  });
});
