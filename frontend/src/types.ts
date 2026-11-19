export type RunStatus = 'queued' | 'running' | 'completed' | 'halted_budget' | 'failed' | 'cancelled';

export interface RoleConfig {
  role: string;
  template_id: string;
  model: string;
  temperature: number;
  est_tokens: number;
  est_cost_usd: number;
  est_wall_seconds: number;
  est_calls: number;
}

export interface TeamConfig {
  config_id: string;
  team_size: number;
  origin: 'estimator' | 'memory' | 'baseline' | 'oracle';
  topology: 'linear' | 'hub_and_spoke' | 'parallel';
  roles: RoleConfig[];
  justification: string;
}

export interface BudgetLimits {
  max_tokens: number;
  max_cost_usd: number;
  max_wall_seconds: number;
  max_calls: number;
}

export interface RunTotals {
  tokens: number;
  cost_usd: number;
  wall_seconds: number;
  calls: number;
  supervision_tokens: number;
  supervision_share: number;
}

export interface RunCompliance {
  tokens: boolean;
  cost: boolean;
  wall: boolean;
  calls: boolean;
  all: boolean;
}

export interface RunQuality {
  composite?: number | null;
  l3_verifier?: number | null;
  l2_judge?: number | null;
  judge_model?: string;
  rubric_version?: string;
}

export interface RunDetail {
  run_id: string;
  task_id?: string | null;
  task_statement?: string | null;
  status: RunStatus;
  status_detail?: string | null;
  arm?: string;
  seed?: number;
  framework?: string;
  complexity_tier?: string;
  config?: TeamConfig | null;
  totals: RunTotals;
  compliance: RunCompliance;
  quality: RunQuality;
  budget?: BudgetLimits | null;
  replacements: number;
  replacements_denied: number;
  final_output_ref?: string | null;
  created_at?: string;
}

export interface DecisionRecord {
  decision_id: string;
  run_id?: string;
  ts?: string;
  timestamp?: string;
  component: string;
  decision: string;
  inputs?: Record<string, any>;
  justification: string;
}

export interface ReplacementEvent {
  event_id: string;
  run_id?: string;
  agent_instance_id?: string;
  incumbent_agent?: string;
  replacement_candidate?: string;
  trigger_reason?: string;
  outcome: 'executed' | 'denied_budget' | 'denied_cap' | 'denied_adapter';
  justification: string;
  timestamp?: string;
}

export interface ScoreRecord {
  score_id: string;
  output_id?: string;
  layer: 'l1_embed' | 'l2_judge' | 'l3_verifier';
  composite: number | null;
  timestamp?: string;
}

export interface RunEventData {
  event_id?: string;
  action?: string;
  dimension?: string;
  justification?: string;
  outcome?: ReplacementEvent['outcome'];
  incumbent_agent?: string;
  replacement_candidate?: string;
  trigger_reason?: string;
  score_id?: string;
  output_id?: string;
  layer?: ScoreRecord['layer'];
  composite?: number | null;
  status?: RunStatus;
}

export interface SpendStatus {
  cumulative_spend_usd: number;
  ceiling_usd: number;
  kill_multiplier: number;
  kill_ceiling_usd: number;
  ceiling_exceeded: boolean;
}

export interface BenchmarkArmSummary {
  total_runs: number;
  quality: { median: number | null; ci_95: [number | null, number | null] };
  cost_usd: { median: number; ci_95: [number, number] };
  tokens: { median: number; ci_95: [number, number] };
  compliance_rate: number;
  wilcoxon_p_value: number | null;
  effect_size: number | null;
  holm_adjusted_p_value: number | null;
}

export interface BenchmarkSummary {
  reference_arm: string;
  arms: Record<string, BenchmarkArmSummary>;
}
