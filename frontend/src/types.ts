export type RunStatus = 'queued' | 'running' | 'completed' | 'halted_budget' | 'failed' | 'cancelled';

export interface RoleConfig {
  role_name: string;
  model: string;
  temperature: number;
  tools: string[];
}

export interface TeamConfig {
  config_id: string;
  team_size: number;
  origin: 'tce' | 'cm' | 'reduced';
  topology: 'hub_and_spoke' | 'pipeline' | 'hierarchical';
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
  benchmark_task_id?: string;
  status: RunStatus;
  status_detail?: string | null;
  arm?: string;
  seed?: number;
  framework?: string;
  task_statement?: string;
  complexity_tier?: string;
  config?: TeamConfig | null;
  totals: RunTotals;
  compliance: RunCompliance;
  quality: RunQuality;
  budget?: BudgetLimits;
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
  composite: number;
  timestamp?: string;
}

export interface SpendStatus {
  cumulative_spend_usd: number;
  spend_ceiling_usd: number;
  kill_multiplier: number;
  hard_ceiling_usd: number;
  ceiling_breached: boolean;
  calls_count?: number;
  updated_at?: string;
}
