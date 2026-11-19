import { BenchmarkSummary, DecisionRecord, RunDetail, RunEventData, SpendStatus } from './types';

const API_BASE = ''; // Relative path, handled by Vite proxy in dev or Nginx in docker
const API_KEY = import.meta.env.VITE_OASIS_API_KEY || 'dev-local-key';
const API_HEADERS = { 'X-API-Key': API_KEY };

async function assertOk(res: Response, action: string): Promise<void> {
  if (!res.ok) {
    throw new Error(`${action} failed (${res.status}${res.statusText ? ` ${res.statusText}` : ''})`);
  }
}

export async function fetchRun(runId: string): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/v1/runs/${runId}`, {
    headers: API_HEADERS,
  });
  await assertOk(res, 'Fetching run');
  return res.json();
}

export async function fetchDecisions(runId: string): Promise<DecisionRecord[]> {
  const res = await fetch(`${API_BASE}/v1/runs/${runId}/decisions`, {
    headers: API_HEADERS,
  });
  await assertOk(res, 'Fetching decisions');
  const data = await res.json();
  return data.decisions;
}

export async function fetchSpend(): Promise<SpendStatus> {
  const res = await fetch(`${API_BASE}/v1/spend`, {
    headers: API_HEADERS,
  });
  await assertOk(res, 'Fetching spend');
  return res.json();
}

export async function fetchMetricsSummary(): Promise<BenchmarkSummary> {
  const res = await fetch(`${API_BASE}/v1/metrics/summary`, {
    headers: API_HEADERS,
  });
  await assertOk(res, 'Fetching benchmark summary');
  return res.json();
}

export async function cancelRun(runId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/runs/${runId}/cancel`, {
    method: 'POST',
    headers: API_HEADERS,
  });
  await assertOk(res, 'Cancelling run');
}

export function subscribeToRunEvents(
  runId: string,
  onEvent: (type: string, data: RunEventData) => void,
  onError?: (err: Error) => void,
  onConnected?: () => void
): () => void {
  const controller = new AbortController();
  void (async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/runs/${runId}/events`, {
        headers: { ...API_HEADERS, Accept: 'text/event-stream' },
        signal: controller.signal,
      });
      await assertOk(res, 'Connecting to run event stream');
      if (!res.body) throw new Error('Run event stream response has no body');
      onConnected?.();

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let separator = /\r?\n\r?\n/.exec(buffer);
        while (separator) {
          const block = buffer.slice(0, separator.index);
          buffer = buffer.slice(separator.index + separator[0].length);
          const lines = block.split(/\r?\n/);
          const eventType = lines.find((line) => line.startsWith('event:'))?.slice(6).trim() || 'message';
          const data = lines
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trimStart())
            .join('\n');
          if (data) onEvent(eventType, JSON.parse(data));
          separator = /\r?\n\r?\n/.exec(buffer);
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    }
  })();

  return () => controller.abort();
}
