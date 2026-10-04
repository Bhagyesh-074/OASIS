import { RunDetail, DecisionRecord, SpendStatus } from './types';

const API_BASE = ''; // Relative path, handled by Vite proxy in dev or Nginx in docker

export async function fetchRun(runId: string): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/v1/runs/${runId}`, {
    headers: {
      'X-API-Key': 'oasis-dev-key',
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch run: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchDecisions(runId: string): Promise<DecisionRecord[]> {
  const res = await fetch(`${API_BASE}/v1/runs/${runId}/decisions`, {
    headers: {
      'X-API-Key': 'oasis-dev-key',
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch decisions: ${res.statusText}`);
  }
  const data = await res.json();
  return data.decisions || [];
}

export async function fetchSpend(): Promise<SpendStatus> {
  const res = await fetch(`${API_BASE}/v1/spend`, {
    headers: {
      'X-API-Key': 'oasis-dev-key',
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch spend: ${res.statusText}`);
  }
  return res.json();
}

export async function cancelRun(runId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/runs/${runId}/cancel`, {
    method: 'POST',
    headers: {
      'X-API-Key': 'oasis-dev-key',
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to cancel run: ${res.statusText}`);
  }
}

export function subscribeToRunEvents(
  runId: string,
  onEvent: (type: string, data: any) => void,
  onError?: (err: any) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/v1/runs/${runId}/events`);

  eventSource.addEventListener('budget_event', (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent('budget_event', data);
    } catch (err) {
      console.error('Error parsing budget_event', err);
    }
  });

  eventSource.addEventListener('replacement', (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent('replacement', data);
    } catch (err) {
      console.error('Error parsing replacement event', err);
    }
  });

  eventSource.addEventListener('score_computed', (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent('score_computed', data);
    } catch (err) {
      console.error('Error parsing score_computed event', err);
    }
  });

  eventSource.addEventListener('run_completed', (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent('run_completed', data);
    } catch (err) {
      console.error('Error parsing run_completed event', err);
    }
  });

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      onEvent('message', data);
    } catch (err) {
      // heartbeats or plaintext
    }
  };

  eventSource.onerror = (err) => {
    if (onError) onError(err);
  };

  return () => {
    eventSource.close();
  };
}
