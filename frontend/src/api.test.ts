import { afterEach, describe, expect, it, vi } from 'vitest';
import { subscribeToRunEvents } from './api';

describe('run event stream client', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('authenticates the SSE fetch and parses CRLF-delimited events', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('event: budget_event\r\ndata: {"action":"admit"}\r\n\r\n'),
    );
    vi.stubGlobal('fetch', fetchMock);
    const onEvent = vi.fn();
    const onConnected = vi.fn();
    const unsubscribe = subscribeToRunEvents('run-1', onEvent, vi.fn(), onConnected);

    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledWith('budget_event', { action: 'admit' }));
    expect(onConnected).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      '/v1/runs/run-1/events',
      expect.objectContaining({
        headers: { 'X-API-Key': 'dev-local-key', Accept: 'text/event-stream' },
      }),
    );
    unsubscribe();
  });
});
