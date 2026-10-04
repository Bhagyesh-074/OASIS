import '@testing-library/jest-dom';

// Mock EventSource for Vitest jsdom
class MockEventSource {
  url: string;
  listeners: Record<string, ((event: any) => void)[]> = {};
  onmessage: ((event: any) => void) | null = null;
  onerror: ((event: any) => void) | null = null;

  constructor(url: string) {
    this.url = url;
  }

  addEventListener(type: string, listener: (event: any) => void) {
    if (!this.listeners[type]) {
      this.listeners[type] = [];
    }
    this.listeners[type].push(listener);
  }

  removeEventListener(type: string, listener: (event: any) => void) {
    if (this.listeners[type]) {
      this.listeners[type] = this.listeners[type].filter((l) => l !== listener);
    }
  }

  close() {}
}

(globalThis as any).EventSource = MockEventSource;
