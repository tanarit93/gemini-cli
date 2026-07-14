/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { OpenAiContentGenerator } from './openAiContentGenerator.js';
import { fetchWithTimeout } from '../utils/fetch.js';
import type { GenerateContentParameters } from '@google/genai';
import { LlmRole } from '../telemetry/llmRole.js';

vi.mock('../utils/fetch.js', () => ({
  fetchWithTimeout: vi.fn(),
}));

describe('OpenAiContentGenerator - OpenRouter provider configuration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv('OPENROUTER_PROVIDER_ORDER', '');
    vi.stubEnv('OPENROUTER_ALLOW_FALLBACKS', '');
    vi.stubEnv('OPENROUTER_PROVIDER_IGNORE', '');
    vi.stubEnv('OPENROUTER_PROVIDER_SKIPS', '');
    vi.stubEnv('OPENROUTER_PROVIDER_QUANTIZATIONS', '');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  const dummyRequest: GenerateContentParameters = {
    model: 'meta-llama/llama-3-70b-instruct',
    contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
  };

  it('should not include provider field by default when baseUrl is openrouter.ai', async () => {
    const generator = new OpenAiContentGenerator({
      apiKey: 'test-key',
      baseUrl: 'https://openrouter.ai/api/v1',
    });

    vi.mocked(fetchWithTimeout).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Hi' } }],
      }),
    } as unknown as Response);

    await generator.generateContent(dummyRequest, 'prompt-id', LlmRole.MAIN);

    expect(fetchWithTimeout).toHaveBeenCalledWith(
      expect.any(String),
      expect.any(Number),
      expect.objectContaining({
        body: expect.stringContaining(
          '"model":"meta-llama/llama-3-70b-instruct"',
        ),
      }),
    );

    const callArgs = vi.mocked(fetchWithTimeout).mock.calls[0];
    const bodyStr = callArgs && callArgs[2] ? callArgs[2].body : undefined;
    const bodyObj = JSON.parse((bodyStr || '{}') as string);
    expect(bodyObj.provider).toBeUndefined();
  });

  it('should include provider.order when OPENROUTER_PROVIDER_ORDER is set', async () => {
    vi.stubEnv('OPENROUTER_PROVIDER_ORDER', 'Together, DeepInfra,  Lepton');
    const generator = new OpenAiContentGenerator({
      apiKey: 'test-key',
      baseUrl: 'https://openrouter.ai/api/v1',
    });

    vi.mocked(fetchWithTimeout).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Hi' } }],
      }),
    } as unknown as Response);

    await generator.generateContent(dummyRequest, 'prompt-id', LlmRole.MAIN);

    const callArgs = vi.mocked(fetchWithTimeout).mock.calls[0];
    const bodyStr = callArgs && callArgs[2] ? callArgs[2].body : undefined;
    const bodyObj = JSON.parse((bodyStr || '{}') as string);
    expect(bodyObj.provider).toEqual({
      order: ['Together', 'DeepInfra', 'Lepton'],
    });
  });

  it('should include provider.allow_fallbacks when OPENROUTER_ALLOW_FALLBACKS is set', async () => {
    vi.stubEnv('OPENROUTER_ALLOW_FALLBACKS', 'false');
    const generator = new OpenAiContentGenerator({
      apiKey: 'test-key',
      baseUrl: 'https://openrouter.ai/api/v1',
    });

    vi.mocked(fetchWithTimeout).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Hi' } }],
      }),
    } as unknown as Response);

    await generator.generateContent(dummyRequest, 'prompt-id', LlmRole.MAIN);

    const callArgs = vi.mocked(fetchWithTimeout).mock.calls[0];
    const bodyStr = callArgs && callArgs[2] ? callArgs[2].body : undefined;
    const bodyObj = JSON.parse((bodyStr || '{}') as string);
    expect(bodyObj.provider).toEqual({
      allow_fallbacks: false,
    });
  });

  it('should include ignore and quantizations when env variables are set', async () => {
    vi.stubEnv('OPENROUTER_PROVIDER_IGNORE', 'Together');
    vi.stubEnv('OPENROUTER_PROVIDER_QUANTIZATIONS', 'int4, bf16');
    const generator = new OpenAiContentGenerator({
      apiKey: 'test-key',
      baseUrl: 'https://openrouter.ai/api/v1',
    });

    vi.mocked(fetchWithTimeout).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Hi' } }],
      }),
    } as unknown as Response);

    await generator.generateContent(dummyRequest, 'prompt-id', LlmRole.MAIN);

    const callArgs = vi.mocked(fetchWithTimeout).mock.calls[0];
    const bodyStr = callArgs && callArgs[2] ? callArgs[2].body : undefined;
    const bodyObj = JSON.parse((bodyStr || '{}') as string);
    expect(bodyObj.provider).toEqual({
      ignore: ['Together'],
      quantizations: ['int4', 'bf16'],
    });
  });

  it('should fallback to map OPENROUTER_PROVIDER_SKIPS to ignore', async () => {
    vi.stubEnv('OPENROUTER_PROVIDER_SKIPS', 'Lepton');
    const generator = new OpenAiContentGenerator({
      apiKey: 'test-key',
      baseUrl: 'https://openrouter.ai/api/v1',
    });

    vi.mocked(fetchWithTimeout).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Hi' } }],
      }),
    } as unknown as Response);

    await generator.generateContent(dummyRequest, 'prompt-id', LlmRole.MAIN);

    const callArgs = vi.mocked(fetchWithTimeout).mock.calls[0];
    const bodyStr = callArgs && callArgs[2] ? callArgs[2].body : undefined;
    const bodyObj = JSON.parse((bodyStr || '{}') as string);
    expect(bodyObj.provider).toEqual({
      ignore: ['Lepton'],
    });
  });

  it('should not include provider field even if env variables are set when baseUrl is not openrouter.ai', async () => {
    vi.stubEnv('OPENROUTER_PROVIDER_ORDER', 'Together');
    const generator = new OpenAiContentGenerator({
      apiKey: 'test-key',
      baseUrl: 'https://api.openai.com/v1',
    });

    vi.mocked(fetchWithTimeout).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Hi' } }],
      }),
    } as unknown as Response);

    await generator.generateContent(dummyRequest, 'prompt-id', LlmRole.MAIN);

    const callArgs = vi.mocked(fetchWithTimeout).mock.calls[0];
    const bodyStr = callArgs && callArgs[2] ? callArgs[2].body : undefined;
    const bodyObj = JSON.parse((bodyStr || '{}') as string);
    expect(bodyObj.provider).toBeUndefined();
  });
});
