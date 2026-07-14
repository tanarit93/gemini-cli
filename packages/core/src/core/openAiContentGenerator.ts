/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  type CountTokensResponse,
  type GenerateContentParameters,
  type CountTokensParameters,
  type EmbedContentResponse,
  type EmbedContentParameters,
  type Content,
  type Part,
} from '@google/genai';
import { toContents } from '../code_assist/converter.js';
import type {
  ContentGenerator,
  ContentGeneratorConfig,
} from './contentGenerator.js';
import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import type { LlmRole } from '../telemetry/types.js';
import { fetchWithTimeout } from '../utils/fetch.js';

interface OpenAiToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface OpenAiMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string;
  tool_calls?: OpenAiToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface OpenAiTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters: unknown;
  };
}

interface OpenAiRequest {
  model: string;
  messages: OpenAiMessage[];
  stream: boolean;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stop?: string[];
  tools?: OpenAiTool[];
  tool_choice?: 'required' | 'none' | 'auto';
  response_format?:
    | { type: 'json_object' }
    | {
        type: 'json_schema';
        json_schema: {
          name: string;
          schema: unknown;
          strict: boolean;
        };
      };
  provider?: {
    order?: string[];
    allow_fallbacks?: boolean;
    skips?: string[];
    quantizations?: string[];
  };
}

interface OpenAiResponseChoice {
  message?: {
    role: string;
    content?: string | null;
    tool_calls?: OpenAiToolCall[];
  };
  delta?: {
    role?: string;
    content?: string | null;
    tool_calls?: Array<{
      index: number;
      id?: string;
      type?: 'function';
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };
  finish_reason?: string;
}

interface OpenAiResponse {
  choices?: OpenAiResponseChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface OpenAiStreamChunk {
  choices?: OpenAiResponseChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

export class OpenAiContentGenerator implements ContentGenerator {
  userTier?: UserTierId;
  userTierName?: string;
  paidTier?: GeminiUserTier;

  constructor(private readonly config: ContentGeneratorConfig) {}

  private getEndpointUrl(): string {
    const baseUrl =
      this.config.baseUrl?.replace(/\/+$/, '') || 'https://api.openai.com/v1';
    return `${baseUrl}/chat/completions`;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.customHeaders,
    };
    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }
    return headers;
  }

  private convertToOpenAiMessages(
    contents: Content[],
    systemInstruction?: unknown,
  ): OpenAiMessage[] {
    const messages: OpenAiMessage[] = [];

    // Prepend system instruction if exists
    if (systemInstruction) {
      let systemText = '';
      if (typeof systemInstruction === 'string') {
        systemText = systemInstruction;
      } else if (Array.isArray(systemInstruction)) {
        systemText = systemInstruction
          .map((p) => {
            if (p && typeof p === 'object' && 'text' in p) {
              return String(p.text || '');
            }
            return String(p || '');
          })
          .join('');
      } else if (systemInstruction && typeof systemInstruction === 'object') {
        if (
          'parts' in systemInstruction &&
          Array.isArray(systemInstruction.parts)
        ) {
          systemText = systemInstruction.parts
            .map((p) => {
              if (p && typeof p === 'object' && 'text' in p) {
                return String(p.text || '');
              }
              return String(p || '');
            })
            .join('');
        } else if (
          'text' in systemInstruction &&
          typeof systemInstruction.text === 'string'
        ) {
          systemText = systemInstruction.text || '';
        }
      }

      if (systemText) {
        messages.push({
          role: 'system',
          content: systemText,
        });
      }
    }

    // Convert contents
    for (const content of contents) {
      let role: 'user' | 'assistant' | 'tool' =
        content.role === 'model' ? 'assistant' : 'user';
      if (content.role === 'function') {
        role = 'tool';
      }

      const parts = content.parts || [];
      const textParts = parts
        .filter((p) => p.text)
        .map((p) => p.text)
        .join('');
      const toolCalls: OpenAiToolCall[] = parts
        .filter((p) => p.functionCall)
        .map((p) => {
          const fc = p.functionCall!;
          return {
            id: `call_${Math.random().toString(36).substring(2, 9)}`,
            type: 'function',
            function: {
              name: fc.name || '',
              arguments: JSON.stringify(fc.args || {}),
            },
          };
        });

      const toolResponses: OpenAiMessage[] = parts
        .filter((p) => p.functionResponse)
        .map((p) => {
          const fr = p.functionResponse!;
          return {
            role: 'tool',
            name: fr.name || '',
            tool_call_id: `call_${fr.name || ''}`,
            content: JSON.stringify(fr.response || {}),
          };
        });

      if (toolResponses.length > 0) {
        messages.push(...toolResponses);
      } else {
        const message: OpenAiMessage = {
          role,
          content: textParts,
        };
        if (toolCalls.length > 0) {
          message.tool_calls = toolCalls;
        }
        messages.push(message);
      }
    }

    return messages;
  }

  private convertToOpenAiTools(tools?: unknown): OpenAiTool[] | undefined {
    if (!tools || !Array.isArray(tools)) return undefined;
    const openAiTools: OpenAiTool[] = [];

    for (const tool of tools) {
      if (tool && typeof tool === 'object') {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        const toolObj = tool as Record<string, unknown>;
        const fdList = toolObj['functionDeclarations'];
        if (Array.isArray(fdList)) {
          for (const fd of fdList) {
            if (fd && typeof fd === 'object') {
              // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
              const fdObj = fd as Record<string, unknown>;
              if ('name' in fdObj) {
                openAiTools.push({
                  type: 'function',
                  function: {
                    name: String(fdObj['name']),
                    description:
                      'description' in fdObj
                        ? String(fdObj['description'])
                        : undefined,
                    parameters: fdObj['parameters'] || {
                      type: 'object',
                      properties: {},
                    },
                  },
                });
              }
            }
          }
        }
      }
    }
    return openAiTools.length > 0 ? openAiTools : undefined;
  }

  private convertToOpenAiRequest(
    request: GenerateContentParameters,
    stream: boolean = false,
  ): OpenAiRequest {
    const config = request.config || {};

    const payload: OpenAiRequest = {
      model: request.model,
      messages: this.convertToOpenAiMessages(
        toContents(request.contents),
        config.systemInstruction,
      ),
      stream,
    };

    if (config.temperature !== undefined) {
      payload.temperature = config.temperature;
    }
    if (config.topP !== undefined) {
      payload.top_p = config.topP;
    }
    if (config.maxOutputTokens !== undefined) {
      payload.max_tokens = config.maxOutputTokens;
    }
    if (config.stopSequences !== undefined) {
      payload.stop = config.stopSequences;
    }

    const tools = this.convertToOpenAiTools(config.tools);
    if (tools) {
      payload.tools = tools;
      if (config.toolConfig && config.toolConfig.functionCallingConfig) {
        const mode = config.toolConfig.functionCallingConfig.mode;
        if (mode === 'ANY') {
          payload.tool_choice = 'required';
        } else if (mode === 'NONE') {
          payload.tool_choice = 'none';
        } else {
          payload.tool_choice = 'auto';
        }
      }
    }

    if (config.responseMimeType === 'application/json') {
      if (config.responseSchema) {
        payload.response_format = {
          type: 'json_schema',
          json_schema: {
            name: 'response_schema',
            schema: config.responseSchema,
            strict: false,
          },
        };
      } else {
        payload.response_format = { type: 'json_object' };
      }
    }

    const isOpenRouter = this.config.baseUrl?.includes('openrouter.ai');
    if (isOpenRouter) {
      const orderEnv = process.env['OPENROUTER_PROVIDER_ORDER'];
      const order = orderEnv
        ? orderEnv
            .split(',')
            .map((p) => p.trim())
            .filter(Boolean)
        : undefined;
      const finalOrder = order && order.length > 0 ? order : undefined;

      const allowFallbacksEnv = process.env['OPENROUTER_ALLOW_FALLBACKS'];
      const allow_fallbacks =
        allowFallbacksEnv !== undefined && allowFallbacksEnv !== ''
          ? allowFallbacksEnv === 'true'
          : undefined;

      const skipsEnv = process.env['OPENROUTER_PROVIDER_SKIPS'];
      const skips = skipsEnv
        ? skipsEnv
            .split(',')
            .map((p) => p.trim())
            .filter(Boolean)
        : undefined;
      const finalSkips = skips && skips.length > 0 ? skips : undefined;

      const quantizationsEnv = process.env['OPENROUTER_PROVIDER_QUANTIZATIONS'];
      const quantizations = quantizationsEnv
        ? quantizationsEnv
            .split(',')
            .map((p) => p.trim())
            .filter(Boolean)
        : undefined;
      const finalQuantizations =
        quantizations && quantizations.length > 0 ? quantizations : undefined;

      if (
        finalOrder ||
        allow_fallbacks !== undefined ||
        finalSkips ||
        finalQuantizations
      ) {
        payload.provider = {
          ...(finalOrder && { order: finalOrder }),
          ...(allow_fallbacks !== undefined && { allow_fallbacks }),
          ...(finalSkips && { skips: finalSkips }),
          ...(finalQuantizations && { quantizations: finalQuantizations }),
        };
      }
    }

    return payload;
  }

  private convertToGeminiResponse(
    openAiResponse: OpenAiResponse,
  ): GenerateContentResponse {
    const choice = openAiResponse.choices?.[0];
    const message = choice?.message;

    const parts: Part[] = [];
    if (message?.content) {
      parts.push({ text: message.content });
    }

    if (message?.tool_calls) {
      for (const call of message.tool_calls) {
        if (call.type === 'function') {
          let args = {};
          try {
            const parsedArgs = JSON.parse(call.function.arguments) as unknown;
            // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            args = parsedArgs as Record<string, unknown>;
          } catch {
            // ignore
          }
          parts.push({
            functionCall: {
              name: call.function.name,
              args,
            },
          });
        }
      }
    }

    const geminiResponse = {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason:
            choice?.finish_reason === 'stop'
              ? 'STOP'
              : choice?.finish_reason === 'tool_calls'
                ? 'STOP'
                : choice?.finish_reason === 'length'
                  ? 'MAX_TOKENS'
                  : 'OTHER',
        },
      ],
      usageMetadata: {
        promptTokenCount: openAiResponse.usage?.prompt_tokens || 0,
        candidatesTokenCount: openAiResponse.usage?.completion_tokens || 0,
        totalTokenCount: openAiResponse.usage?.total_tokens || 0,
      },
    };

    Object.setPrototypeOf(geminiResponse, GenerateContentResponse.prototype);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    return geminiResponse as unknown as GenerateContentResponse;
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const url = this.getEndpointUrl();
    const payload = this.convertToOpenAiRequest(request, false);
    const abortSignal = request.config?.abortSignal || undefined;
    const timeout = 300000;

    const response = await fetchWithTimeout(url, timeout, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(payload),
      signal: abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      throw new Error(`OpenAI API Error (${response.status}): ${errorText}`);
    }

    const rawJson = (await response.json()) as unknown;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const data = rawJson as OpenAiResponse;
    return this.convertToGeminiResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const url = this.getEndpointUrl();
    const payload = this.convertToOpenAiRequest(request, true);
    const abortSignal = request.config?.abortSignal || undefined;
    const timeout = 300000;

    const response = await fetchWithTimeout(url, timeout, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(payload),
      signal: abortSignal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      throw new Error(`OpenAI API Error (${response.status}): ${errorText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    const stream = async function* () {
      const activeToolCalls: Record<
        string,
        {
          id?: string;
          type: 'function';
          function: { name: string; arguments: string };
        }
      > = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) {
            continue;
          }

          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') {
            return;
          }

          try {
            const rawChunk = JSON.parse(dataStr) as unknown;
            // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
            const data = rawChunk as OpenAiStreamChunk;
            const choice = data.choices?.[0];
            const delta = choice?.delta || {};

            if (delta.content) {
              const chunkResponse = {
                candidates: [
                  {
                    content: {
                      role: 'model',
                      parts: [{ text: delta.content }],
                    },
                  },
                ],
              };
              Object.setPrototypeOf(
                chunkResponse,
                GenerateContentResponse.prototype,
              );
              // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
              yield chunkResponse as unknown as GenerateContentResponse;
            }

            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                const index = tc.index;
                if (!activeToolCalls[index]) {
                  activeToolCalls[index] = {
                    id: tc.id,
                    type: 'function',
                    function: {
                      name: tc.function?.name || '',
                      arguments: tc.function?.arguments || '',
                    },
                  };
                } else {
                  if (tc.function?.name) {
                    activeToolCalls[index].function.name += tc.function.name;
                  }
                  if (tc.function?.arguments) {
                    activeToolCalls[index].function.arguments +=
                      tc.function.arguments;
                  }
                }
              }
            }

            if (choice?.finish_reason) {
              if (Object.keys(activeToolCalls).length > 0) {
                const parts: Part[] = [];
                for (const idx of Object.keys(activeToolCalls)) {
                  const call = activeToolCalls[idx];
                  let args = {};
                  try {
                    const parsedArgs = JSON.parse(
                      call.function.arguments || '{}',
                    ) as unknown;
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
                    args = parsedArgs as Record<string, unknown>;
                  } catch {
                    // ignore
                  }
                  parts.push({
                    functionCall: {
                      name: call.function.name,
                      args,
                    },
                  });
                }
                const chunkResponse = {
                  candidates: [
                    {
                      content: {
                        role: 'model',
                        parts,
                      },
                      finishReason:
                        choice.finish_reason === 'tool_calls' ? 'STOP' : 'STOP',
                    },
                  ],
                };
                Object.setPrototypeOf(
                  chunkResponse,
                  GenerateContentResponse.prototype,
                );
                // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
                yield chunkResponse as unknown as GenerateContentResponse;
              }
            }
          } catch {
            // ignore
          }
        }
      }
    };

    return stream();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    let totalChars = 0;
    const contents = toContents(request.contents);
    for (const content of contents) {
      for (const part of content.parts || []) {
        if (part.text) {
          totalChars += part.text.length;
        } else if (part.functionCall) {
          totalChars += JSON.stringify(part.functionCall).length;
        } else if (part.functionResponse) {
          totalChars += JSON.stringify(part.functionResponse).length;
        }
      }
    }

    return {
      totalTokens: Math.ceil(totalChars / 4),
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('embedContent is not supported in OpenAiContentGenerator');
  }
}
