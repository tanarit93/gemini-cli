/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { isApiError, isStructuredError } from './quotaErrorDetection.js';
import { DEFAULT_GEMINI_FLASH_MODEL } from '../config/models.js';
import type { UserTierId } from '../code_assist/types.js';
import { AuthType } from '../core/contentGenerator.js';

const RATE_LIMIT_ERROR_MESSAGE_USE_GEMINI =
  '\nPlease wait and try again later. To increase your limits, request a quota increase through AI Studio, or switch to another /auth method';
const RATE_LIMIT_ERROR_MESSAGE_VERTEX =
  '\nPlease wait and try again later. To increase your limits, request a quota increase through Vertex, or switch to another /auth method';
const getRateLimitErrorMessageDefault = (
  fallbackModel: string = DEFAULT_GEMINI_FLASH_MODEL,
) =>
  `\nPossible quota limitations in place or slow response times detected. Switching to the ${fallbackModel} model for the rest of this session.`;

function getRateLimitMessage(
  authType?: AuthType,
  fallbackModel?: string,
): string {
  switch (authType) {
    case AuthType.USE_GEMINI:
      return RATE_LIMIT_ERROR_MESSAGE_USE_GEMINI;
    case AuthType.USE_VERTEX_AI:
      return RATE_LIMIT_ERROR_MESSAGE_VERTEX;
    default:
      return getRateLimitErrorMessageDefault(fallbackModel);
  }
}

export function parseAndFormatApiError(
  error: unknown,
  authType?: AuthType,
  userTier?: UserTierId,
  currentModel?: string,
  fallbackModel?: string,
): string {
  // If the error object has a message property containing a JSON string, let's extract it first.
  if (
    error &&
    typeof error === 'object' &&
    'message' in error &&
    typeof error.message === 'string'
  ) {
    const message = error.message;
    const jsonStart = message.indexOf('{');
    if (jsonStart !== -1) {
      const jsonString = message.substring(jsonStart);
      try {
        const parsed = JSON.parse(jsonString) as unknown;
        if (parsed && typeof parsed === 'object') {
          let finalMessage = '';
          if (
            'error' in parsed &&
            parsed.error &&
            typeof parsed.error === 'object' &&
            'message' in parsed.error
          ) {
            finalMessage = String(parsed.error.message);
          } else if ('message' in parsed) {
            finalMessage = String(parsed.message);
          }
          if (finalMessage) {
            const prefix = message.substring(0, jsonStart);
            let text = `[API Error: ${prefix}${finalMessage}]`;
            const status =
              'status' in error && typeof error.status === 'number'
                ? error.status
                : undefined;
            const code =
              'error' in parsed &&
              parsed.error &&
              typeof parsed.error === 'object' &&
              'code' in parsed.error &&
              typeof parsed.error.code === 'number'
                ? parsed.error.code
                : undefined;
            if (status === 429 || code === 429) {
              text += getRateLimitMessage(authType, fallbackModel);
            }
            return text;
          }
        }
      } catch {
        // ignore, fall through
      }
    }
  }

  if (isStructuredError(error)) {
    let text = `[API Error: ${error.message}]`;
    if (error.status === 429) {
      text += getRateLimitMessage(authType, fallbackModel);
    }
    return text;
  }

  // The error message might be a string containing a JSON object.
  if (typeof error === 'string') {
    const jsonStart = error.indexOf('{');
    if (jsonStart === -1) {
      return `[API Error: ${error}]`; // Not a JSON error, return as is.
    }

    const jsonString = error.substring(jsonStart);

    try {
      const parsedError = JSON.parse(jsonString) as unknown;
      if (isApiError(parsedError)) {
        let finalMessage = parsedError.error.message;
        try {
          // See if the message is a stringified JSON with another error
          const nestedError = JSON.parse(finalMessage) as unknown;
          if (isApiError(nestedError)) {
            finalMessage = nestedError.error.message;
          }
        } catch {
          // It's not a nested JSON error, so we just use the message as is.
        }
        let text = `[API Error: ${finalMessage} (Status: ${parsedError.error.status})]`;
        if (parsedError.error.code === 429) {
          text += getRateLimitMessage(authType, fallbackModel);
        }
        return text;
      }

      if (parsedError && typeof parsedError === 'object') {
        let finalMessage = '';
        if (
          'error' in parsedError &&
          parsedError.error &&
          typeof parsedError.error === 'object' &&
          'message' in parsedError.error
        ) {
          finalMessage = String(parsedError.error.message);
        } else if ('message' in parsedError) {
          finalMessage = String(parsedError.message);
        }
        if (finalMessage) {
          const prefix = error.substring(0, jsonStart);
          return `[API Error: ${prefix}${finalMessage}]`;
        }
      }
    } catch {
      // Not a valid JSON, fall through and return the original message.
    }
    return `[API Error: ${error}]`;
  }

  if (error instanceof Error) {
    return `[API Error: ${error.message}]`;
  }

  try {
    if (typeof error !== 'object' || error === null) {
      return '[API Error: An unknown error occurred.]';
    }
    return `[API Error: An unknown error occurred. Details: ${JSON.stringify(error, Object.getOwnPropertyNames(error))}]`;
  } catch {
    return '[API Error: An unknown error occurred.]';
  }
}
