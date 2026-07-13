/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import {
  BaseDeclarativeTool,
  BaseToolInvocation,
  DeclarativeTool,
  hasCycleInSchema,
  Kind,
  type ToolInvocation,
  type ToolResult,
} from './tools.js';
import { ToolErrorType } from './tool-error.js';
import { createMockMessageBus } from '../test-utils/mock-message-bus.js';
import type { MessageBus } from '../confirmation-bus/message-bus.js';
import { ReadFileTool } from './read-file.js';
import { makeFakeConfig } from '../test-utils/config.js';

class TestToolInvocation implements ToolInvocation<object, ToolResult> {
  constructor(
    readonly params: object,
    private readonly executeFn: () => Promise<ToolResult>,
  ) {}

  getDescription(): string {
    return 'A test invocation';
  }

  toolLocations() {
    return [];
  }

  shouldConfirmExecute(): Promise<false> {
    return Promise.resolve(false);
  }

  execute(): Promise<ToolResult> {
    return this.executeFn();
  }
}

class TestTool extends DeclarativeTool<object, ToolResult> {
  private readonly buildFn: (params: object) => TestToolInvocation;

  constructor(buildFn: (params: object) => TestToolInvocation) {
    super(
      'test-tool',
      'Test Tool',
      'A tool for testing',
      Kind.Other,
      {},
      createMockMessageBus(),
      true, // isOutputMarkdown
      false, // canUpdateOutput
    );
    this.buildFn = buildFn;
  }

  build(params: object): ToolInvocation<object, ToolResult> {
    return this.buildFn(params);
  }
}

describe('DeclarativeTool', () => {
  describe('validateBuildAndExecute', () => {
    const abortSignal = new AbortController().signal;

    it('should return INVALID_TOOL_PARAMS error if build fails', async () => {
      const buildError = new Error('Invalid build parameters');
      const buildFn = vi.fn().mockImplementation(() => {
        throw buildError;
      });
      const tool = new TestTool(buildFn);
      const params = { foo: 'bar' };

      const result = await tool.validateBuildAndExecute(params, abortSignal);

      expect(buildFn).toHaveBeenCalledWith(params);
      expect(result).toEqual({
        llmContent: `Error: Invalid parameters provided. Reason: ${buildError.message}`,
        returnDisplay: buildError.message,
        error: {
          message: buildError.message,
          type: ToolErrorType.INVALID_TOOL_PARAMS,
        },
      });
    });

    it('should return EXECUTION_FAILED error if execute fails', async () => {
      const executeError = new Error('Execution failed');
      const executeFn = vi.fn().mockRejectedValue(executeError);
      const invocation = new TestToolInvocation({}, executeFn);
      const buildFn = vi.fn().mockReturnValue(invocation);
      const tool = new TestTool(buildFn);
      const params = { foo: 'bar' };

      const result = await tool.validateBuildAndExecute(params, abortSignal);

      expect(buildFn).toHaveBeenCalledWith(params);
      expect(executeFn).toHaveBeenCalled();
      expect(result).toEqual({
        llmContent: `Error: Tool call execution failed. Reason: ${executeError.message}`,
        returnDisplay: executeError.message,
        error: {
          message: executeError.message,
          type: ToolErrorType.EXECUTION_FAILED,
        },
      });
    });

    it('should return the result of execute on success', async () => {
      const successResult: ToolResult = {
        llmContent: 'Success!',
        returnDisplay: 'Success!',
      };
      const executeFn = vi.fn().mockResolvedValue(successResult);
      const invocation = new TestToolInvocation({}, executeFn);
      const buildFn = vi.fn().mockReturnValue(invocation);
      const tool = new TestTool(buildFn);
      const params = { foo: 'bar' };

      const result = await tool.validateBuildAndExecute(params, abortSignal);

      expect(buildFn).toHaveBeenCalledWith(params);
      expect(executeFn).toHaveBeenCalled();
      expect(result).toEqual(successResult);
    });
  });
});

describe('hasCycleInSchema', () => {
  it('should detect a simple direct cycle', () => {
    const schema = {
      properties: {
        data: {
          $ref: '#/properties/data',
        },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(true);
  });

  it('should detect a cycle from object properties referencing parent properties', () => {
    const schema = {
      type: 'object',
      properties: {
        data: {
          type: 'object',
          properties: {
            child: { $ref: '#/properties/data' },
          },
        },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(true);
  });

  it('should detect a cycle from array items referencing parent properties', () => {
    const schema = {
      type: 'object',
      properties: {
        data: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              child: { $ref: '#/properties/data/items' },
            },
          },
        },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(true);
  });

  it('should detect a cycle between sibling properties', () => {
    const schema = {
      type: 'object',
      properties: {
        a: {
          type: 'object',
          properties: {
            child: { $ref: '#/properties/b' },
          },
        },
        b: {
          type: 'object',
          properties: {
            child: { $ref: '#/properties/a' },
          },
        },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(true);
  });

  it('should not detect a cycle in a valid schema', () => {
    const schema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        address: { $ref: '#/definitions/address' },
      },
      definitions: {
        address: {
          type: 'object',
          properties: {
            street: { type: 'string' },
            city: { type: 'string' },
          },
        },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(false);
  });

  it('should handle non-cyclic sibling refs', () => {
    const schema = {
      properties: {
        a: { $ref: '#/definitions/stringDef' },
        b: { $ref: '#/definitions/stringDef' },
      },
      definitions: {
        stringDef: { type: 'string' },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(false);
  });

  it('should handle nested but not cyclic refs', () => {
    const schema = {
      properties: {
        a: { $ref: '#/definitions/defA' },
      },
      definitions: {
        defA: { properties: { b: { $ref: '#/definitions/defB' } } },
        defB: { type: 'string' },
      },
    };
    expect(hasCycleInSchema(schema)).toBe(false);
  });

  it('should return false for an empty schema', () => {
    expect(hasCycleInSchema({})).toBe(false);
  });
});

describe('BaseDeclarativeTool parameter normalization', () => {
  const bus = createMockMessageBus();

  class MockToolInvocation
    implements ToolInvocation<Record<string, unknown>, ToolResult>
  {
    constructor(
      readonly params: Record<string, unknown>,
      private readonly executeFn: () => Promise<ToolResult>,
    ) {}

    getDescription(): string {
      return 'A mock invocation';
    }

    toolLocations() {
      return [];
    }

    shouldConfirmExecute(): Promise<false> {
      return Promise.resolve(false);
    }

    execute(): Promise<ToolResult> {
      return this.executeFn();
    }
  }

  class MockBaseTool extends BaseDeclarativeTool<
    Record<string, unknown>,
    ToolResult
  > {
    constructor(parameterSchema: object) {
      super(
        'mock-tool',
        'Mock Tool',
        'A tool for testing parameter normalization',
        Kind.Other,
        parameterSchema,
        bus,
      );
    }

    protected override createInvocation(
      params: Record<string, unknown>,
      _messageBus: MessageBus,
      _toolName?: string,
      _toolDisplayName?: string,
    ): ToolInvocation<Record<string, unknown>, ToolResult> {
      return new MockToolInvocation(params, vi.fn());
    }
  }

  it('should map path to file_path if file_path is expected and path is provided', () => {
    const schema = {
      type: 'object',
      properties: {
        file_path: { type: 'string' },
        content: { type: 'string' },
      },
      required: ['file_path'],
    };
    const tool = new MockBaseTool(schema);
    const params = { path: 'hello.txt', content: 'world' };
    const err = tool.validateToolParams(params);
    expect(err).toBeNull();
    // Verify it mutated the params object
    expect(params).toHaveProperty('file_path', 'hello.txt');
  });

  it('should map path to dir_path if dir_path is expected and path is provided', () => {
    const schema = {
      type: 'object',
      properties: {
        dir_path: { type: 'string' },
      },
      required: ['dir_path'],
    };
    const tool = new MockBaseTool(schema);
    const params = { path: '/some/dir' };
    const err = tool.validateToolParams(params);
    expect(err).toBeNull();
    expect(params).toHaveProperty('dir_path', '/some/dir');
  });

  it('should not map path to file_path if path is already defined in the schema', () => {
    const schema = {
      type: 'object',
      properties: {
        file_path: { type: 'string' },
        path: { type: 'string' },
      },
      required: ['file_path', 'path'],
    };
    const tool = new MockBaseTool(schema);
    const params = { path: 'original_path', file_path: 'original_file_path' };
    const err = tool.validateToolParams(params);
    expect(err).toBeNull();
    expect(params.file_path).toBe('original_file_path');
    expect(params.path).toBe('original_path');
  });

  it('should map path or filename to plan_filename if plan_filename is expected', () => {
    const schema = {
      type: 'object',
      properties: {
        plan_filename: { type: 'string' },
      },
      required: ['plan_filename'],
    };
    const tool = new MockBaseTool(schema);

    const params1 = { path: 'my-plan.md' };
    const err1 = tool.validateToolParams(params1);
    expect(err1).toBeNull();
    expect(params1).toHaveProperty('plan_filename', 'my-plan.md');

    const params2 = { filename: 'another-plan.md' };
    const err2 = tool.validateToolParams(params2);
    expect(err2).toBeNull();
    expect(params2).toHaveProperty('plan_filename', 'another-plan.md');
  });
});

describe('Tools Read-Only property', () => {
  it('should have isReadOnly true for ReadFileTool', () => {
    const config = makeFakeConfig();
    const bus = createMockMessageBus();
    const tool = new ReadFileTool(config, bus);
    expect(tool.isReadOnly).toBe(true);
  });

  it('should derive isReadOnly from Kind', () => {
    const bus = createMockMessageBus();
    class MyTool extends DeclarativeTool<object, ToolResult> {
      build(_params: object): ToolInvocation<object, ToolResult> {
        throw new Error('Not implemented');
      }
    }

    const mutator = new MyTool('m', 'M', 'd', Kind.Edit, {}, bus);
    expect(mutator.isReadOnly).toBe(false);

    const reader = new MyTool('r', 'R', 'd', Kind.Read, {}, bus);
    expect(reader.isReadOnly).toBe(true);

    const searcher = new MyTool('s', 'S', 'd', Kind.Search, {}, bus);
    expect(searcher.isReadOnly).toBe(true);
  });
});

describe('toJSON serialization', () => {
  it('DeclarativeTool.toJSON should return essential metadata', () => {
    const bus = createMockMessageBus();
    class MyTool extends DeclarativeTool<object, ToolResult> {
      build(_params: object): ToolInvocation<object, ToolResult> {
        throw new Error('Not implemented');
      }
    }
    const tool = new MyTool(
      'name',
      'display',
      'desc',
      Kind.Read,
      { type: 'object' },
      bus,
    );
    const json = tool.toJSON();

    expect(json).toEqual({
      name: 'name',
      displayName: 'display',
      description: 'desc',
      kind: Kind.Read,
      parameterSchema: { type: 'object' },
    });
    // Ensure messageBus is NOT included in serialization
    expect(Object.keys(json)).not.toContain('messageBus');
    expect(JSON.stringify(tool)).toContain('"name":"name"');
    expect(JSON.stringify(tool)).not.toContain('messageBus');
  });

  it('BaseToolInvocation.toJSON should return only params', () => {
    const bus = createMockMessageBus();
    const params = { foo: 'bar' };
    class MyInvocation extends BaseToolInvocation<object, ToolResult> {
      getDescription() {
        return 'desc';
      }
      async execute() {
        return { llmContent: '', returnDisplay: '' };
      }
    }
    const invocation = new MyInvocation(params, bus, 'tool');
    const json = invocation.toJSON();

    expect(json).toEqual({ params });
    // Ensure messageBus is NOT included in serialization
    expect(Object.keys(json)).not.toContain('messageBus');
    expect(JSON.stringify(invocation)).toBe('{"params":{"foo":"bar"}}');
  });
});
