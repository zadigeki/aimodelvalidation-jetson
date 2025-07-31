/**
 * Jest Polyfills for TDD London School Environment
 * 
 * Provides additional polyfills and utilities for consistent testing
 * across different Node.js versions and environments.
 */

// Global fetch polyfill for testing HTTP interactions
if (!global.fetch) {
  global.fetch = require('node-fetch');
}

// URL polyfill for testing URL construction
if (!global.URL) {
  global.URL = require('url').URL;
}

// URLSearchParams polyfill
if (!global.URLSearchParams) {
  global.URLSearchParams = require('url').URLSearchParams;
}

// TextEncoder/TextDecoder for binary data testing
if (!global.TextEncoder) {
  const { TextEncoder, TextDecoder } = require('util');
  global.TextEncoder = TextEncoder;
  global.TextDecoder = TextDecoder;
}

// Performance API for timing tests
if (!global.performance) {
  global.performance = require('perf_hooks').performance;
}

// AbortController for cancellation testing
if (!global.AbortController) {
  global.AbortController = require('abort-controller').AbortController;
}

// Crypto polyfill for testing cryptographic functions
if (!global.crypto) {
  global.crypto = require('crypto').webcrypto;
}

// FormData polyfill for multipart form testing
if (!global.FormData) {
  global.FormData = require('form-data');
}

// Set immediate polyfill for async testing
if (!global.setImmediate) {
  global.setImmediate = (callback, ...args) => {
    return setTimeout(callback, 0, ...args);
  };
}

// Clear immediate polyfill
if (!global.clearImmediate) {
  global.clearImmediate = (id) => {
    return clearTimeout(id);
  };
}

// Console time polyfills for performance testing
if (!console.time) {
  const timers = new Map();
  console.time = (label = 'default') => {
    timers.set(label, Date.now());
  };
  
  console.timeEnd = (label = 'default') => {
    const start = timers.get(label);
    if (start) {
      console.log(`${label}: ${Date.now() - start}ms`);
      timers.delete(label);
    }
  };
  
  console.timeLog = (label = 'default', ...data) => {
    const start = timers.get(label);
    if (start) {
      console.log(`${label}: ${Date.now() - start}ms`, ...data);
    }
  };
}

// Promise.allSettled polyfill for comprehensive async testing
if (!Promise.allSettled) {
  Promise.allSettled = function(promises) {
    return Promise.all(
      promises.map(promise =>
        Promise.resolve(promise)
          .then(value => ({ status: 'fulfilled', value }))
          .catch(reason => ({ status: 'rejected', reason }))
      )
    );
  };
}

// Array.flat polyfill for data processing tests
if (!Array.prototype.flat) {
  Array.prototype.flat = function(depth = 1) {
    const flatDeep = (arr, d) => {
      return d > 0 ? arr.reduce((acc, val) => acc.concat(Array.isArray(val) ? flatDeep(val, d - 1) : val), [])
                   : arr.slice();
    };
    return flatDeep(this, depth);
  };
}

// Array.flatMap polyfill
if (!Array.prototype.flatMap) {
  Array.prototype.flatMap = function(callback, thisArg) {
    return this.map(callback, thisArg).flat();
  };
}

// Object.fromEntries polyfill for data transformation tests
if (!Object.fromEntries) {
  Object.fromEntries = function(iterable) {
    return [...iterable].reduce((obj, [key, val]) => {
      obj[key] = val;
      return obj;
    }, {});
  };
}

// String.matchAll polyfill for regex testing
if (!String.prototype.matchAll) {
  String.prototype.matchAll = function(regexp) {
    if (!regexp.global) {
      throw new TypeError('String.prototype.matchAll called with a non-global RegExp argument');
    }
    const matches = [];
    let match;
    const regex = new RegExp(regexp);
    while ((match = regex.exec(this)) !== null) {
      matches.push(match);
    }
    return matches[Symbol.iterator]();
  };
}

// BigInt polyfill for large number testing (basic implementation)
if (typeof BigInt === 'undefined') {
  global.BigInt = function(value) {
    if (typeof value === 'string' || typeof value === 'number') {
      return { 
        toString: () => value.toString(),
        valueOf: () => Number(value)
      };
    }
    throw new TypeError('Cannot convert value to BigInt');
  };
}

// WeakRef polyfill for memory management testing (simplified)
if (!global.WeakRef) {
  global.WeakRef = class WeakRef {
    constructor(target) {
      this._target = target;
    }
    
    deref() {
      return this._target;
    }
  };
}

// FinalizationRegistry polyfill (simplified)
if (!global.FinalizationRegistry) {
  global.FinalizationRegistry = class FinalizationRegistry {
    constructor(callback) {
      this._callback = callback;
      this._registry = new Map();
    }
    
    register(target, heldValue, unregisterToken) {
      this._registry.set(unregisterToken || target, { target, heldValue });
    }
    
    unregister(unregisterToken) {
      return this._registry.delete(unregisterToken);
    }
  };
}

// Error cause polyfill for better error handling in tests
if (!Error.prototype.cause) {
  Object.defineProperty(Error.prototype, 'cause', {
    get() {
      return this._cause;
    },
    set(value) {
      this._cause = value;
    },
    configurable: true,
    enumerable: false
  });
}

// Global test utilities for London School TDD
global.testUtils = global.testUtils || {};

// Add timing utilities for performance testing
global.testUtils.timing = {
  measure: async (fn) => {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    return {
      result,
      duration: end - start
    };
  },
  
  timeout: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  retry: async (fn, maxAttempts = 3, delay = 100) => {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxAttempts) throw error;
        await global.testUtils.timing.timeout(delay * attempt);
      }
    }
  }
};

// Memory testing utilities
global.testUtils.memory = {
  getUsage: () => {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage();
    }
    return { heapUsed: 0, heapTotal: 0, external: 0, rss: 0 };
  },
  
  gc: () => {
    if (typeof global.gc === 'function') {
      global.gc();
    }
  }
};

console.log('🧪 Jest polyfills loaded for London School TDD environment');