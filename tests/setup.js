/**
 * Jest Setup for TDD London School Approach
 * Configures mock-first testing environment with behavior verification
 */

import { jest } from '@jest/globals';

// Global test configuration following London School principles
global.beforeEach(() => {
  // Clear all mocks before each test to ensure isolation
  jest.clearAllMocks();
  
  // Reset mock call history
  jest.resetAllMocks();
});

// Mock factory for creating test doubles with interaction tracking
global.createMockObject = (name, methods = {}) => {
  const mock = {};
  
  Object.keys(methods).forEach(methodName => {
    mock[methodName] = jest.fn().mockName(`${name}.${methodName}`);
    
    if (methods[methodName]) {
      mock[methodName].mockImplementation(methods[methodName]);
    }
  });
  
  // Add mock identification for debugging
  mock._mockName = name;
  mock._mockType = 'london-school-mock';
  
  return mock;
};

// Helper for creating spy objects that track interactions
global.createSpy = (target, methodName) => {
  return jest.spyOn(target, methodName);
};

// Contract verification helper for London School approach
global.verifyContract = (mock, expectedInteractions) => {
  expectedInteractions.forEach(interaction => {
    expect(mock[interaction.method]).toHaveBeenCalledWith(...interaction.args);
  });
};

// Behavior verification matchers
expect.extend({
  toHaveBeenCalledBefore(received, comparison) {
    const receivedCalls = received.mock.invocationCallOrder;
    const comparisonCalls = comparison.mock.invocationCallOrder;
    
    if (!receivedCalls.length) {
      return {
        message: () => `Expected ${received.getMockName()} to have been called`,
        pass: false,
      };
    }
    
    if (!comparisonCalls.length) {
      return {
        message: () => `Expected ${comparison.getMockName()} to have been called`,
        pass: false,
      };
    }
    
    const pass = Math.max(...receivedCalls) < Math.min(...comparisonCalls);
    
    return {
      message: () => 
        pass 
          ? `Expected ${received.getMockName()} not to have been called before ${comparison.getMockName()}`
          : `Expected ${received.getMockName()} to have been called before ${comparison.getMockName()}`,
      pass,
    };
  },
  
  toSatisfyContract(received, contract) {
    const requiredMethods = Object.keys(contract);
    
    // Check both own properties and prototype methods
    const availableMethods = [];
    const obj = received;
    
    // Get instance methods
    Object.getOwnPropertyNames(obj).forEach(key => {
      if (typeof obj[key] === 'function' && !key.startsWith('_') && !key.startsWith('mock')) {
        availableMethods.push(key);
      }
    });
    
    // Get prototype methods
    if (obj.constructor && obj.constructor.prototype) {
      Object.getOwnPropertyNames(obj.constructor.prototype).forEach(key => {
        if (typeof obj.constructor.prototype[key] === 'function' && key !== 'constructor') {
          availableMethods.push(key);
        }
      });
    }
    
    const missingMethods = requiredMethods.filter(method => 
      !availableMethods.includes(method)
    );
    
    const pass = missingMethods.length === 0;
    
    return {
      message: () => 
        pass 
          ? `Service satisfies contract`
          : `Service missing methods: ${missingMethods.join(', ')}. Available: ${availableMethods.join(', ')}`,
      pass,
    };
  }
});

// London School test patterns
global.describeUnit = (unitName, tests) => {
  describe(`Unit: ${unitName}`, () => {
    describe('London School TDD - Mock-driven behavior verification', tests);
  });
};

global.describeCollaboration = (collaborationName, tests) => {
  describe(`Collaboration: ${collaborationName}`, () => {
    describe('Object interaction patterns', tests);
  });
};

global.describeContract = (contractName, tests) => {
  describe(`Contract: ${contractName}`, () => {
    describe('Interface definition and verification', tests);
  });
};

// Swarm coordination helpers
global.notifySwarm = async (message) => {
  // In a real swarm environment, this would notify other agents
  console.log(`[SWARM] ${message}`);
};

global.shareTestResults = async (results) => {
  // Share test results with other swarm agents
  console.log(`[SWARM] Test Results: ${JSON.stringify(results, null, 2)}`);
};

console.log('🎭 TDD London School test environment initialized');
console.log('📋 Mock-first approach enabled with behavior verification');