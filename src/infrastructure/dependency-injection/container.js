/**
 * Dependency Injection Container
 * London School TDD - Enables mock injection for testability
 * Supports constructor injection and interface-based design
 */

export class DIContainer {
  constructor() {
    this.dependencies = new Map();
    this.singletons = new Map();
    this.factories = new Map();
  }

  /**
   * Register a dependency with its implementation
   * @param {string} name - Dependency name
   * @param {Function|Object} implementation - Implementation or factory function
   * @param {Object} options - Registration options
   */
  register(name, implementation, options = {}) {
    const config = {
      singleton: options.singleton || false,
      factory: options.factory || false,
      dependencies: options.dependencies || [],
      ...options
    };

    this.dependencies.set(name, { implementation, config });
    return this;
  }

  /**
   * Register a singleton dependency
   * @param {string} name - Dependency name
   * @param {Function|Object} implementation - Implementation
   * @param {Object} options - Registration options
   */
  registerSingleton(name, implementation, options = {}) {
    return this.register(name, implementation, { ...options, singleton: true });
  }

  /**
   * Register a factory function
   * @param {string} name - Dependency name
   * @param {Function} factory - Factory function
   * @param {Object} options - Registration options
   */
  registerFactory(name, factory, options = {}) {
    return this.register(name, factory, { ...options, factory: true });
  }

  /**
   * Resolve a dependency by name
   * @param {string} name - Dependency name
   * @returns {Object} Resolved dependency instance
   */
  resolve(name) {
    if (!this.dependencies.has(name)) {
      throw new Error(`Dependency '${name}' not registered`);
    }

    const { implementation, config } = this.dependencies.get(name);

    // Return singleton if already created
    if (config.singleton && this.singletons.has(name)) {
      return this.singletons.get(name);
    }

    let instance;

    if (config.factory) {
      // Create instance using factory function
      instance = implementation(this);
    } else if (typeof implementation === 'function') {
      // Create instance using constructor with dependency injection
      const dependencyInstances = config.dependencies.map(dep => this.resolve(dep));
      instance = new implementation(...dependencyInstances);
    } else {
      // Use object directly (for mocks and pre-created instances)
      instance = implementation;
    }

    // Store singleton
    if (config.singleton) {
      this.singletons.set(name, instance);
    }

    return instance;
  }

  /**
   * Create a child container for test isolation
   * London School: Each test gets its own container with mocks
   */
  createChildContainer() {
    const child = new DIContainer();
    
    // Copy parent registrations
    for (const [name, config] of this.dependencies.entries()) {
      child.dependencies.set(name, { ...config });
    }
    
    return child;
  }

  /**
   * Override dependency for testing (London School mock injection)
   * @param {string} name - Dependency name
   * @param {Object} mockImplementation - Mock object
   */
  mock(name, mockImplementation) {
    this.register(name, mockImplementation, { singleton: true });
    return this;
  }

  /**
   * Clear all dependencies (useful for test cleanup)
   */
  clear() {
    this.dependencies.clear();
    this.singletons.clear();
    this.factories.clear();
  }

  /**
   * Get all registered dependency names
   */
  getRegisteredNames() {
    return Array.from(this.dependencies.keys());
  }

  /**
   * Check if a dependency is registered
   */
  isRegistered(name) {
    return this.dependencies.has(name);
  }
}

// Default container instance
export const container = new DIContainer();

// London School TDD helper: Create test container with mocks
export const createTestContainer = (mocks = {}) => {
  const testContainer = container.createChildContainer();
  
  // Inject all provided mocks
  Object.entries(mocks).forEach(([name, mock]) => {
    testContainer.mock(name, mock);
  });
  
  return testContainer;
};

// Decorator for dependency injection (London School style)
export const injectable = (dependencies = []) => {
  return (target) => {
    target.dependencies = dependencies;
    return target;
  };
};

// Helper to create factory functions for complex dependencies
export const factory = (name, dependencies = []) => {
  return (container) => {
    const deps = dependencies.map(dep => container.resolve(dep));
    return container.resolve(name, ...deps);
  };
};