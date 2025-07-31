/**
 * ServiceBase - Common functionality for all services
 * REFACTOR: Extract common patterns following DRY principle
 */

export class ServiceBase {
  constructor(dependencies) {
    this.dependencies = dependencies;
    this.logger = dependencies.logger;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) {
      this.logger.warn('Service already initialized');
      return true;
    }

    try {
      await this._doInitialize();
      this.isInitialized = true;
      this.logger.info(`${this.constructor.name} initialized successfully`);
      return true;
    } catch (error) {
      this.logger.error(`Failed to initialize ${this.constructor.name}`, { error });
      throw error;
    }
  }

  // Template method - subclasses override this
  async _doInitialize() {
    // Default implementation - can be overridden
  }

  _validateRequiredDependencies(required) {
    const missing = required.filter(dep => !this.dependencies[dep]);
    if (missing.length > 0) {
      throw new Error(`Missing required dependencies: ${missing.join(', ')}`);
    }
  }

  _logOperation(operation, params = {}) {
    this.logger.info(`${this.constructor.name}: ${operation}`, params);
  }

  _logError(operation, error, params = {}) {
    this.logger.error(`${this.constructor.name}: ${operation} failed`, { 
      error: error.message, 
      ...params 
    });
  }
}