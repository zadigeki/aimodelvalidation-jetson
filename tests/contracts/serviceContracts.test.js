/**
 * Service Contract Tests
 * 
 * London School contract testing to ensure service boundaries are properly defined
 * and maintained. These tests verify that our mocks accurately represent the 
 * expected service interfaces and behaviors.
 */

import {
  AIModelServiceContract,
  ValidationRepositoryContract,
  NotificationServiceContract,
  MetricsCollectorContract,
  createAIModelServiceMock,
  createValidationRepositoryMock,
  createNotificationServiceMock,
  createMetricsCollectorMock
} from '../mocks/mockFactory.js';

describe('Service Contracts - London School Contract Testing', () => {
  
  describe('AIModelService Contract', () => {
    let aiModelService;

    beforeEach(() => {
      aiModelService = createAIModelServiceMock();
    });

    it('should implement validate method contract', async () => {
      const modelData = { id: 'test-model', type: 'classification' };
      const validationRules = { accuracyThreshold: 0.8 };

      const result = await aiModelService.validate(modelData, validationRules);

      // Contract verification: validate method signature and return type
      expect(aiModelService.validate).toHaveBeenCalledWith(modelData, validationRules);
      expect(result).toEqual(
        expect.objectContaining({
          isValid: expect.any(Boolean),
          score: expect.any(Number),
          errors: expect.any(Array),
          warnings: expect.any(Array),
          metadata: expect.objectContaining({
            modelId: expect.any(String),
            timestamp: expect.any(Number)
          })
        })
      );
    });

    it('should implement predict method contract', async () => {
      const inputData = [1, 2, 3, 4];

      const result = await aiModelService.predict(inputData);

      expect(aiModelService.predict).toHaveBeenCalledWith(inputData);
      expect(result).toEqual(
        expect.objectContaining({
          prediction: expect.any(Array),
          confidence: expect.any(Number),
          modelVersion: expect.any(String),
          processingTime: expect.any(Number)
        })
      );
    });

    it('should implement train method contract', async () => {
      const trainingData = { inputs: [[1, 2]], outputs: [[0.8]] };
      const config = { epochs: 100, learningRate: 0.01 };

      const result = await aiModelService.train(trainingData, config);

      expect(aiModelService.train).toHaveBeenCalledWith(trainingData, config);
      expect(result).toEqual(
        expect.objectContaining({
          modelId: expect.any(String),
          accuracy: expect.any(Number),
          loss: expect.any(Number),
          epochs: expect.any(Number),
          duration: expect.any(Number)
        })
      );
    });

    it('should implement evaluate method contract', async () => {
      const testData = { inputs: [[1, 2]], expectedOutputs: [[0.8]] };
      const metrics = ['accuracy', 'precision'];

      const result = await aiModelService.evaluate(testData, metrics);

      expect(aiModelService.evaluate).toHaveBeenCalledWith(testData, metrics);
      expect(result).toEqual(
        expect.objectContaining({
          accuracy: expect.any(Number),
          precision: expect.any(Number),
          recall: expect.any(Number),
          f1Score: expect.any(Number),
          confusionMatrix: expect.any(Array)
        })
      );
    });

    it('should maintain contract consistency across multiple calls', async () => {
      // Verify consistent behavior across multiple invocations
      const modelData = { id: 'consistent-test' };
      const rules = { threshold: 0.5 };

      const result1 = await aiModelService.validate(modelData, rules);
      const result2 = await aiModelService.validate(modelData, rules);

      // Both results should follow the same contract structure
      expect(result1).toEqual(expect.objectContaining({
        isValid: expect.any(Boolean),
        score: expect.any(Number)
      }));
      expect(result2).toEqual(expect.objectContaining({
        isValid: expect.any(Boolean),
        score: expect.any(Number)
      }));

      expect(aiModelService.validate).toHaveBeenCalledTimes(2);
    });
  });

  describe('ValidationRepository Contract', () => {
    let repository;

    beforeEach(() => {
      repository = createValidationRepositoryMock();
    });

    it('should implement save method contract', async () => {
      const validationResult = {
        isValid: true,
        score: 0.95,
        modelId: 'test-model'
      };

      const savedId = await repository.save(validationResult);

      expect(repository.save).toHaveBeenCalledWith(validationResult);
      expect(typeof savedId).toBe('string');
      expect(savedId.length).toBeGreaterThan(0);
    });

    it('should implement findById method contract', async () => {
      const testId = 'validation-123';

      const result = await repository.findById(testId);

      expect(repository.findById).toHaveBeenCalledWith(testId);
      expect(result).toEqual(
        expect.objectContaining({
          id: expect.any(String),
          status: expect.any(String),
          result: expect.any(Object)
        })
      );
    });

    it('should implement findByModel method contract', async () => {
      const modelId = 'model-456';

      const results = await repository.findByModel(modelId);

      expect(repository.findByModel).toHaveBeenCalledWith(modelId);
      expect(Array.isArray(results)).toBe(true);
      if (results.length > 0) {
        expect(results[0]).toEqual(
          expect.objectContaining({
            id: expect.any(String),
            modelId: expect.any(String),
            status: expect.any(String)
          })
        );
      }
    });

    it('should implement delete method contract', async () => {
      const testId = 'validation-to-delete';

      const result = await repository.delete(testId);

      expect(repository.delete).toHaveBeenCalledWith(testId);
      expect(typeof result).toBe('boolean');
    });
  });

  describe('NotificationService Contract', () => {
    let notificationService;

    beforeEach(() => {
      notificationService = createNotificationServiceMock();
    });

    it('should implement sendValidationComplete method contract', async () => {
      const validationResult = {
        isValid: true,
        modelId: 'test-model',
        score: 0.92
      };

      const result = await notificationService.sendValidationComplete(validationResult);

      expect(notificationService.sendValidationComplete).toHaveBeenCalledWith(validationResult);
      expect(typeof result).toBe('boolean');
    });

    it('should implement sendValidationFailed method contract', async () => {
      const error = new Error('Validation failed');
      const context = { modelId: 'failed-model', errors: ['Low accuracy'] };

      const result = await notificationService.sendValidationFailed(error, context);

      expect(notificationService.sendValidationFailed).toHaveBeenCalledWith(error, context);
      expect(typeof result).toBe('boolean');
    });

    it('should implement sendAlert method contract', async () => {
      const message = 'System alert: High memory usage';
      const severity = 'warning';

      const result = await notificationService.sendAlert(message, severity);

      expect(notificationService.sendAlert).toHaveBeenCalledWith(message, severity);
      expect(typeof result).toBe('boolean');
    });
  });

  describe('MetricsCollector Contract', () => {
    let metricsCollector;

    beforeEach(() => {
      metricsCollector = createMetricsCollectorMock();
    });

    it('should implement recordValidation method contract', () => {
      const modelId = 'metrics-test-model';
      const duration = 1500;
      const result = { isValid: true, score: 0.88 };

      metricsCollector.recordValidation(modelId, duration, result);

      expect(metricsCollector.recordValidation).toHaveBeenCalledWith(modelId, duration, result);
    });

    it('should implement recordPrediction method contract', () => {
      const modelId = 'prediction-model';
      const inputSize = 1024;
      const latency = 250;

      metricsCollector.recordPrediction(modelId, inputSize, latency);

      expect(metricsCollector.recordPrediction).toHaveBeenCalledWith(modelId, inputSize, latency);
    });

    it('should implement getMetrics method contract', async () => {
      const modelId = 'metrics-model';
      const timeRange = '24h';

      const result = await metricsCollector.getMetrics(modelId, timeRange);

      expect(metricsCollector.getMetrics).toHaveBeenCalledWith(modelId, timeRange);
      expect(result).toEqual(
        expect.objectContaining({
          validations: expect.any(Number),
          predictions: expect.any(Number),
          averageLatency: expect.any(Number),
          errorRate: expect.any(Number)
        })
      );
    });
  });

  describe('Cross-Contract Integration', () => {
    it('should verify contract compatibility between services', async () => {
      const aiService = createAIModelServiceMock();
      const repository = createValidationRepositoryMock();
      const notifier = createNotificationServiceMock();
      const metrics = createMetricsCollectorMock();

      // Simulate a workflow that uses multiple contracts
      const modelData = { id: 'integration-test' };
      const rules = { threshold: 0.8 };

      // Step 1: Validation
      const validationResult = await aiService.validate(modelData, rules);
      
      // Step 2: Storage (contract compatibility check)
      const savedId = await repository.save(validationResult);
      
      // Step 3: Notification (contract compatibility check)
      await notifier.sendValidationComplete(validationResult);
      
      // Step 4: Metrics (contract compatibility check)
      metrics.recordValidation(modelData.id, 1000, validationResult);

      // Verify all contracts work together
      expect(typeof savedId).toBe('string');
      expect(aiService.validate).toHaveBeenCalledWith(modelData, rules);
      expect(repository.save).toHaveBeenCalledWith(validationResult);
      expect(notifier.sendValidationComplete).toHaveBeenCalledWith(validationResult);
      expect(metrics.recordValidation).toHaveBeenCalledWith(modelData.id, 1000, validationResult);
    });

    it('should maintain contract evolution compatibility', async () => {
      // Test that contract changes don't break existing integrations
      const aiService = createAIModelServiceMock({
        validate: async (modelData, rules) => ({
          isValid: true,
          score: 0.95,
          errors: [],
          warnings: [],
          metadata: { modelId: modelData.id, timestamp: Date.now() },
          // New field added to contract
          experimentalFeatures: { enabled: true }
        })
      });

      const result = await aiService.validate({ id: 'evolution-test' }, {});

      // Original contract still works
      expect(result.isValid).toBe(true);
      expect(result.score).toBe(0.95);
      
      // New features are backward compatible
      expect(result.experimentalFeatures).toEqual({ enabled: true });
    });
  });

  describe('Contract Validation Rules', () => {
    it('should enforce required contract fields', async () => {
      const aiService = createAIModelServiceMock();
      
      const result = await aiService.validate({ id: 'field-test' }, {});
      
      // Verify all required fields are present
      const requiredFields = ['isValid', 'score', 'errors', 'warnings', 'metadata'];
      requiredFields.forEach(field => {
        expect(result).toHaveProperty(field);
      });
    });

    it('should validate method signatures match contracts', () => {
      const contracts = [
        AIModelServiceContract,
        ValidationRepositoryContract, 
        NotificationServiceContract,
        MetricsCollectorContract
      ];

      contracts.forEach(contract => {
        Object.entries(contract).forEach(([methodName, signature]) => {
          expect(signature).toHaveProperty('params');
          expect(signature).toHaveProperty('returns');
          expect(Array.isArray(signature.params)).toBe(true);
          expect(typeof signature.returns).toBe('string');
        });
      });
    });
  });
});