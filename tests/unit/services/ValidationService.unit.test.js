/**
 * ValidationService Unit Tests
 * London School TDD - Deepchecks integration with behavior verification
 */

import { jest, describe, it, beforeEach, expect } from '@jest/globals';
import { ValidationService } from '../../../src/services/ValidationService.js';
import { createDeepchecksClientMock, createLoggerMock } from '../../mocks/index.js';

describeUnit('ValidationService', () => {
  let validationService;
  let mockDeepchecksClient;
  let mockLogger;

  beforeEach(() => {
    // London School: Mocks define collaboration contracts
    mockDeepchecksClient = createDeepchecksClientMock();
    mockLogger = createLoggerMock();

    validationService = new ValidationService({
      deepchecksClient: mockDeepchecksClient,
      logger: mockLogger
    });
  });

  describeCollaboration('Dataset Validation Workflow', () => {
    it('should validate dataset and coordinate with Deepchecks client', async () => {
      // Arrange
      const datasetPath = '/path/to/dataset';

      // Act
      const result = await validationService.validateDataset(datasetPath);

      // Assert - London School: Verify collaboration pattern
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating dataset with Deepchecks',
        { datasetPath }
      );
      expect(mockDeepchecksClient.runSuite).toHaveBeenCalled();
      
      // Verify result structure follows contract
      expect(result).toMatchObject({
        passed: true,
        score: 0.95,
        checks: expect.arrayContaining([
          expect.objectContaining({ name: 'data_integrity', passed: true }),
          expect.objectContaining({ name: 'label_distribution', passed: true })
        ])
      });
    });
  });

  describeCollaboration('Model Performance Validation', () => {
    it('should validate model performance and coordinate check execution', async () => {
      // Arrange
      const modelPath = '/path/to/model.pt';
      const testDataPath = '/path/to/test/data';

      // Act
      const result = await validationService.validateModel(modelPath, testDataPath);

      // Assert - Verify service coordination
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating model performance',
        { modelPath, testDataPath }
      );
      expect(mockDeepchecksClient.getCheckResults).toHaveBeenCalled();
      
      // London School: Focus on result structure and collaboration
      expect(result).toEqual({
        passed: true,
        performance: { accuracy: 0.92, precision: 0.89, recall: 0.94 },
        checks: ['model_performance', 'data_drift', 'feature_importance']
      });
    });
  });

  describeCollaboration('Report Generation Workflow', () => {
    it('should generate validation report and coordinate HTML generation', async () => {
      // Act
      const result = await validationService.generateReport();

      // Assert - Verify coordination with report generator
      expect(mockLogger.info).toHaveBeenCalledWith('Generating validation report');
      expect(mockDeepchecksClient.generateHtmlReport).toHaveBeenCalled();
      
      expect(result).toMatchObject({
        reportId: expect.stringMatching(/^validation-report-\d+$/),
        htmlPath: '/mock/deepchecks-report.html',
        summary: 'All validation checks passed'
      });
    });
  });

  describeCollaboration('Validation Workflow Sequence', () => {
    it('should coordinate validation before report generation', async () => {
      // Act - Execute validation workflow
      await validationService.validateDataset('/dataset');
      await validationService.generateReport();

      // Assert - London School: Verify interaction sequence
      expect(mockDeepchecksClient.runSuite)
        .toHaveBeenCalledBefore(mockDeepchecksClient.generateHtmlReport);
    });

    it('should coordinate model validation with performance checks', async () => {
      // Act
      await validationService.validateModel('/model', '/test-data');

      // Assert - Verify check coordination happens before result processing
      expect(mockDeepchecksClient.getCheckResults).toHaveBeenCalled();
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating model performance',
        expect.any(Object)
      );
    });
  });

  describeContract('ValidationService Contract', () => {
    it('should satisfy Deepchecks integration contract', () => {
      const expectedContract = {
        validateDataset: expect.any(Function),
        validateModel: expect.any(Function),
        generateReport: expect.any(Function)
      };

      expect(validationService).toSatisfyContract(expectedContract);
    });
  });

  describeCollaboration('Error Handling Coordination', () => {
    it('should handle validation failures gracefully', async () => {
      // Arrange - Mock a validation failure
      mockDeepchecksClient.runSuite.mockResolvedValue({ 
        suite_result: 'failed' 
      });

      // Act
      const result = await validationService.validateDataset('/bad-dataset');

      // Assert - Service should still coordinate properly during failures
      expect(mockDeepchecksClient.runSuite).toHaveBeenCalled();
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Validating dataset with Deepchecks',
        { datasetPath: '/bad-dataset' }
      );
      
      // London School: Focus on behavior, not internal state
      expect(result).toBeDefined();
    });
  });

  describeCollaboration('Complete Validation Pipeline', () => {
    it('should coordinate complete validation pipeline sequence', async () => {
      // Act - Execute complete validation pipeline
      const datasetResult = await validationService.validateDataset('/dataset');
      const modelResult = await validationService.validateModel('/model', '/test');
      const report = await validationService.generateReport([datasetResult, modelResult]);

      // Assert - London School: Verify all collaborations occurred in sequence
      expect(mockLogger.info).toHaveBeenCalledTimes(3);
      expect(mockDeepchecksClient.runSuite).toHaveBeenCalled();
      expect(mockDeepchecksClient.getCheckResults).toHaveBeenCalled();
      expect(mockDeepchecksClient.generateHtmlReport).toHaveBeenCalled();
      
      // Verify pipeline coordination
      expect(report.reportId).toBeDefined();
    });
  });
});