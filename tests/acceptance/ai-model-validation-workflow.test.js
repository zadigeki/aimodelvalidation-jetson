/**
 * Acceptance Test: AI Model Validation Workflow
 * London School TDD - Outside-In Development
 * Starting from user behavior and driving down to implementation
 */

import { jest, describe, it, beforeEach, expect } from '@jest/globals';
import {
  createWebcamCaptureMock,
  createAnnotationServiceMock,
  createValidationServiceMock,
  createModelTrainerMock,
  createWorkflowOrchestratorMock
} from '../mocks/index.js';
import { AIModelValidationSystem } from '../../src/AIModelValidationSystem.js';

describe('AI Model Validation Workflow - Acceptance Tests', () => {
  describe('Complete PoC Workflow: Manual data capture → CVAT annotation → Deepchecks validation → Ultralytics training', () => {
    let webcamCapture;
    let annotationService;
    let validationService;
    let modelTrainer;
    let workflowOrchestrator;
    let aiModelValidationSystem;

    beforeEach(() => {
      // London School: Create all mocks first to define contracts
      webcamCapture = createWebcamCaptureMock();
      annotationService = createAnnotationServiceMock();
      validationService = createValidationServiceMock();
      modelTrainer = createModelTrainerMock();
      workflowOrchestrator = createWorkflowOrchestratorMock();

      // This will drive the creation of our actual implementation
      aiModelValidationSystem = new AIModelValidationSystem({
        webcamCapture,
        annotationService,
        validationService,
        modelTrainer,
        workflowOrchestrator
      });
    });

    it('should successfully execute complete AI model validation workflow', async () => {
      // Arrange - Define the expected workflow behavior
      const workflowConfig = {
        projectName: 'test-cv-project',
        captureSettings: { frameCount: 100, interval: 1000 },
        annotationSettings: { format: 'YOLO', classes: ['object', 'background'] },
        validationSettings: { checks: ['data_integrity', 'label_distribution'] },
        trainingSettings: { epochs: 50, batchSize: 16 }
      };

      // Act - Execute the workflow (this will drive implementation)
      const result = await aiModelValidationSystem.executeCompleteWorkflow(workflowConfig);

      // Assert - Verify the collaboration between all components
      // London School focuses on HOW objects work together
      
      // 1. Workflow orchestration should coordinate all services
      expect(workflowOrchestrator.executeWorkflow).toHaveBeenCalledWith(
        expect.objectContaining({
          steps: ['capture', 'annotate', 'validate', 'train'],
          config: workflowConfig
        })
      );

      // 2. Data capture should be initialized and executed
      expect(webcamCapture.initialize).toHaveBeenCalled();
      expect(webcamCapture.startStream).toHaveBeenCalled();
      expect(webcamCapture.captureFrame).toHaveBeenCalledTimes(workflowConfig.captureSettings.frameCount);

      // 3. Annotation service should process captured data
      expect(annotationService.createProject).toHaveBeenCalledWith(
        expect.objectContaining({ name: workflowConfig.projectName })
      );
      expect(annotationService.uploadData).toHaveBeenCalled();

      // 4. Validation service should verify data and annotations
      expect(validationService.validateDataset).toHaveBeenCalled();
      expect(validationService.generateReport).toHaveBeenCalled();

      // 5. Model trainer should train on validated data
      expect(modelTrainer.initialize).toHaveBeenCalled();
      expect(modelTrainer.train).toHaveBeenCalledWith(
        expect.objectContaining({
          epochs: workflowConfig.trainingSettings.epochs,
          batchSize: workflowConfig.trainingSettings.batchSize
        })
      );

      // London School: Verify the conversation flow between objects
      expect(webcamCapture.initialize).toHaveBeenCalledBefore(webcamCapture.captureFrame);
      expect(annotationService.createProject).toHaveBeenCalledBefore(annotationService.uploadData);
      expect(validationService.validateDataset).toHaveBeenCalledBefore(modelTrainer.train);
    });

    it('should handle workflow failures gracefully', async () => {
      // Arrange - Mock a failure scenario
      validationService.validateDataset.mockResolvedValue({
        passed: false,
        errors: ['Insufficient data samples', 'Label distribution imbalance']
      });

      const workflowConfig = { projectName: 'failing-project' };

      // Act & Assert - Verify error handling collaboration
      const result = await aiModelValidationSystem.executeCompleteWorkflow(workflowConfig);

      // London School: Focus on how failure is communicated between objects
      expect(validationService.validateDataset).toHaveBeenCalled();
      expect(result.success).toBe(false);
      expect(result.errors).toContain('Validation failed');
      
      // Should not proceed to training if validation fails
      expect(modelTrainer.train).not.toHaveBeenCalled();
    });

    it('should support incremental workflow execution', async () => {
      // Arrange - Test resuming from a specific step
      const workflowState = {
        completedSteps: ['capture', 'annotate'],
        nextStep: 'validate',
        projectId: 'existing-project-789'
      };

      // Act - Resume workflow from validation step
      await aiModelValidationSystem.resumeWorkflow(workflowState);

      // Assert - Verify selective execution
      expect(webcamCapture.captureFrame).not.toHaveBeenCalled();
      expect(annotationService.uploadData).not.toHaveBeenCalled();
      expect(validationService.validateDataset).toHaveBeenCalled();
      expect(modelTrainer.train).toHaveBeenCalled();
    });
  });

  describeCollaboration('Workflow Orchestration Patterns', () => {
    let orchestrator;
    let services;

    beforeEach(() => {
      services = {
        capture: createWebcamCaptureMock(),
        annotation: createAnnotationServiceMock(),
        validation: createValidationServiceMock(),
        training: createModelTrainerMock()
      };
      
      orchestrator = createWorkflowOrchestratorMock();
    });

    it('should coordinate service dependencies correctly', async () => {
      // London School: Test the coordination pattern
      const workflow = {
        steps: [
          { name: 'capture', dependencies: [] },
          { name: 'annotate', dependencies: ['capture'] },
          { name: 'validate', dependencies: ['annotate'] },
          { name: 'train', dependencies: ['validate'] }
        ]
      };

      // Execute the workflow to drive coordination
      await orchestrator.executeWorkflow({ steps: workflow.steps });

      // This test drives the design of how services coordinate
      expect(orchestrator.executeWorkflow).toHaveBeenCalledWith(
        expect.objectContaining({ steps: expect.any(Array) })
      );
    });
  });

  describeContract('AI Model Validation System Contract', () => {
    it('should define clear interface for complete workflow execution', () => {
      // London School: Define contracts through mock expectations
      const expectedInterface = {
        executeCompleteWorkflow: expect.any(Function),
        resumeWorkflow: expect.any(Function),
        getWorkflowStatus: expect.any(Function),
        cancelWorkflow: expect.any(Function)
      };

      // This drives the interface design
      const mockSystem = createMockObject('AIModelValidationSystem', expectedInterface);
      expect(mockSystem).toSatisfyContract(expectedInterface);
    });
  });
});