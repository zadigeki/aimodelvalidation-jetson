/**
 * AnnotationService Unit Tests
 * London School TDD - CVAT integration with mock-driven behavior verification
 */

import { jest, describe, it, beforeEach, expect } from '@jest/globals';
import { AnnotationService } from '../../../src/services/AnnotationService.js';
import { createCvatClientMock, createLoggerMock } from '../../mocks/index.js';

describeUnit('AnnotationService', () => {
  let annotationService;
  let mockCvatClient;
  let mockLogger;

  beforeEach(() => {
    // London School: Mock contracts define expected collaborations
    mockCvatClient = createCvatClientMock();
    mockLogger = createLoggerMock();

    annotationService = new AnnotationService({
      cvatClient: mockCvatClient,
      logger: mockLogger
    });
  });

  describeCollaboration('Project Management', () => {
    it('should create annotation project and coordinate with CVAT client', async () => {
      // Arrange
      const projectConfig = { name: 'test-cv-project' };

      // Act
      const result = await annotationService.createProject(projectConfig);

      // Assert - London School: Verify collaboration patterns
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Creating annotation project',
        { name: projectConfig.name }
      );
      expect(mockCvatClient.createTask).toHaveBeenCalled();
      expect(result.projectId).toMatch(/^cvat-project-\d+$/);
    });
  });

  describeCollaboration('Data Upload Workflow', () => {
    it('should upload data and coordinate CVAT image upload', async () => {
      // Act
      const result = await annotationService.uploadData();

      // Assert - Verify service coordination
      expect(mockLogger.info).toHaveBeenCalledWith('Uploading data to CVAT');
      expect(mockCvatClient.uploadImages).toHaveBeenCalled();
      expect(result.taskId).toMatch(/^cvat-task-\d+$/);
    });

    it('should coordinate upload before retrieval workflow', async () => {
      // Arrange - Set up workflow sequence
      await annotationService.uploadData();
      
      // Act
      await annotationService.getAnnotations('task-123');

      // Assert - London School: Verify call sequence
      expect(mockCvatClient.uploadImages)
        .toHaveBeenCalledBefore(mockLogger.info.mock.calls.find(
          call => call[0] === 'Retrieving annotations'
        ));
    });
  });

  describeCollaboration('Annotation Retrieval', () => {
    it('should retrieve annotations with proper logging coordination', async () => {
      // Arrange
      const taskId = 'test-task-456';

      // Act
      const result = await annotationService.getAnnotations(taskId);

      // Assert - Verify logging and data structure
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Retrieving annotations',
        { taskId }
      );
      expect(result.annotations).toEqual([
        { id: 1, label: 'object', bbox: [10, 10, 50, 50] }
      ]);
    });
  });

  describeCollaboration('Annotation Export Workflow', () => {
    it('should export annotations in specified format', async () => {
      // Act
      const result = await annotationService.exportAnnotations('YOLO');

      // Assert - London School: Focus on object interaction
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Exporting annotations',
        { format: 'YOLO' }
      );
      expect(result).toEqual({
        format: 'YOLO',
        data: 'mock-export-data'
      });
    });

    it('should default to YOLO format when no format specified', async () => {
      // Act
      const result = await annotationService.exportAnnotations();

      // Assert
      expect(result.format).toBe('YOLO');
    });
  });

  describeCollaboration('Annotation Validation', () => {
    it('should validate annotations and provide validation result', async () => {
      // Act
      const result = await annotationService.validateAnnotations();

      // Assert - Verify validation collaboration
      expect(mockLogger.info).toHaveBeenCalledWith('Validating annotations');
      expect(result).toEqual({
        valid: true,
        errors: []
      });
    });
  });

  describeContract('AnnotationService Contract', () => {
    it('should satisfy CVAT integration contract', () => {
      const expectedContract = {
        createProject: expect.any(Function),
        uploadData: expect.any(Function),
        getAnnotations: expect.any(Function),
        exportAnnotations: expect.any(Function),
        validateAnnotations: expect.any(Function)
      };

      expect(annotationService).toSatisfyContract(expectedContract);
    });
  });

  describeCollaboration('Complete Annotation Workflow', () => {
    it('should coordinate complete annotation workflow sequence', async () => {
      // Arrange
      const projectConfig = { name: 'workflow-test' };

      // Act - Execute complete workflow
      await annotationService.createProject(projectConfig);
      await annotationService.uploadData();
      await annotationService.getAnnotations('task-id');
      await annotationService.validateAnnotations();
      await annotationService.exportAnnotations('YOLO');

      // Assert - London School: Verify collaboration sequence
      expect(mockCvatClient.createTask).toHaveBeenCalledBefore(mockCvatClient.uploadImages);
      
      // Verify all workflow steps were logged
      expect(mockLogger.info).toHaveBeenCalledTimes(5);
    });
  });
});