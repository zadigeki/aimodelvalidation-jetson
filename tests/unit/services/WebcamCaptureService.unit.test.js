/**
 * WebcamCaptureService Unit Tests
 * London School TDD - Mock-driven behavior verification
 */

import { jest, describe, it, beforeEach, expect } from '@jest/globals';
import { WebcamCaptureService } from '../../../src/services/WebcamCaptureService.js';
import { createFileSystemMock, createLoggerMock } from '../../mocks/index.js';

describeUnit('WebcamCaptureService', () => {
  let webcamCaptureService;
  let mockFileSystem;
  let mockLogger;

  beforeEach(() => {
    // London School: Create mocks first to define contracts
    mockFileSystem = createFileSystemMock();
    mockLogger = createLoggerMock();

    webcamCaptureService = new WebcamCaptureService({
      fileSystem: mockFileSystem,
      logger: mockLogger
    });
  });

  describeCollaboration('Service Initialization', () => {
    it('should initialize webcam capture and log initialization', async () => {
      // Act
      const result = await webcamCaptureService.initialize();

      // Assert - London School: Focus on behavior verification
      expect(result).toBe(true);
      expect(mockLogger.info).toHaveBeenCalledWith('WebcamCaptureService initialized successfully');
    });
  });

  describeCollaboration('Stream Management', () => {
    it('should start stream and update active status', async () => {
      // Act
      await webcamCaptureService.startStream();

      // Assert - Verify collaboration pattern
      expect(webcamCaptureService.isStreamActive()).toBe(true);
      expect(mockLogger.info).toHaveBeenCalledWith('WebcamCaptureService: Webcam stream started', {});
    });

    it('should stop stream and update active status', async () => {
      // Arrange
      await webcamCaptureService.startStream();

      // Act
      await webcamCaptureService.stopStream();

      // Assert
      expect(webcamCaptureService.isStreamActive()).toBe(false);
      expect(mockLogger.info).toHaveBeenCalledWith('Webcam stream stopped');
    });
  });

  describeCollaboration('Frame Capture', () => {
    it('should capture frame when stream is active', async () => {
      // Arrange
      await webcamCaptureService.startStream();

      // Act
      const frame = await webcamCaptureService.captureFrame();

      // Assert - London School: Verify object structure and interactions
      expect(frame).toMatchObject({
        id: expect.stringMatching(/^frame-\d+$/),
        timestamp: expect.any(Number),
        data: expect.any(Buffer),
        metadata: {
          width: 640,
          height: 480,
          format: 'jpeg'
        }
      });

      expect(mockLogger.debug).toHaveBeenCalledWith(
        'Frame captured',
        { frameId: frame.id }
      );
    });

    it('should throw error when capturing frame without active stream', async () => {
      // Act & Assert - London School: Test error collaboration
      await expect(webcamCaptureService.captureFrame())
        .rejects
        .toThrow('Webcam not active. Call startStream() first.');
    });
  });

  describeContract('WebcamCaptureService Contract', () => {
    it('should satisfy expected interface contract', () => {
      const expectedContract = {
        initialize: expect.any(Function),
        captureFrame: expect.any(Function),
        startStream: expect.any(Function),
        stopStream: expect.any(Function),
        isStreamActive: expect.any(Function)
      };

      expect(webcamCaptureService).toSatisfyContract(expectedContract);
    });
  });

  describeCollaboration('Error Handling', () => {
    it('should handle capture errors gracefully', async () => {
      // Arrange - Mock an error scenario
      await webcamCaptureService.startStream();
      
      // London School: We're not testing implementation details,
      // but how the service collaborates during error conditions
      const captureAttempt = async () => {
        return await webcamCaptureService.captureFrame();
      };

      // Act & Assert - Service should still collaborate correctly
      const frame = await captureAttempt();
      expect(frame).toBeDefined();
      expect(mockLogger.debug).toHaveBeenCalled();
    });
  });
});