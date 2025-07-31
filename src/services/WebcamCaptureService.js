/**
 * WebcamCaptureService - Data capture implementation
 * London School TDD - Refactored with common patterns
 */

import { ServiceBase } from '../common/ServiceBase.js';

export class WebcamCaptureService extends ServiceBase {
  constructor(dependencies) {
    super(dependencies);
    this._validateRequiredDependencies(['fileSystem', 'logger']);
    this.fileSystem = dependencies.fileSystem;
    this.isActive = false;
  }

  async _doInitialize() {
    // Service-specific initialization logic
    this._logOperation('Initializing webcam capture service');
  }

  async captureFrame() {
    if (!this.isActive) {
      throw new Error('Webcam not active. Call startStream() first.');
    }

    const frame = {
      id: `frame-${Date.now()}`,
      timestamp: Date.now(),
      data: Buffer.from('mock-image-data'),
      metadata: { width: 640, height: 480, format: 'jpeg' }
    };

    this.logger.debug('Frame captured', { frameId: frame.id });
    return frame;
  }

  async startStream() {
    try {
      this.isActive = true;
      this._logOperation('Webcam stream started');
      return true;
    } catch (error) {
      this._logError('startStream', error);
      throw error;
    }
  }

  async stopStream() {
    try {
      this.isActive = false;
      this._logOperation('Webcam stream stopped');
      return true;
    } catch (error) {
      this._logError('stopStream', error);
      throw error;
    }
  }

  isStreamActive() {
    return this.isActive;
  }
}