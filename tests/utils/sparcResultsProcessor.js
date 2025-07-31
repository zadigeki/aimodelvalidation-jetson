/**
 * SPARC Test Results Processor
 * 
 * Processes Jest test results to provide SPARC methodology insights
 * and track progress through specification, pseudocode, architecture, 
 * refinement, and completion phases.
 */

import fs from 'fs';
import path from 'path';

/**
 * Processes test results and generates SPARC methodology reports
 * @param {Object} results - Jest test results
 * @returns {Object} - Processed results with SPARC insights
 */
export default function sparcResultsProcessor(results) {
  const sparcReport = generateSparcReport(results);
  const londonSchoolMetrics = generateLondonSchoolMetrics(results);
  
  // Write SPARC progress report
  const reportPath = path.join(process.cwd(), 'coverage', 'sparc-report.json');
  fs.writeFileSync(reportPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    sparcReport,
    londonSchoolMetrics,
    rawResults: {
      numTotalTests: results.numTotalTests,
      numPassedTests: results.numPassedTests, 
      numFailedTests: results.numFailedTests,
      numPendingTests: results.numPendingTests
    }
  }, null, 2));

  // Generate human-readable SPARC progress
  generateSparcProgressReport(sparcReport, londonSchoolMetrics);
  
  return results;
}

function generateSparcReport(results) {
  const sparcPhases = {
    specification: { tests: [], passed: 0, failed: 0, total: 0 },
    pseudocode: { tests: [], passed: 0, failed: 0, total: 0 },
    architecture: { tests: [], passed: 0, failed: 0, total: 0 },
    refinement: { tests: [], passed: 0, failed: 0, total: 0 },
    completion: { tests: [], passed: 0, failed: 0, total: 0 }
  };

  results.testResults.forEach(testFile => {
    testFile.assertionResults.forEach(test => {
      const phase = identifySparcPhase(testFile.testFilePath, test.title);
      sparcPhases[phase].tests.push({
        title: test.title,
        status: test.status,
        filePath: testFile.testFilePath,
        duration: test.duration
      });
      
      sparcPhases[phase].total++;
      if (test.status === 'passed') {
        sparcPhases[phase].passed++;
      } else if (test.status === 'failed') {
        sparcPhases[phase].failed++;
      }
    });
  });

  return sparcPhases;
}

function generateLondonSchoolMetrics(results) {
  let mockInteractions = 0;
  let behaviorVerifications = 0;
  let contractTests = 0;
  let outsideInTests = 0;

  results.testResults.forEach(testFile => {
    testFile.assertionResults.forEach(test => {
      // Analyze test content for London School patterns
      if (test.title.includes('mock') || test.title.includes('Mock')) {
        mockInteractions++;
      }
      
      if (test.title.includes('should') && (
          test.title.includes('interact') || 
          test.title.includes('coordinate') ||
          test.title.includes('collaborate')
        )) {
        behaviorVerifications++;
      }
      
      if (testFile.testFilePath.includes('contract')) {
        contractTests++;
      }
      
      if (testFile.testFilePath.includes('acceptance')) {
        outsideInTests++;
      }
    });
  });

  return {
    mockInteractions,
    behaviorVerifications, 
    contractTests,
    outsideInTests,
    totalTests: results.numTotalTests,
    mockCoverage: (mockInteractions / results.numTotalTests * 100).toFixed(1),
    behaviorCoverage: (behaviorVerifications / results.numTotalTests * 100).toFixed(1)
  };
}

function identifySparcPhase(filePath, testTitle) {
  // Identify SPARC phase based on file path and test content
  if (filePath.includes('acceptance')) {
    return 'specification';
  }
  
  if (filePath.includes('contract')) {
    return 'architecture';
  }
  
  if (filePath.includes('integration')) {
    return 'completion';
  }
  
  if (testTitle.includes('pseudocode') || testTitle.includes('algorithm')) {
    return 'pseudocode';
  }
  
  // Default to refinement phase for unit tests
  return 'refinement';
}

function generateSparcProgressReport(sparcReport, londonMetrics) {
  const progressPath = path.join(process.cwd(), 'coverage', 'sparc-progress.md');
  
  const content = `# SPARC Methodology Progress Report
Generated: ${new Date().toISOString()}

## London School TDD Metrics
- **Mock Interactions**: ${londonMetrics.mockInteractions} (${londonMetrics.mockCoverage}% coverage)
- **Behavior Verifications**: ${londonMetrics.behaviorVerifications} (${londonMetrics.behaviorCoverage}% coverage)
- **Contract Tests**: ${londonMetrics.contractTests}
- **Outside-In Tests**: ${londonMetrics.outsideInTests}

## SPARC Phase Progress

### 📋 Specification Phase (Acceptance Tests)
- **Tests**: ${sparcReport.specification.total}
- **Passed**: ${sparcReport.specification.passed} ✅
- **Failed**: ${sparcReport.specification.failed} ❌
- **Success Rate**: ${sparcReport.specification.total > 0 ? (sparcReport.specification.passed / sparcReport.specification.total * 100).toFixed(1) : 0}%

### 🧮 Pseudocode Phase (Algorithm Tests)  
- **Tests**: ${sparcReport.pseudocode.total}
- **Passed**: ${sparcReport.pseudocode.passed} ✅
- **Failed**: ${sparcReport.pseudocode.failed} ❌
- **Success Rate**: ${sparcReport.pseudocode.total > 0 ? (sparcReport.pseudocode.passed / sparcReport.pseudocode.total * 100).toFixed(1) : 0}%

### 🏗️ Architecture Phase (Contract Tests)
- **Tests**: ${sparcReport.architecture.total}
- **Passed**: ${sparcReport.architecture.passed} ✅  
- **Failed**: ${sparcReport.architecture.failed} ❌
- **Success Rate**: ${sparcReport.architecture.total > 0 ? (sparcReport.architecture.passed / sparcReport.architecture.total * 100).toFixed(1) : 0}%

### 🔧 Refinement Phase (Unit Tests)
- **Tests**: ${sparcReport.refinement.total}
- **Passed**: ${sparcReport.refinement.passed} ✅
- **Failed**: ${sparcReport.refinement.failed} ❌
- **Success Rate**: ${sparcReport.refinement.total > 0 ? (sparcReport.refinement.passed / sparcReport.refinement.total * 100).toFixed(1) : 0}%

### ✅ Completion Phase (Integration Tests)
- **Tests**: ${sparcReport.completion.total}
- **Passed**: ${sparcReport.completion.passed} ✅
- **Failed**: ${sparcReport.completion.failed} ❌
- **Success Rate**: ${sparcReport.completion.total > 0 ? (sparcReport.completion.passed / sparcReport.completion.total * 100).toFixed(1) : 0}%

## Overall Progress
- **Total Tests**: ${londonMetrics.totalTests}
- **Overall Success Rate**: ${londonMetrics.totalTests > 0 ? ((sparcReport.specification.passed + sparcReport.pseudocode.passed + sparcReport.architecture.passed + sparcReport.refinement.passed + sparcReport.completion.passed) / londonMetrics.totalTests * 100).toFixed(1) : 0}%

## Recommendations

### London School TDD Focus Areas:
${londonMetrics.mockCoverage < 60 ? '- ⚠️ **Increase mock usage**: Consider more mock-first development' : '- ✅ **Good mock coverage**: Mock-first approach is being followed'}
${londonMetrics.behaviorCoverage < 50 ? '- ⚠️ **Add behavior verification**: Focus more on interaction testing' : '- ✅ **Good behavior coverage**: Interaction testing is well implemented'}
${londonMetrics.contractTests < 5 ? '- ⚠️ **Add contract tests**: Improve service boundary testing' : '- ✅ **Good contract coverage**: Service boundaries are well tested'}

### SPARC Methodology Progress:
${sparcReport.specification.total === 0 ? '- 🔴 **Start with acceptance tests**: Begin with specification phase' : '- ✅ **Specification phase active**: Acceptance tests are in place'}
${sparcReport.architecture.total < 3 ? '- 🟡 **Expand architecture tests**: Add more contract/boundary tests' : '- ✅ **Architecture phase covered**: Contract tests are comprehensive'}
${sparcReport.refinement.total < 10 ? '- 🟡 **Add unit tests**: Expand refinement phase testing' : '- ✅ **Refinement phase active**: Unit tests are comprehensive'}

---
*Generated by SPARC Test Results Processor - London School TDD*
`;

  fs.writeFileSync(progressPath, content);
  console.log(`\n📊 SPARC Progress Report generated: ${progressPath}`);
  console.log(`🧪 London School TDD Metrics: ${londonMetrics.mockCoverage}% mock coverage, ${londonMetrics.behaviorCoverage}% behavior coverage`);
}