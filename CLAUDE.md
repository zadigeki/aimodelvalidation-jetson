# Claude Code Configuration - SPARC Development Environment (Batchtools Optimized)

## 🚨 CRITICAL: CONCURRENT EXECUTION FOR ALL ACTIONS

**ABSOLUTE RULE**: ALL operations MUST be concurrent/parallel in a single message:

### 🔴 MANDATORY CONCURRENT PATTERNS:
1. **TodoWrite**: ALWAYS batch ALL todos in ONE call (5-10+ todos minimum)
2. **Task tool**: ALWAYS spawn ALL agents in ONE message with full instructions
3. **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
4. **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
5. **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**Examples of CORRECT concurrent execution:**
```javascript
// ✅ CORRECT: Everything in ONE message
[Single Message]:
  - TodoWrite { todos: [10+ todos with all statuses/priorities] }
  - Task("Agent 1 with full instructions and hooks")
  - Task("Agent 2 with full instructions and hooks")
  - Task("Agent 3 with full instructions and hooks")
  - Read("file1.js")
  - Read("file2.js")
  - Write("output1.js", content)
  - Write("output2.js", content)
  - Bash("npm install")
  - Bash("npm test")
  - Bash("npm run build")
```

**Examples of WRONG sequential execution:**
```javascript
// ❌ WRONG: Multiple messages (NEVER DO THIS)
Message 1: TodoWrite { todos: [single todo] }
Message 2: Task("Agent 1")
Message 3: Task("Agent 2")
Message 4: Read("file1.js")
Message 5: Write("output1.js")
Message 6: Bash("npm install")
// This is 6x slower and breaks coordination!
```

### 🎯 CONCURRENT EXECUTION CHECKLIST:

Before sending ANY message, ask yourself:
- ✅ Are ALL related TodoWrite operations batched together?
- ✅ Are ALL Task spawning operations in ONE message?
- ✅ Are ALL file operations (Read/Write/Edit) batched together?
- ✅ Are ALL bash commands grouped in ONE message?
- ✅ Are ALL memory operations concurrent?

If ANY answer is "No", you MUST combine operations into a single message!

## Project Overview
This project uses the SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology for systematic Test-Driven Development with AI assistance through Claude-Flow orchestration.

**🚀 Batchtools Optimization Enabled**: This configuration includes optimized prompts and parallel processing capabilities for improved performance and efficiency.

## SPARC Development Commands

### Core SPARC Commands
- `npx claude-flow sparc modes`: List all available SPARC development modes
- `npx claude-flow sparc run <mode> "<task>"`: Execute specific SPARC mode for a task
- `npx claude-flow sparc tdd "<feature>"`: Run complete TDD workflow using SPARC methodology
- `npx claude-flow sparc info <mode>`: Get detailed information about a specific mode

### Batchtools Commands (Optimized)
- `npx claude-flow sparc batch <modes> "<task>"`: Execute multiple SPARC modes in parallel
- `npx claude-flow sparc pipeline "<task>"`: Execute full SPARC pipeline with parallel processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"`: Process multiple tasks concurrently

### Standard Build Commands
- `npm run build`: Build the project
- `npm run test`: Run the test suite
- `npm run lint`: Run linter and format checks
- `npm run typecheck`: Run TypeScript type checking

## 🎯 SPARC+TDD INTEGRATED METHODOLOGY (London School Enhanced)

### 🚨 CRITICAL: SPARC Phases with Embedded TDD Cycles

**ABSOLUTE RULE**: Each SPARC phase MUST include Red-Green-Refactor cycles with mock-first testing:

### 📋 SPARC PHASE OVERVIEW WITH TDD INTEGRATION

```
🔄 SPARC-TDD CYCLE:
Specification → Write Tests → Pseudocode → Write Tests → Architecture → Write Tests → Refinement (TDD) → Completion (Integration Tests)

 Each phase includes:
 ├── 🔴 RED: Write failing tests first
 ├── 🟢 GREEN: Implement minimal code to pass
 ├── 🔵 REFACTOR: Improve code quality
 └── 🔄 REPEAT: Until phase complete
```

## 🔥 PHASE-BY-PHASE SPARC+TDD WORKFLOW

### 1. 📋 SPECIFICATION PHASE + TDD

**Objective**: Define requirements WITH testable acceptance criteria

#### TDD Integration in Specification:
```bash
# Concurrent specification with test scenario generation
npx claude-flow sparc run specification "user authentication system" --tdd-scenarios --parallel
```

#### Phase Activities (All Concurrent):
- **Requirements Analysis** (parallel requirement sources)
- **Acceptance Criteria Definition** (with test scenarios)
- **Edge Case Identification** (with error test cases)
- **Test Scenario Creation** (BDD-style scenarios)
- **Mock Strategy Planning** (London School preparation)

#### Quality Gate 1: Specification Complete
- ✅ All requirements documented with test scenarios
- ✅ Acceptance criteria are testable
- ✅ Mock strategy defined
- ✅ Test scenarios cover happy path + edge cases

#### Concurrent Commands:
```bash
# Run ALL specification activities in parallel
[BatchTool]:
  - Write("specs/requirements.md", requirementContent)
  - Write("specs/acceptance-criteria.md", criteriaContent) 
  - Write("specs/test-scenarios.md", scenarioContent)
  - Write("specs/mock-strategy.md", mockContent)
  - Bash("mkdir -p specs/{requirements,acceptance,scenarios,mocks}")
```

### 2. 🧠 PSEUDOCODE PHASE + TDD

**Objective**: Algorithm design WITH test-driven logic validation

#### TDD Integration in Pseudocode:
```bash
# Algorithm design with parallel test logic creation
npx claude-flow sparc run pseudocode "authentication flow" --test-logic --mock-design --parallel
```

#### Phase Activities (All Concurrent):
- **Algorithm Structure Design** (with test points)
- **Data Flow Mapping** (with mock boundaries)
- **Logic Validation** (through test scenarios)
- **Mock Interface Design** (London School interfaces)
- **Test Logic Creation** (parallel algorithm validation)

#### Quality Gate 2: Pseudocode + Test Logic Complete
- ✅ Algorithms validated through test logic
- ✅ Mock interfaces designed
- ✅ Data structures optimized for testability
- ✅ Edge cases covered in pseudocode

#### Concurrent Commands:
```bash
# Run ALL pseudocode activities in parallel
[BatchTool]:
  - Write("pseudocode/algorithms.md", algorithmContent)
  - Write("pseudocode/test-logic.md", testLogicContent)
  - Write("pseudocode/mock-interfaces.md", mockInterfaceContent)
  - Write("pseudocode/data-flow.md", dataFlowContent)
  - Bash("mkdir -p pseudocode/{algorithms,test-logic,mocks,data-flow}")
```

### 3. 🏗️ ARCHITECTURE PHASE + TDD

**Objective**: System design WITH testable component boundaries

#### TDD Integration in Architecture:
```bash
# Architecture design with parallel component testing strategy
npx claude-flow sparc run architecture "system design" --component-tests --integration-strategy --parallel
```

#### Phase Activities (All Concurrent):
- **Component Design** (with test boundaries)
- **Interface Contracts** (mock-friendly interfaces)
- **Integration Planning** (test integration points)
- **Dependency Injection Setup** (for London School mocking)
- **Test Architecture** (parallel test strategy design)

#### Quality Gate 3: Architecture + Test Strategy Complete
- ✅ Components designed for testability
- ✅ Interfaces support mocking
- ✅ Integration points identified
- ✅ Test architecture planned

#### Concurrent Commands:
```bash
# Run ALL architecture activities in parallel
[BatchTool]:
  - Write("architecture/components.md", componentContent)
  - Write("architecture/interfaces.md", interfaceContent)
  - Write("architecture/integration.md", integrationContent)
  - Write("architecture/test-strategy.md", testStrategyContent)
  - Bash("mkdir -p architecture/{components,interfaces,integration,test-strategy}")
```

### 4. 🔧 REFINEMENT PHASE (PURE TDD IMPLEMENTATION)

**Objective**: Implementation using strict Red-Green-Refactor cycles

#### London School TDD Process:
```bash
# Full TDD implementation with mock-first development
npx claude-flow sparc tdd "implement authentication" --london-school --parallel-testing
```

#### Phase Activities (Red-Green-Refactor Cycles):
1. **🔴 RED**: Write failing test with mocks
2. **🟢 GREEN**: Implement minimal code to pass
3. **🔵 REFACTOR**: Improve code quality
4. **🔄 REPEAT**: For each component

#### TDD Cycle Commands (Concurrent per Feature):
```bash
# Execute TDD cycles for multiple features in parallel
[BatchTool]:
  - Write("tests/auth.test.js", failingAuthTest)     # RED
  - Write("tests/user.test.js", failingUserTest)     # RED
  - Write("tests/token.test.js", failingTokenTest)   # RED
  - Bash("npm test -- --watch")                      # Run tests
  - Write("src/auth.js", minimalAuthImpl)            # GREEN
  - Write("src/user.js", minimalUserImpl)            # GREEN
  - Write("src/token.js", minimalTokenImpl)          # GREEN
```

#### Quality Gate 4: TDD Implementation Complete
- ✅ All tests passing (100% pass rate)
- ✅ Code coverage > 90%
- ✅ All mocks properly isolated
- ✅ Refactoring maintains test suite

### 5. ✅ COMPLETION PHASE + INTEGRATION TESTING

**Objective**: Integration testing with end-to-end validation

#### Integration Testing Strategy:
```bash
# Integration testing with parallel validation and documentation
npx claude-flow sparc run integration "full system test" --e2e-tests --parallel-validation
```

#### Phase Activities (All Concurrent):
- **Integration Test Suite** (component integration validation)
- **End-to-End Testing** (full workflow testing)
- **Performance Testing** (parallel load testing)
- **Documentation Generation** (concurrent doc creation)
- **Deployment Preparation** (production readiness)

#### Quality Gate 5: System Integration Complete
- ✅ All integration tests passing
- ✅ E2E scenarios validated
- ✅ Performance benchmarks met
- ✅ Documentation complete
- ✅ Production ready

#### Concurrent Commands:
```bash
# Run ALL completion activities in parallel
[BatchTool]:
  - Write("tests/integration/", integrationTests)
  - Write("tests/e2e/", e2eTests)
  - Write("docs/api.md", apiDocumentation)
  - Write("docs/deployment.md", deploymentGuide)
  - Bash("npm run test:integration")
  - Bash("npm run test:e2e")
  - Bash("npm run build:production")
```

## 🧪 LONDON SCHOOL TDD INTEGRATION

### 🎯 Mock-First Development Strategy

**CRITICAL PRINCIPLE**: London School TDD prioritizes interaction testing over state testing

#### Core London School Concepts:
- **Mock-First**: Create mocks before implementation
- **Interaction Testing**: Verify how objects collaborate
- **Outside-In**: Start with external interfaces, work inward
- **Isolated Unit Testing**: Each test focuses on one unit in isolation

### 🔧 TDD Tools Configuration

#### Essential Testing Stack:
```javascript
// package.json testing dependencies (concurrent installation)
{
  "devDependencies": {
    "jest": "^29.7.0",
    "@jest/globals": "^29.7.0",
    "jest-mock-extended": "^3.0.5",
    "supertest": "^6.3.3",
    "@testing-library/jest-dom": "^6.1.4",
    "sinon": "^17.0.1",
    "nock": "^13.4.0",
    "msw": "^2.0.9"
  }
}
```

#### Jest Configuration for London School:
```javascript
// jest.config.js - Optimized for mock-first testing
module.exports = {
  testEnvironment: 'node',
  collectCoverage: true,
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    }
  },
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  testMatch: [
    '<rootDir>/tests/**/*.test.js',
    '<rootDir>/tests/**/*.spec.js'
  ],
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true
};
```

### 🔄 RED-GREEN-REFACTOR CYCLE COMMANDS

#### Automated TDD Cycle Execution:
```bash
# Concurrent TDD cycle execution for multiple features
[BatchTool]:
  # RED Phase - Write failing tests
  - Write("tests/unit/auth.test.js", redPhaseAuthTest)
  - Write("tests/unit/user.test.js", redPhaseUserTest)
  - Write("tests/unit/token.test.js", redPhaseTokenTest)
  
  # Verify tests fail
  - Bash("npm run test:watch -- --testPathPattern=auth")
  - Bash("npm run test:watch -- --testPathPattern=user")
  - Bash("npm run test:watch -- --testPathPattern=token")
  
  # GREEN Phase - Minimal implementation
  - Write("src/auth.js", greenPhaseAuthImpl)
  - Write("src/user.js", greenPhaseUserImpl)
  - Write("src/token.js", greenPhaseTokenImpl)
  
  # REFACTOR Phase - Improve code
  - Edit("src/auth.js", refactoredAuthImpl)
  - Edit("src/user.js", refactoredUserImpl)
  - Edit("src/token.js", refactoredTokenImpl)
```

### 🎭 MOCK STRATEGY PATTERNS

#### Pattern 1: Dependency Injection for Testability
```javascript
// Example: Authentication service with injected dependencies
class AuthService {
  constructor(userRepository, tokenService, emailService) {
    this.userRepository = userRepository;
    this.tokenService = tokenService;
    this.emailService = emailService;
  }
  
  async authenticate(email, password) {
    // Implementation that can be fully mocked
  }
}
```

#### Pattern 2: Mock-First Test Structure
```javascript
// London School test example
describe('AuthService', () => {
  let authService;
  let mockUserRepository;
  let mockTokenService;
  let mockEmailService;
  
  beforeEach(() => {
    // Create mocks FIRST
    mockUserRepository = jest.createMockFromModule('../repositories/UserRepository');
    mockTokenService = jest.createMockFromModule('../services/TokenService');
    mockEmailService = jest.createMockFromModule('../services/EmailService');
    
    // Inject mocks
    authService = new AuthService(mockUserRepository, mockTokenService, mockEmailService);
  });
  
  it('should authenticate valid user', async () => {
    // Arrange - Setup mock expectations
    mockUserRepository.findByEmail.mockResolvedValue({ id: 1, password: 'hashed' });
    mockTokenService.generateToken.mockReturnValue('jwt-token');
    
    // Act
    const result = await authService.authenticate('user@example.com', 'password');
    
    // Assert - Verify interactions
    expect(mockUserRepository.findByEmail).toHaveBeenCalledWith('user@example.com');
    expect(mockTokenService.generateToken).toHaveBeenCalledWith({ id: 1 });
    expect(result).toEqual({ token: 'jwt-token' });
  });
});
```

### 📊 SPARC-TDD SUCCESS METRICS

#### Phase-Specific Metrics:

**Specification Phase:**
- ✅ Test Scenario Coverage: 100% of requirements have test scenarios
- ✅ Acceptance Criteria Testability: All criteria can be automated
- ✅ Mock Strategy Completeness: All external dependencies identified

**Pseudocode Phase:**
- ✅ Algorithm Test Logic: Every algorithm has corresponding test logic
- ✅ Mock Interface Design: All interfaces support mocking
- ✅ Edge Case Coverage: Test scenarios cover all edge cases

**Architecture Phase:**
- ✅ Testable Component Design: All components have clear test boundaries
- ✅ Dependency Injection: All dependencies can be mocked
- ✅ Integration Test Points: All integration points identified

**Refinement Phase:**
- ✅ Test Coverage: >90% line coverage, >85% branch coverage
- ✅ Test Quality: All tests follow London School principles
- ✅ Red-Green-Refactor Compliance: Strict TDD cycle adherence

**Completion Phase:**
- ✅ Integration Test Suite: 100% integration scenarios covered
- ✅ E2E Test Coverage: Complete user journey validation
- ✅ Performance Benchmarks: All performance criteria met

### 🚀 PARALLEL TDD EXECUTION PATTERNS

#### Multi-Feature TDD Development:
```bash
# Concurrent TDD for multiple features
[BatchTool - Feature Parallel Development]:
  # Authentication Feature
  - Task("TDD Auth Agent: Implement authentication with London School TDD")
  
  # User Management Feature  
  - Task("TDD User Agent: Implement user CRUD with mock-first approach")
  
  # Token Management Feature
  - Task("TDD Token Agent: Implement JWT handling with interaction testing")
  
  # Integration Feature
  - Task("TDD Integration Agent: Create integration tests for all features")
  
  # Update todos for ALL features
  - TodoWrite { todos: [
      {id: "auth-red", content: "Write failing auth tests", status: "in_progress", priority: "high"},
      {id: "auth-green", content: "Implement minimal auth code", status: "pending", priority: "high"},
      {id: "auth-refactor", content: "Refactor auth implementation", status: "pending", priority: "medium"},
      {id: "user-red", content: "Write failing user tests", status: "pending", priority: "high"},
      {id: "user-green", content: "Implement minimal user code", status: "pending", priority: "high"},
      {id: "token-red", content: "Write failing token tests", status: "pending", priority: "high"},
      {id: "integration-tests", content: "Create integration test suite", status: "pending", priority: "medium"}
    ]}
```

### 🔧 AUTOMATED QUALITY GATES

#### Pre-commit Hooks for TDD Compliance:
```bash
# .husky/pre-commit - Ensure TDD compliance
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run tests BEFORE allowing commit
npm run test:unit
npm run test:integration
npm run lint
npm run coverage:check

# Verify TDD compliance
npm run tdd:verify
```

#### TDD Compliance Checker:
```javascript
// scripts/tdd-verify.js - Verify London School compliance
const fs = require('fs');
const path = require('path');

function verifyTDDCompliance() {
  const testFiles = findTestFiles('./tests');
  const srcFiles = findSourceFiles('./src');
  
  // Verify every source file has corresponding test
  // Verify tests follow London School patterns
  // Verify mock usage compliance
  // Verify interaction testing over state testing
}
```

## 📊 SPARC+TDD PERFORMANCE BENCHMARKS

### 🚀 TDD-Enhanced Performance Improvements
- **Test-First Development**: 250% faster bug detection with upfront testing
- **Mock-Driven Design**: 300% improvement in component isolation and testing speed
- **Parallel Test Execution**: 400% faster test suite execution with concurrent testing
- **Specification-to-Test Traceability**: 200% improvement in requirement validation
- **Refactoring Confidence**: 180% faster code improvements with comprehensive test coverage
- **Integration Validation**: 320% faster system integration with test-driven architecture

### 🎯 SPARC Phase Performance Metrics

#### Specification Phase Performance:
- **Requirement Analysis**: 2.8x faster with parallel requirement processing
- **Test Scenario Generation**: 3.2x faster with automated BDD scenario creation
- **Acceptance Criteria**: 2.5x faster with testable criteria generation

#### TDD Implementation Performance:
- **Red-Green-Refactor Cycles**: 4.1x faster with automated test generation
- **Mock Creation**: 3.8x faster with template-based mock generation
- **Code Coverage Analysis**: 2.9x faster with real-time coverage tracking

#### Quality Gate Performance:
- **Automated Quality Checks**: 5.2x faster with parallel validation
- **Integration Testing**: 3.6x faster with mock-based integration tests
- **Documentation Generation**: 2.7x faster with test-driven documentation

## 🏗️ SPARC+TDD DEVELOPMENT PRINCIPLES

### 🎯 Core SPARC-TDD Integration Principles

#### 1. **Test-Driven Specification** 
- Requirements MUST be written with testable acceptance criteria
- Every specification includes corresponding test scenarios
- BDD (Behavior-Driven Development) integration for natural language tests
- Concurrent specification analysis with automated test generation

#### 2. **Mock-First Architecture**
- London School TDD: Design with mocking in mind
- Dependency injection for all external dependencies
- Interface-driven design for maximum testability
- Parallel mock creation with implementation

#### 3. **Red-Green-Refactor Discipline**
- STRICT adherence to TDD cycles in all phases
- Never write production code without a failing test
- Concurrent test execution across multiple features
- Continuous refactoring with test safety net

#### 4. **Quality Gate Enforcement**
- Automated quality gates between each SPARC phase
- No phase progression without passing all tests
- Parallel quality validation across all aspects
- Real-time feedback on test coverage and quality

#### 5. **Integration-Driven Completion**
- End-to-end testing validates complete SPARC implementation
- Integration tests verify phase-to-phase consistency
- Parallel integration validation with production simulation

### 🛠️ SPARC+TDD Best Practices

#### File Organization (TDD-Optimized):
```
project/
├── specs/                    # SPARC Specification phase
│   ├── requirements/
│   ├── acceptance-criteria/
│   ├── test-scenarios/
│   └── mock-strategy/
├── pseudocode/              # SPARC Pseudocode phase
│   ├── algorithms/
│   ├── test-logic/
│   └── mock-interfaces/
├── architecture/            # SPARC Architecture phase
│   ├── components/
│   ├── interfaces/
│   └── test-strategy/
├── src/                     # SPARC Refinement phase (TDD implementation)
│   ├── components/
│   ├── services/
│   └── utils/
├── tests/                   # Comprehensive test suites
│   ├── unit/               # London School unit tests
│   ├── integration/        # Component integration tests
│   ├── e2e/               # End-to-end system tests
│   └── mocks/             # Reusable mock objects
and docs/                   # SPARC Completion phase
    ├── api/
    ├── deployment/
    └── user-guides/
```

#### TDD Code Quality Standards:
- **Test Coverage**: Minimum 90% line coverage, 85% branch coverage
- **Test Quality**: Every test follows London School principles
- **Mock Usage**: All external dependencies mocked
- **Test Isolation**: Each test runs independently
- **Test Speed**: Unit tests complete in <100ms each
- **Integration Tests**: Integration scenarios complete in <5s each

#### Concurrent Development Patterns:
- **Parallel Feature Development**: Multiple features developed simultaneously with TDD
- **Concurrent Test Execution**: All test suites run in parallel
- **Batch Quality Validation**: All quality checks run together
- **Parallel Documentation**: Test-driven documentation generation

### 🔄 CONTINUOUS TDD FEEDBACK LOOPS

#### Real-Time Test Feedback:
```bash
# Concurrent test monitoring for multiple features
[BatchTool - Continuous TDD]:
  - Bash("npm run test:watch -- --testPathPattern=auth")
  - Bash("npm run test:watch -- --testPathPattern=user")
  - Bash("npm run test:watch -- --testPathPattern=token")
  - Bash("npm run test:watch -- --testPathPattern=integration")
  - Bash("npm run coverage:watch")
```

#### Automated TDD Compliance:
- **Pre-commit hooks** verify TDD compliance
- **CI/CD integration** enforces test-first development
- **Real-time metrics** track TDD cycle adherence
- **Quality dashboards** monitor test coverage and quality trends

## 🚨 CRITICAL SPARC+TDD REQUIREMENTS

### 🔴 MANDATORY TDD COMPLIANCE RULES

1. **NEVER write production code without a failing test first** 
   - Every line of code MUST be driven by a failing test
   - Use `npm run test:watch` for real-time feedback
   - Verify tests fail before implementing

2. **STRICT Red-Green-Refactor cycle adherence**
   - 🔴 **RED**: Write the simplest failing test
   - 🟢 **GREEN**: Write the minimal code to pass
   - 🔵 **REFACTOR**: Improve code while keeping tests green
   - 🔄 **REPEAT**: Continue until feature complete

3. **London School TDD principles**
   - Mock ALL external dependencies
   - Test interactions, not just state
   - Start with interfaces, work inward
   - Isolate units completely

4. **Quality gate enforcement**
   - No SPARC phase progression without passing quality gates
   - Automated quality checks before commits
   - Real-time test coverage monitoring
   - Continuous integration validation

### 🎯 SPARC PHASE TRANSITION RULES

#### Phase Progression Requirements:
```bash
# Quality gate validation before phase transitions
[BatchTool - Phase Validation]:
  - Bash("npm run test:unit")              # All unit tests pass
  - Bash("npm run test:integration")       # Integration tests pass  
  - Bash("npm run coverage:check")         # Coverage thresholds met
  - Bash("npm run lint")                   # Code quality standards
  - Bash("npm run tdd:compliance")         # TDD compliance verified
```

### 📊 MONITORING AND METRICS

#### Real-Time TDD Dashboards:
- **Test Coverage**: Live coverage metrics with trend analysis
- **Test Quality**: Mock usage, isolation, and speed metrics
- **TDD Compliance**: Red-Green-Refactor cycle adherence
- **Phase Progress**: SPARC phase completion with quality gates
- **Performance**: Test execution speed and system performance

#### Automated Alerts:
- Coverage drops below 90%
- Tests take longer than thresholds
- TDD cycle violations detected
- Quality gate failures
- Integration test failures

### 🔧 ESSENTIAL TDD COMMANDS

#### Daily TDD Workflow:
```bash
# Morning setup - Start TDD session
[BatchTool - TDD Session Start]:
  - Bash("npm run test:watch")             # Start test watcher
  - Bash("npm run coverage:watch")         # Monitor coverage
  - mcp__claude-flow__memory_usage { action: "retrieve", key: "tdd-session" }
  - TodoWrite { todos: [daily TDD todos] }

# Feature development - TDD cycle
[BatchTool - TDD Feature Cycle]:
  - Write("tests/feature.test.js", failingTest)    # RED
  - Bash("npm test -- --testNamePattern='feature'")
  - Write("src/feature.js", minimalImplementation) # GREEN
  - Bash("npm test -- --testNamePattern='feature'")
  - Edit("src/feature.js", refactoredCode)         # REFACTOR
  - Bash("npm test -- --testNamePattern='feature'")

# End of session - Validate and store
[BatchTool - TDD Session End]:
  - Bash("npm run test:all")               # Full test suite
  - Bash("npm run coverage:report")        # Generate coverage report
  - mcp__claude-flow__memory_usage { action: "store", key: "tdd-session-end" }
```

### 🏆 SUCCESS CRITERIA

#### Project Completion Checklist:
- ✅ All SPARC phases completed with quality gates passed
- ✅ Test coverage > 90% (lines), > 85% (branches)
- ✅ All tests follow London School TDD principles
- ✅ Zero production code written without failing tests first
- ✅ Complete integration test suite covering all scenarios
- ✅ End-to-end tests validate complete user journeys
- ✅ Performance benchmarks met or exceeded
- ✅ Documentation generated from tests and specifications
- ✅ Deployment pipeline includes comprehensive test validation

## Available Agents (54 Total)

### 🚀 Concurrent Agent Usage

**CRITICAL**: Always spawn multiple agents concurrently using the Task tool in a single message:

```javascript
// ✅ CORRECT: Concurrent agent deployment
[Single Message]:
  - Task("Agent 1", "full instructions", "agent-type-1")
  - Task("Agent 2", "full instructions", "agent-type-2") 
  - Task("Agent 3", "full instructions", "agent-type-3")
  - Task("Agent 4", "full instructions", "agent-type-4")
  - Task("Agent 5", "full instructions", "agent-type-5")
```

### 📋 Agent Categories & Concurrent Patterns

#### **Core Development Agents**
- `coder` - Implementation specialist
- `reviewer` - Code quality assurance
- `tester` - Test creation and validation
- `planner` - Strategic planning
- `researcher` - Information gathering

**Concurrent Usage:**
```bash
# Deploy full development swarm
Task("Research requirements", "...", "researcher")
Task("Plan architecture", "...", "planner") 
Task("Implement features", "...", "coder")
Task("Create tests", "...", "tester")
Task("Review code", "...", "reviewer")
```

#### **Swarm Coordination Agents**
- `hierarchical-coordinator` - Queen-led coordination
- `mesh-coordinator` - Peer-to-peer networks
- `adaptive-coordinator` - Dynamic topology
- `collective-intelligence-coordinator` - Hive-mind intelligence
- `swarm-memory-manager` - Distributed memory

**Concurrent Swarm Deployment:**
```bash
# Deploy multi-topology coordination
Task("Hierarchical coordination", "...", "hierarchical-coordinator")
Task("Mesh network backup", "...", "mesh-coordinator")
Task("Adaptive optimization", "...", "adaptive-coordinator")
```

#### **Consensus & Distributed Systems**
- `byzantine-coordinator` - Byzantine fault tolerance
- `raft-manager` - Leader election protocols
- `gossip-coordinator` - Epidemic dissemination
- `consensus-builder` - Decision-making algorithms
- `crdt-synchronizer` - Conflict-free replication
- `quorum-manager` - Dynamic quorum management
- `security-manager` - Cryptographic security

#### **Performance & Optimization**
- `perf-analyzer` - Bottleneck identification
- `performance-benchmarker` - Performance testing
- `task-orchestrator` - Workflow optimization
- `memory-coordinator` - Memory management
- `smart-agent` - Intelligent coordination

#### **GitHub & Repository Management**
- `github-modes` - Comprehensive GitHub integration
- `pr-manager` - Pull request management
- `code-review-swarm` - Multi-agent code review
- `issue-tracker` - Issue management
- `release-manager` - Release coordination
- `workflow-automation` - CI/CD automation
- `project-board-sync` - Project tracking
- `repo-architect` - Repository optimization
- `multi-repo-swarm` - Cross-repository coordination

#### **SPARC+TDD Methodology Agents**
- `sparc-coord` - SPARC orchestration with TDD integration
- `sparc-coder` - London School TDD implementation
- `specification` - Requirements analysis with test scenarios
- `pseudocode` - Algorithm design with test logic
- `architecture` - System design with testable components
- `refinement` - TDD Red-Green-Refactor implementation
- `tdd-london-agent` - London School TDD specialist
- `test-designer` - Test scenario and mock strategy designer
- `quality-gate-validator` - SPARC phase quality gate enforcement
- `integration-tester` - Integration and E2E test specialist

#### **Specialized Development**
- `backend-dev` - API development
- `mobile-dev` - React Native development
- `ml-developer` - Machine learning
- `cicd-engineer` - CI/CD pipelines
- `api-docs` - OpenAPI documentation
- `system-architect` - High-level design
- `code-analyzer` - Code quality analysis
- `base-template-generator` - Boilerplate creation

#### **Testing & Validation**
- `tdd-london-swarm` - Mock-driven TDD
- `production-validator` - Real implementation validation

#### **Migration & Planning**
- `migration-planner` - System migrations
- `swarm-init` - Topology initialization

### 🎯 Concurrent Agent Patterns

#### **Full-Stack Development Swarm (8 agents)**
```bash
Task("System architecture", "...", "system-architect")
Task("Backend APIs", "...", "backend-dev") 
Task("Frontend mobile", "...", "mobile-dev")
Task("Database design", "...", "coder")
Task("API documentation", "...", "api-docs")
Task("CI/CD pipeline", "...", "cicd-engineer")
Task("Performance testing", "...", "performance-benchmarker")
Task("Production validation", "...", "production-validator")
```

#### **Distributed System Swarm (6 agents)**
```bash
Task("Byzantine consensus", "...", "byzantine-coordinator")
Task("Raft coordination", "...", "raft-manager")
Task("Gossip protocols", "...", "gossip-coordinator") 
Task("CRDT synchronization", "...", "crdt-synchronizer")
Task("Security management", "...", "security-manager")
Task("Performance monitoring", "...", "perf-analyzer")
```

#### **GitHub Workflow Swarm (5 agents)**
```bash
Task("PR management", "...", "pr-manager")
Task("Code review", "...", "code-review-swarm")
Task("Issue tracking", "...", "issue-tracker")
Task("Release coordination", "...", "release-manager")
Task("Workflow automation", "...", "workflow-automation")
```

#### **SPARC+TDD Complete Swarm (10 agents)**
```bash
# Comprehensive SPARC+TDD agent deployment
Task("Requirements with test scenarios", "MANDATORY: Use hooks. Create specs with testable acceptance criteria", "specification")
Task("Algorithm design with test logic", "MANDATORY: Use hooks. Design algorithms with embedded test validation", "pseudocode")
Task("Testable system architecture", "MANDATORY: Use hooks. Design components with mock-friendly interfaces", "architecture") 
Task("London School TDD implementation", "MANDATORY: Use hooks. Implement using strict Red-Green-Refactor cycles", "tdd-london-agent")
Task("Mock strategy and test design", "MANDATORY: Use hooks. Create comprehensive mock strategies", "test-designer")
Task("Quality gate validation", "MANDATORY: Use hooks. Enforce quality gates between phases", "quality-gate-validator")
Task("Unit test implementation", "MANDATORY: Use hooks. Create isolated unit tests with mocks", "tester")
Task("Integration test suite", "MANDATORY: Use hooks. Build comprehensive integration tests", "integration-tester")
Task("Code review and refinement", "MANDATORY: Use hooks. Review code quality and test coverage", "reviewer")
Task("Production validation", "MANDATORY: Use hooks. Validate production readiness", "production-validator")
```

### ⚡ Performance Optimization

**Agent Selection Strategy:**
- **High Priority**: Use 3-5 agents max for critical path
- **Medium Priority**: Use 5-8 agents for complex features
- **Large Projects**: Use 8+ agents with proper coordination

**Memory Management:**
- Use `memory-coordinator` for cross-agent state
- Implement `swarm-memory-manager` for distributed coordination
- Apply `collective-intelligence-coordinator` for decision-making

## 🎯 SPARC+TDD COMMAND REFERENCE

### 🔧 Essential SPARC+TDD Commands

#### Phase-Specific Command Patterns:

**Specification Phase Commands:**
```bash
# Concurrent specification with test scenario generation
[BatchTool - Specification Phase]
  - npx claude-flow sparc run specification "user authentication" --test-scenarios --parallel
  - npx claude-flow sparc generate acceptance-criteria "authentication requirements" --testable
  - npx claude-flow sparc create mock-strategy "external dependencies" --london-school
  - Write("specs/requirements.md", detailedRequirements)
  - Write("specs/test-scenarios.bdd", bddScenarios)
  - Write("specs/acceptance-criteria.md", testableCriteria)
  - Bash("mkdir -p specs/{requirements,scenarios,criteria,mocks}")
```

**Pseudocode Phase Commands:**
```bash
# Algorithm design with parallel test logic creation
[BatchTool - Pseudocode Phase]
  - npx claude-flow sparc run pseudocode "authentication flow" --test-logic --parallel
  - npx claude-flow sparc design algorithm "login process" --mock-points
  - npx claude-flow sparc validate logic "authentication algorithm" --test-driven
  - Write("pseudocode/auth-algorithm.md", algorithmDesign)
  - Write("pseudocode/test-logic.md", testLogicDesign)
  - Write("pseudocode/mock-interfaces.md", mockInterfaceSpecs)
  - Bash("mkdir -p pseudocode/{algorithms,test-logic,mock-interfaces}")
```

**Architecture Phase Commands:**
```bash
# System design with testable component architecture
[BatchTool - Architecture Phase]
  - npx claude-flow sparc run architecture "system design" --testable-components --parallel
  - npx claude-flow sparc design interfaces "component contracts" --mock-friendly
  - npx claude-flow sparc plan integration "system integration" --test-points
  - Write("architecture/system-design.md", architecturalDesign)
  - Write("architecture/component-interfaces.md", interfaceContracts)
  - Write("architecture/test-strategy.md", testingStrategy)
  - Bash("mkdir -p architecture/{design,interfaces,integration,testing}")
```

**Refinement Phase (TDD Implementation) Commands:**
```bash
# Pure TDD implementation with London School approach
[BatchTool - TDD Refinement Phase]
  - npx claude-flow sparc tdd "implement authentication" --london-school --parallel
  - npx claude-flow tdd cycle "auth service" --red-green-refactor
  - npx claude-flow tdd mock-first "user repository" --interactions
  - Write("tests/unit/auth.test.js", redPhaseTests)
  - Write("src/auth/AuthService.js", greenPhaseImplementation)
  - Write("tests/mocks/UserRepository.mock.js", mockImplementations)
  - Bash("npm run test:watch -- --testPathPattern=auth")
  - Bash("npm run coverage:watch")
```

**Completion Phase Commands:**
```bash
# Integration testing and system validation
[BatchTool - Completion Phase]
  - npx claude-flow sparc run completion "system integration" --e2e-tests --parallel
  - npx claude-flow sparc validate system "full integration" --performance
  - npx claude-flow sparc generate docs "api documentation" --test-driven
  - Write("tests/integration/auth-integration.test.js", integrationTests)
  - Write("tests/e2e/user-journey.test.js", e2eTests)
  - Write("docs/api/authentication.md", apiDocumentation)
  - Bash("npm run test:integration")
  - Bash("npm run test:e2e")
  - Bash("npm run build:production")
```

### 🎭 QUALITY GATE SPECIFICATIONS

#### Quality Gate 1: Specification Complete
```javascript
// Automated validation criteria
const specificationQualityGate = {
  requirements: {
    documented: true,
    testable: true,
    coverage: 100
  },
  acceptanceCriteria: {
    defined: true,
    automatable: true,
    coverage: 100
  },
  testScenarios: {
    created: true,
    bddFormat: true,
    edgeCases: true
  },
  mockStrategy: {
    planned: true,
    londonSchool: true,
    dependencies: 'all_identified'
  }
};
```

#### Quality Gate 2: Pseudocode + Test Logic Complete
```javascript
// Algorithm and test logic validation
const pseudocodeQualityGate = {
  algorithms: {
    designed: true,
    testable: true,
    optimized: true
  },
  testLogic: {
    created: true,
    comprehensive: true,
    mockPoints: 'identified'
  },
  mockInterfaces: {
    designed: true,
    contractsDefined: true,
    testable: true
  },
  validation: {
    algorithmTested: true,
    edgeCasesCovered: true,
    performanceConsidered: true
  }
};
```

#### Quality Gate 3: Architecture + Test Strategy Complete
```javascript
// System architecture and testing strategy validation
const architectureQualityGate = {
  systemDesign: {
    components: 'defined',
    interfaces: 'specified',
    testable: true
  },
  testStrategy: {
    unitTesting: 'planned',
    integrationTesting: 'planned',
    e2eTesting: 'planned'
  },
  dependencyInjection: {
    configured: true,
    mockable: true,
    londonSchoolCompliant: true
  },
  integrationPoints: {
    identified: true,
    testable: true,
    documented: true
  }
};
```

#### Quality Gate 4: TDD Implementation Complete
```javascript
// TDD implementation validation
const tddQualityGate = {
  testCoverage: {
    lines: '>= 90%',
    branches: '>= 85%',
    functions: '>= 90%'
  },
  testQuality: {
    londonSchool: true,
    isolated: true,
    fast: true,
    mockUsage: 'appropriate'
  },
  tddCompliance: {
    redGreenRefactor: true,
    testFirst: true,
    noProdCodeWithoutTest: true
  },
  codeQuality: {
    linted: true,
    formatted: true,
    refactored: true
  }
};
```

#### Quality Gate 5: System Integration Complete
```javascript
// Complete system validation
const integrationQualityGate = {
  integrationTests: {
    allScenarios: 'covered',
    passing: true,
    performance: 'acceptable'
  },
  e2eTests: {
    userJourneys: 'complete',
    crossBrowser: true,
    responsive: true
  },
  productionReadiness: {
    deployed: true,
    monitored: true,
    documented: true
  },
  qualityMetrics: {
    allQualityGatesPassed: true,
    performanceBenchmarks: 'met',
    securityValidated: true
  }
};
```

### 🔄 AUTOMATED SPARC+TDD WORKFLOW

```bash
# Complete SPARC+TDD automation
#!/bin/bash

# Phase 1: Specification with TDD Planning
echo "🎯 Starting SPARC Specification Phase with TDD Integration..."
npx claude-flow sparc run specification "$FEATURE" --test-scenarios --parallel
npm run validate:specification

# Phase 2: Pseudocode with Test Logic
echo "🧠 Starting SPARC Pseudocode Phase with Test Logic..."
npx claude-flow sparc run pseudocode "$FEATURE" --test-logic --mock-design --parallel
npm run validate:pseudocode

# Phase 3: Architecture with Test Strategy
echo "🏗️ Starting SPARC Architecture Phase with Test Strategy..."
npx claude-flow sparc run architecture "$FEATURE" --testable-components --parallel
npm run validate:architecture

# Phase 4: TDD Implementation (Red-Green-Refactor)
echo "🔄 Starting SPARC Refinement Phase with London School TDD..."
npx claude-flow sparc tdd "$FEATURE" --london-school --parallel
npm run validate:tdd-implementation

# Phase 5: Integration and Completion
echo "✅ Starting SPARC Completion Phase with Integration Testing..."
npx claude-flow sparc run completion "$FEATURE" --integration-tests --parallel
npm run validate:completion

echo "🎉 SPARC+TDD cycle complete for $FEATURE!"
```

For more information about SPARC+TDD methodology and implementation patterns, see: 
- SPARC+TDD Guide: https://github.com/ruvnet/claude-code-flow/docs/sparc-tdd.md
- London School TDD: https://github.com/ruvnet/claude-code-flow/docs/london-tdd.md
- Quality Gates: https://github.com/ruvnet/claude-code-flow/docs/quality-gates.md

# important-instruction-reminders
Message 3: Task("Agent 2")
Message 4: Read("file1.js")
Message 5: Write("output1.js")
Message 6: Bash("npm install")
// This is 6x slower and breaks coordination!
```

### 🎯 CONCURRENT EXECUTION CHECKLIST:

Before sending ANY message, ask yourself:

- ✅ Are ALL related TodoWrite operations batched together?
- ✅ Are ALL Task spawning operations in ONE message?
- ✅ Are ALL file operations (Read/Write/Edit) batched together?
- ✅ Are ALL bash commands grouped in ONE message?
- ✅ Are ALL memory operations concurrent?

If ANY answer is "No", you MUST combine operations into a single message!

## 🚀 CRITICAL: Claude Code Does ALL Real Work

### 🎯 CLAUDE CODE IS THE ONLY EXECUTOR

**ABSOLUTE RULE**: Claude Code performs ALL actual work:

### ✅ Claude Code ALWAYS Handles:

- 🔧 **ALL file operations** (Read, Write, Edit, MultiEdit, Glob, Grep)
- 💻 **ALL code generation** and programming tasks
- 🖥️ **ALL bash commands** and system operations
- 🏗️ **ALL actual implementation** work
- 🔍 **ALL project navigation** and code analysis
- 📝 **ALL TodoWrite** and task management
- 🔄 **ALL git operations** (commit, push, merge)
- 📦 **ALL package management** (npm, pip, etc.)
- 🧪 **ALL testing** and validation
- 🔧 **ALL debugging** and troubleshooting

### 🧠 Claude Flow MCP Tools ONLY Handle:

- 🎯 **Coordination only** - Planning Claude Code's actions
- 💾 **Memory management** - Storing decisions and context
- 🤖 **Neural features** - Learning from Claude Code's work
- 📊 **Performance tracking** - Monitoring Claude Code's efficiency
- 🐝 **Swarm orchestration** - Coordinating multiple Claude Code instances
- 🔗 **GitHub integration** - Advanced repository coordination

### 🚨 CRITICAL SEPARATION OF CONCERNS:

**❌ MCP Tools NEVER:**

- Write files or create content
- Execute bash commands
- Generate code
- Perform file operations
- Handle TodoWrite operations
- Execute system commands
- Do actual implementation work

**✅ MCP Tools ONLY:**

- Coordinate and plan
- Store memory and context
- Track performance
- Orchestrate workflows
- Provide intelligence insights

### ⚠️ Key Principle:

**MCP tools coordinate, Claude Code executes.** Think of MCP tools as the "brain" that plans and coordinates, while Claude Code is the "hands" that do all the actual work.

### 🔄 WORKFLOW EXECUTION PATTERN:

**✅ CORRECT Workflow:**

1. **MCP**: `mcp__claude-flow__swarm_init` (coordination setup)
2. **MCP**: `mcp__claude-flow__agent_spawn` (planning agents)
3. **MCP**: `mcp__claude-flow__task_orchestrate` (task coordination)
4. **Claude Code**: `Task` tool to spawn agents with coordination instructions
5. **Claude Code**: `TodoWrite` with ALL todos batched (5-10+ in ONE call)
6. **Claude Code**: `Read`, `Write`, `Edit`, `Bash` (actual work)
7. **MCP**: `mcp__claude-flow__memory_usage` (store results)

**❌ WRONG Workflow:**

1. **MCP**: `mcp__claude-flow__terminal_execute` (DON'T DO THIS)
2. **MCP**: File creation via MCP (DON'T DO THIS)
3. **MCP**: Code generation via MCP (DON'T DO THIS)
4. **Claude Code**: Sequential Task calls (DON'T DO THIS)
5. **Claude Code**: Individual TodoWrite calls (DON'T DO THIS)

### 🚨 REMEMBER:

- **MCP tools** = Coordination, planning, memory, intelligence
- **Claude Code** = All actual execution, coding, file operations

## 🚀 CRITICAL: Parallel Execution & Batch Operations

### 🚨 MANDATORY RULE #1: BATCH EVERYTHING

**When using swarms, you MUST use BatchTool for ALL operations:**

1. **NEVER** send multiple messages for related operations
2. **ALWAYS** combine multiple tool calls in ONE message
3. **PARALLEL** execution is MANDATORY, not optional

### ⚡ THE GOLDEN RULE OF SWARMS

```
If you need to do X operations, they should be in 1 message, not X messages
```

### 🚨 MANDATORY TODO AND TASK BATCHING

**CRITICAL RULE FOR TODOS AND TASKS:**

1. **TodoWrite** MUST ALWAYS include ALL todos in ONE call (5-10+ todos)
2. **Task** tool calls MUST be batched - spawn multiple agents in ONE message
3. **NEVER** update todos one by one - this breaks parallel coordination
4. **NEVER** spawn agents sequentially - ALL agents spawn together

### 📦 BATCH TOOL EXAMPLES

**✅ CORRECT - Everything in ONE Message:**

```javascript
[Single Message with BatchTool]:
  // MCP coordination setup
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "analyst" }
  mcp__claude-flow__agent_spawn { type: "tester" }
  mcp__claude-flow__agent_spawn { type: "coordinator" }

  // Claude Code execution - ALL in parallel
  Task("You are researcher agent. MUST coordinate via hooks...")
  Task("You are coder agent. MUST coordinate via hooks...")
  Task("You are analyst agent. MUST coordinate via hooks...")
  Task("You are tester agent. MUST coordinate via hooks...")
  TodoWrite { todos: [5-10 todos with all priorities and statuses] }

  // File operations in parallel
  Bash "mkdir -p app/{src,tests,docs}"
  Write "app/package.json"
  Write "app/README.md"
  Write "app/src/index.js"
```

**❌ WRONG - Multiple Messages (NEVER DO THIS):**

```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("researcher agent")
Message 3: Task("coder agent")
Message 4: TodoWrite({ todo: "single todo" })
Message 5: Bash "mkdir src"
Message 6: Write "package.json"
// This is 6x slower and breaks parallel coordination!
```

### 🎯 BATCH OPERATIONS BY TYPE

**Todo and Task Operations (Single Message):**

- **TodoWrite** → ALWAYS include 5-10+ todos in ONE call
- **Task agents** → Spawn ALL agents with full instructions in ONE message
- **Agent coordination** → ALL Task calls must include coordination hooks
- **Status updates** → Update ALL todo statuses together
- **NEVER** split todos or Task calls across messages!

**File Operations (Single Message):**

- Read 10 files? → One message with 10 Read calls
- Write 5 files? → One message with 5 Write calls
- Edit 1 file many times? → One MultiEdit call

**Swarm Operations (Single Message):**

- Need 8 agents? → One message with swarm_init + 8 agent_spawn calls
- Multiple memories? → One message with all memory_usage calls
- Task + monitoring? → One message with task_orchestrate + swarm_monitor

**Command Operations (Single Message):**

- Multiple directories? → One message with all mkdir commands
- Install + test + lint? → One message with all npm commands
- Git operations? → One message with all git commands

## 🚀 Quick Setup (Stdio MCP - Recommended)

### 1. Add MCP Server (Stdio - No Port Needed)

```bash
# Add Claude Flow MCP server to Claude Code using stdio
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

### 2. Use MCP Tools for Coordination in Claude Code

Once configured, Claude Flow MCP tools enhance Claude Code's coordination:

**Initialize a swarm:**

- Use the `mcp__claude-flow__swarm_init` tool to set up coordination topology
- Choose: mesh, hierarchical, ring, or star
- This creates a coordination framework for Claude Code's work

**Spawn agents:**

- Use `mcp__claude-flow__agent_spawn` tool to create specialized coordinators
- Agent types represent different thinking patterns, not actual coders
- They help Claude Code approach problems from different angles

**Orchestrate tasks:**

- Use `mcp__claude-flow__task_orchestrate` tool to coordinate complex workflows
- This breaks down tasks for Claude Code to execute systematically
- The agents don't write code - they coordinate Claude Code's actions

## Available MCP Tools for Coordination

### Coordination Tools:

- `mcp__claude-flow__swarm_init` - Set up coordination topology for Claude Code
- `mcp__claude-flow__agent_spawn` - Create cognitive patterns to guide Claude Code
- `mcp__claude-flow__task_orchestrate` - Break down and coordinate complex tasks

### Monitoring Tools:

- `mcp__claude-flow__swarm_status` - Monitor coordination effectiveness
- `mcp__claude-flow__agent_list` - View active cognitive patterns
- `mcp__claude-flow__agent_metrics` - Track coordination performance
- `mcp__claude-flow__task_status` - Check workflow progress
- `mcp__claude-flow__task_results` - Review coordination outcomes

### Memory & Neural Tools:

- `mcp__claude-flow__memory_usage` - Persistent memory across sessions
- `mcp__claude-flow__neural_status` - Neural pattern effectiveness
- `mcp__claude-flow__neural_train` - Improve coordination patterns
- `mcp__claude-flow__neural_patterns` - Analyze thinking approaches

### GitHub Integration Tools (NEW!):

- `mcp__claude-flow__github_swarm` - Create specialized GitHub management swarms
- `mcp__claude-flow__repo_analyze` - Deep repository analysis with AI
- `mcp__claude-flow__pr_enhance` - AI-powered pull request improvements
- `mcp__claude-flow__issue_triage` - Intelligent issue classification
- `mcp__claude-flow__code_review` - Automated code review with swarms

### System Tools:

- `mcp__claude-flow__benchmark_run` - Measure coordination efficiency
- `mcp__claude-flow__features_detect` - Available capabilities
- `mcp__claude-flow__swarm_monitor` - Real-time coordination tracking

## Workflow Examples (Coordination-Focused)

### Research Coordination Example

**Context:** Claude Code needs to research a complex topic systematically

**Step 1:** Set up research coordination

- Tool: `mcp__claude-flow__swarm_init`
- Parameters: `{"topology": "mesh", "maxAgents": 5, "strategy": "balanced"}`
- Result: Creates a mesh topology for comprehensive exploration

**Step 2:** Define research perspectives

- Tool: `mcp__claude-flow__agent_spawn`
- Parameters: `{"type": "researcher", "name": "Literature Review"}`
- Tool: `mcp__claude-flow__agent_spawn`
- Parameters: `{"type": "analyst", "name": "Data Analysis"}`
- Result: Different cognitive patterns for Claude Code to use

**Step 3:** Coordinate research execution

- Tool: `mcp__claude-flow__task_orchestrate`
- Parameters: `{"task": "Research neural architecture search papers", "strategy": "adaptive"}`
- Result: Claude Code systematically searches, reads, and analyzes papers

**What Actually Happens:**

1. The swarm sets up a coordination framework
2. Each agent MUST use Claude Flow hooks for coordination:
   - `npx claude-flow@alpha hooks pre-task` before starting
   - `npx claude-flow@alpha hooks post-edit` after each file operation
   - `npx claude-flow@alpha hooks notify` to share decisions
3. Claude Code uses its native Read, WebSearch, and Task tools
4. The swarm coordinates through shared memory and hooks
5. Results are synthesized by Claude Code with full coordination history

### Development Coordination Example

**Context:** Claude Code needs to build a complex system with multiple components

**Step 1:** Set up development coordination

- Tool: `mcp__claude-flow__swarm_init`
- Parameters: `{"topology": "hierarchical", "maxAgents": 8, "strategy": "specialized"}`
- Result: Hierarchical structure for organized development

**Step 2:** Define development perspectives

- Tool: `mcp__claude-flow__agent_spawn`
- Parameters: `{"type": "architect", "name": "System Design"}`
- Result: Architectural thinking pattern for Claude Code

**Step 3:** Coordinate implementation

- Tool: `mcp__claude-flow__task_orchestrate`
- Parameters: `{"task": "Implement user authentication with JWT", "strategy": "parallel"}`
- Result: Claude Code implements features using its native tools

**What Actually Happens:**

1. The swarm creates a development coordination plan
2. Each agent coordinates using mandatory hooks:
   - Pre-task hooks for context loading
   - Post-edit hooks for progress tracking
   - Memory storage for cross-agent coordination
3. Claude Code uses Write, Edit, Bash tools for implementation
4. Agents share progress through Claude Flow memory
5. All code is written by Claude Code with full coordination

### GitHub Repository Management Example (NEW!)

**Context:** Claude Code needs to manage a complex GitHub repository

**Step 1:** Initialize GitHub swarm

- Tool: `mcp__claude-flow__github_swarm`
- Parameters: `{"repository": "owner/repo", "agents": 5, "focus": "maintenance"}`
- Result: Specialized swarm for repository management

**Step 2:** Analyze repository health

- Tool: `mcp__claude-flow__repo_analyze`
- Parameters: `{"deep": true, "include": ["issues", "prs", "code"]}`
- Result: Comprehensive repository analysis

**Step 3:** Enhance pull requests

- Tool: `mcp__claude-flow__pr_enhance`
- Parameters: `{"pr_number": 123, "add_tests": true, "improve_docs": true}`
- Result: AI-powered PR improvements

## Best Practices for Coordination

### ✅ DO:

- Use MCP tools to coordinate Claude Code's approach to complex tasks
- Let the swarm break down problems into manageable pieces
- Use memory tools to maintain context across sessions
- Monitor coordination effectiveness with status tools
- Train neural patterns for better coordination over time
- Leverage GitHub tools for repository management

### ❌ DON'T:

- Expect agents to write code (Claude Code does all implementation)
- Use MCP tools for file operations (use Claude Code's native tools)
- Try to make agents execute bash commands (Claude Code handles this)
- Confuse coordination with execution (MCP coordinates, Claude executes)

## Memory and Persistence

The swarm provides persistent memory that helps Claude Code:

- Remember project context across sessions
- Track decisions and rationale
- Maintain consistency in large projects
- Learn from previous coordination patterns
- Store GitHub workflow preferences

## Performance Benefits

When using Claude Flow coordination with Claude Code:

- **84.8% SWE-Bench solve rate** - Better problem-solving through coordination
- **32.3% token reduction** - Efficient task breakdown reduces redundancy
- **2.8-4.4x speed improvement** - Parallel coordination strategies
- **27+ neural models** - Diverse cognitive approaches
- **GitHub automation** - Streamlined repository management

## Claude Code Hooks Integration

Claude Flow includes powerful hooks that automate coordination:

### Pre-Operation Hooks

- **Auto-assign agents** before file edits based on file type
- **Validate commands** before execution for safety
- **Prepare resources** automatically for complex operations
- **Optimize topology** based on task complexity analysis
- **Cache searches** for improved performance
- **GitHub context** loading for repository operations

### Post-Operation Hooks

- **Auto-format code** using language-specific formatters
- **Train neural patterns** from successful operations
- **Update memory** with operation context
- **Analyze performance** and identify bottlenecks
- **Track token usage** for efficiency metrics
- **Sync GitHub** state for consistency

### Session Management

- **Generate summaries** at session end
- **Persist state** across Claude Code sessions
- **Track metrics** for continuous improvement
- **Restore previous** session context automatically
- **Export workflows** for reuse

### Advanced Features (v2.0.0!)

- **🚀 Automatic Topology Selection** - Optimal swarm structure for each task
- **⚡ Parallel Execution** - 2.8-4.4x speed improvements
- **🧠 Neural Training** - Continuous learning from operations
- **📊 Bottleneck Analysis** - Real-time performance optimization
- **🤖 Smart Auto-Spawning** - Zero manual agent management
- **🛡️ Self-Healing Workflows** - Automatic error recovery
- **💾 Cross-Session Memory** - Persistent learning & context
- **🔗 GitHub Integration** - Repository-aware swarms

### Configuration

Hooks are pre-configured in `.claude/settings.json`. Key features:

- Automatic agent assignment for different file types
- Code formatting on save
- Neural pattern learning from edits
- Session state persistence
- Performance tracking and optimization
- Intelligent caching and token reduction
- GitHub workflow automation

See `.claude/commands/` for detailed documentation on all features.

## Integration Tips

1. **Start Simple**: Begin with basic swarm init and single agent
2. **Scale Gradually**: Add more agents as task complexity increases
3. **Use Memory**: Store important decisions and context
4. **Monitor Progress**: Regular status checks ensure effective coordination
5. **Train Patterns**: Let neural agents learn from successful coordinations
6. **Enable Hooks**: Use the pre-configured hooks for automation
7. **GitHub First**: Use GitHub tools for repository management

## 🧠 SWARM ORCHESTRATION PATTERN

### You are the SWARM ORCHESTRATOR. **IMMEDIATELY SPAWN AGENTS IN PARALLEL** to execute tasks

### 🚨 CRITICAL INSTRUCTION: You are the SWARM ORCHESTRATOR

**MANDATORY**: When using swarms, you MUST:

1. **SPAWN ALL AGENTS IN ONE BATCH** - Use multiple tool calls in a SINGLE message
2. **EXECUTE TASKS IN PARALLEL** - Never wait for one task before starting another
3. **USE BATCHTOOL FOR EVERYTHING** - Multiple operations = Single message with multiple tools
4. **ALL AGENTS MUST USE COORDINATION TOOLS** - Every spawned agent MUST use claude-flow hooks and memory

### 🎯 AGENT COUNT CONFIGURATION

**CRITICAL: Dynamic Agent Count Rules**

1. **Check CLI Arguments First**: If user runs `npx claude-flow@alpha --agents 5`, use 5 agents
2. **Auto-Decide if No Args**: Without CLI args, analyze task complexity:
   - Simple tasks (1-3 components): 3-4 agents
   - Medium tasks (4-6 components): 5-7 agents
   - Complex tasks (7+ components): 8-12 agents
3. **Agent Type Distribution**: Balance agent types based on task:
   - Always include 1 coordinator
   - For code-heavy tasks: more coders
   - For design tasks: more architects/analysts
   - For quality tasks: more testers/reviewers

**Example Auto-Decision Logic:**

```javascript
// If CLI args provided: npx claude-flow@alpha --agents 6
maxAgents = CLI_ARGS.agents || determineAgentCount(task);

function determineAgentCount(task) {
  // Analyze task complexity
  if (task.includes(['API', 'database', 'auth', 'tests'])) return 8;
  if (task.includes(['frontend', 'backend'])) return 6;
  if (task.includes(['simple', 'script'])) return 3;
  return 5; // default
}
```

## 📋 MANDATORY AGENT COORDINATION PROTOCOL

### 🔴 CRITICAL: Every Agent MUST Follow This Protocol

When you spawn an agent using the Task tool, that agent MUST:

**1️⃣ BEFORE Starting Work:**

```bash
# Check previous work and load context
npx claude-flow@alpha hooks pre-task --description "[agent task]" --auto-spawn-agents false
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]" --load-memory true
```

**2️⃣ DURING Work (After EVERY Major Step):**

```bash
# Store progress in memory after each file operation
npx claude-flow@alpha hooks post-edit --file "[filepath]" --memory-key "swarm/[agent]/[step]"

# Store decisions and findings
npx claude-flow@alpha hooks notify --message "[what was done]" --telemetry true

# Check coordination with other agents
npx claude-flow@alpha hooks pre-search --query "[what to check]" --cache-results true
```

**3️⃣ AFTER Completing Work:**

```bash
# Save all results and learnings
npx claude-flow@alpha hooks post-task --task-id "[task]" --analyze-performance true
npx claude-flow@alpha hooks session-end --export-metrics true --generate-summary true
```

### 🎯 AGENT PROMPT TEMPLATE

When spawning agents, ALWAYS include these coordination instructions:

```
You are the [Agent Type] agent in a coordinated swarm.

MANDATORY COORDINATION:
1. START: Run `npx claude-flow@alpha hooks pre-task --description "[your task]"`
2. DURING: After EVERY file operation, run `npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "agent/[step]"`
3. MEMORY: Store ALL decisions using `npx claude-flow@alpha hooks notify --message "[decision]"`
4. END: Run `npx claude-flow@alpha hooks post-task --task-id "[task]" --analyze-performance true`

Your specific task: [detailed task description]

REMEMBER: Coordinate with other agents by checking memory BEFORE making decisions!
```

### ⚡ PARALLEL EXECUTION IS MANDATORY

**THIS IS WRONG ❌ (Sequential - NEVER DO THIS):**

```
Message 1: Initialize swarm
Message 2: Spawn agent 1
Message 3: Spawn agent 2
Message 4: TodoWrite (single todo)
Message 5: Create file 1
Message 6: TodoWrite (another single todo)
```

**THIS IS CORRECT ✅ (Parallel - ALWAYS DO THIS):**

```
Message 1: [BatchTool]
  // MCP coordination setup
  - mcp__claude-flow__swarm_init
  - mcp__claude-flow__agent_spawn (researcher)
  - mcp__claude-flow__agent_spawn (coder)
  - mcp__claude-flow__agent_spawn (analyst)
  - mcp__claude-flow__agent_spawn (tester)
  - mcp__claude-flow__agent_spawn (coordinator)

Message 2: [BatchTool - Claude Code execution]
  // Task agents with full coordination instructions
  - Task("You are researcher agent. MANDATORY: Run hooks pre-task, post-edit, post-task. Task: Research API patterns")
  - Task("You are coder agent. MANDATORY: Run hooks pre-task, post-edit, post-task. Task: Implement REST endpoints")
  - Task("You are analyst agent. MANDATORY: Run hooks pre-task, post-edit, post-task. Task: Analyze performance")
  - Task("You are tester agent. MANDATORY: Run hooks pre-task, post-edit, post-task. Task: Write comprehensive tests")

  // TodoWrite with ALL todos batched
  - TodoWrite { todos: [
      {id: "research", content: "Research API patterns", status: "in_progress", priority: "high"},
      {id: "design", content: "Design database schema", status: "pending", priority: "high"},
      {id: "implement", content: "Build REST endpoints", status: "pending", priority: "high"},
      {id: "test", content: "Write unit tests", status: "pending", priority: "medium"},
      {id: "docs", content: "Create API documentation", status: "pending", priority: "low"},
      {id: "deploy", content: "Setup deployment", status: "pending", priority: "medium"}
    ]}

  // File operations in parallel
  - Write "api/package.json"
  - Write "api/server.js"
  - Write "api/routes/users.js"
  - Bash "mkdir -p api/{routes,models,tests}"
```

### 🎯 MANDATORY SWARM PATTERN

When given ANY complex task with swarms:

```
STEP 1: IMMEDIATE PARALLEL SPAWN (Single Message!)
[BatchTool]:
  // IMPORTANT: Check CLI args for agent count, otherwise auto-decide based on task complexity
  - mcp__claude-flow__swarm_init {
      topology: "hierarchical",
      maxAgents: CLI_ARGS.agents || AUTO_DECIDE(task_complexity), // Use CLI args or auto-decide
      strategy: "parallel"
    }

  // Spawn agents based on maxAgents count and task requirements
  // If CLI specifies 3 agents, spawn 3. If no args, auto-decide optimal count (3-12)
  - mcp__claude-flow__agent_spawn { type: "architect", name: "System Designer" }
  - mcp__claude-flow__agent_spawn { type: "coder", name: "API Developer" }
  - mcp__claude-flow__agent_spawn { type: "coder", name: "Frontend Dev" }
  - mcp__claude-flow__agent_spawn { type: "analyst", name: "DB Designer" }
  - mcp__claude-flow__agent_spawn { type: "tester", name: "QA Engineer" }
  - mcp__claude-flow__agent_spawn { type: "researcher", name: "Tech Lead" }
  - mcp__claude-flow__agent_spawn { type: "coordinator", name: "PM" }
  - TodoWrite { todos: [multiple todos at once] }

STEP 2: PARALLEL TASK EXECUTION (Single Message!)
[BatchTool]:
  - mcp__claude-flow__task_orchestrate { task: "main task", strategy: "parallel" }
  - mcp__claude-flow__memory_usage { action: "store", key: "init", value: {...} }
  - Multiple Read operations
  - Multiple Write operations
  - Multiple Bash commands

STEP 3: CONTINUE PARALLEL WORK (Never Sequential!)
```

### 📊 VISUAL TASK TRACKING FORMAT

Use this format when displaying task progress:

```
📊 Progress Overview
   ├── Total Tasks: X
   ├── ✅ Completed: X (X%)
   ├── 🔄 In Progress: X (X%)
   ├── ⭕ Todo: X (X%)
   └── ❌ Blocked: X (X%)

📋 Todo (X)
   └── 🔴 001: [Task description] [PRIORITY] ▶

🔄 In progress (X)
   ├── 🟡 002: [Task description] ↳ X deps ▶
   └── 🔴 003: [Task description] [PRIORITY] ▶

✅ Completed (X)
   ├── ✅ 004: [Task description]
   └── ... (more completed tasks)

Priority indicators: 🔴 HIGH/CRITICAL, 🟡 MEDIUM, 🟢 LOW
Dependencies: ↳ X deps | Actionable: ▶
```

### 🎯 REAL EXAMPLE: Full-Stack App Development

**Task**: "Build a complete REST API with authentication, database, and tests"

**🚨 MANDATORY APPROACH - Everything in Parallel:**

```javascript
// ✅ CORRECT: SINGLE MESSAGE with ALL operations
[BatchTool - Message 1]:
  // Initialize and spawn ALL agents at once
  mcp__claude-flow__swarm_init { topology: "hierarchical", maxAgents: 8, strategy: "parallel" }
  mcp__claude-flow__agent_spawn { type: "architect", name: "System Designer" }
  mcp__claude-flow__agent_spawn { type: "coder", name: "API Developer" }
  mcp__claude-flow__agent_spawn { type: "coder", name: "Auth Expert" }
  mcp__claude-flow__agent_spawn { type: "analyst", name: "DB Designer" }
  mcp__claude-flow__agent_spawn { type: "tester", name: "Test Engineer" }
  mcp__claude-flow__agent_spawn { type: "coordinator", name: "Lead" }

  // Update ALL todos at once - NEVER split todos!
  TodoWrite { todos: [
    { id: "design", content: "Design API architecture", status: "in_progress", priority: "high" },
    { id: "auth", content: "Implement authentication", status: "pending", priority: "high" },
    { id: "db", content: "Design database schema", status: "pending", priority: "high" },
    { id: "api", content: "Build REST endpoints", status: "pending", priority: "high" },
    { id: "tests", content: "Write comprehensive tests", status: "pending", priority: "medium" },
    { id: "docs", content: "Document API endpoints", status: "pending", priority: "low" },
    { id: "deploy", content: "Setup deployment pipeline", status: "pending", priority: "medium" },
    { id: "monitor", content: "Add monitoring", status: "pending", priority: "medium" }
  ]}

  // Start orchestration
  mcp__claude-flow__task_orchestrate { task: "Build REST API", strategy: "parallel" }

  // Store initial memory
  mcp__claude-flow__memory_usage { action: "store", key: "project/init", value: { started: Date.now() } }

[BatchTool - Message 2]:
  // Create ALL directories at once
  Bash("mkdir -p test-app/{src,tests,docs,config}")
  Bash("mkdir -p test-app/src/{models,routes,middleware,services}")
  Bash("mkdir -p test-app/tests/{unit,integration}")

  // Write ALL base files at once
  Write("test-app/package.json", packageJsonContent)
  Write("test-app/.env.example", envContent)
  Write("test-app/README.md", readmeContent)
  Write("test-app/src/server.js", serverContent)
  Write("test-app/src/config/database.js", dbConfigContent)

[BatchTool - Message 3]:
  // Read multiple files for context
  Read("test-app/package.json")
  Read("test-app/src/server.js")
  Read("test-app/.env.example")

  // Run multiple commands
  Bash("cd test-app && npm install")
  Bash("cd test-app && npm run lint")
  Bash("cd test-app && npm test")
```

### 🚫 NEVER DO THIS (Sequential = WRONG):

```javascript
// ❌ WRONG: Multiple messages, one operation each
Message 1: mcp__claude-flow__swarm_init
Message 2: mcp__claude-flow__agent_spawn (just one agent)
Message 3: mcp__claude-flow__agent_spawn (another agent)
Message 4: TodoWrite (single todo)
Message 5: Write (single file)
// This is 5x slower and wastes swarm coordination!
```

### 🔄 MEMORY COORDINATION PATTERN

Every agent coordination step MUST use memory:

```
// After each major decision or implementation
mcp__claude-flow__memory_usage
  action: "store"
  key: "swarm-{id}/agent-{name}/{step}"
  value: {
    timestamp: Date.now(),
    decision: "what was decided",
    implementation: "what was built",
    nextSteps: ["step1", "step2"],
    dependencies: ["dep1", "dep2"]
  }

// To retrieve coordination data
mcp__claude-flow__memory_usage
  action: "retrieve"
  key: "swarm-{id}/agent-{name}/{step}"

// To check all swarm progress
mcp__claude-flow__memory_usage
  action: "list"
  pattern: "swarm-{id}/*"
```

### ⚡ PERFORMANCE TIPS

1. **Batch Everything**: Never operate on single files when multiple are needed
2. **Parallel First**: Always think "what can run simultaneously?"
3. **Memory is Key**: Use memory for ALL cross-agent coordination
4. **Monitor Progress**: Use mcp**claude-flow**swarm_monitor for real-time tracking
5. **Auto-Optimize**: Let hooks handle topology and agent selection

### 🎨 VISUAL SWARM STATUS

When showing swarm status, use this format:

```
🐝 Swarm Status: ACTIVE
├── 🏗️ Topology: hierarchical
├── 👥 Agents: 6/8 active
├── ⚡ Mode: parallel execution
├── 📊 Tasks: 12 total (4 complete, 6 in-progress, 2 pending)
└── 🧠 Memory: 15 coordination points stored

Agent Activity:
├── 🟢 architect: Designing database schema...
├── 🟢 coder-1: Implementing auth endpoints...
├── 🟢 coder-2: Building user CRUD operations...
├── 🟢 analyst: Optimizing query performance...
├── 🟡 tester: Waiting for auth completion...
└── 🟢 coordinator: Monitoring progress...
```

## 📝 CRITICAL: TODOWRITE AND TASK TOOL BATCHING

### 🚨 MANDATORY BATCHING RULES FOR TODOS AND TASKS

**TodoWrite Tool Requirements:**

1. **ALWAYS** include 5-10+ todos in a SINGLE TodoWrite call
2. **NEVER** call TodoWrite multiple times in sequence
3. **BATCH** all todo updates together - status changes, new todos, completions
4. **INCLUDE** all priority levels (high, medium, low) in one call

**Task Tool Requirements:**

1. **SPAWN** all agents using Task tool in ONE message
2. **NEVER** spawn agents one by one across multiple messages
3. **INCLUDE** full task descriptions and coordination instructions
4. **BATCH** related Task calls together for parallel execution

**Example of CORRECT TodoWrite usage:**

```javascript
// ✅ CORRECT - All todos in ONE call
TodoWrite { todos: [
  { id: "1", content: "Initialize system", status: "completed", priority: "high" },
  { id: "2", content: "Analyze requirements", status: "in_progress", priority: "high" },
  { id: "3", content: "Design architecture", status: "pending", priority: "high" },
  { id: "4", content: "Implement core", status: "pending", priority: "high" },
  { id: "5", content: "Build features", status: "pending", priority: "medium" },
  { id: "6", content: "Write tests", status: "pending", priority: "medium" },
  { id: "7", content: "Add monitoring", status: "pending", priority: "medium" },
  { id: "8", content: "Documentation", status: "pending", priority: "low" },
  { id: "9", content: "Performance tuning", status: "pending", priority: "low" },
  { id: "10", content: "Deploy to production", status: "pending", priority: "high" }
]}
```

**Example of WRONG TodoWrite usage:**

```javascript
// ❌ WRONG - Multiple TodoWrite calls
Message 1: TodoWrite { todos: [{ id: "1", content: "Task 1", ... }] }
Message 2: TodoWrite { todos: [{ id: "2", content: "Task 2", ... }] }
Message 3: TodoWrite { todos: [{ id: "3", content: "Task 3", ... }] }
// This breaks parallel coordination!
```

## Claude Flow v2.0.0 Features

Claude Flow extends the base coordination with:

- **🔗 GitHub Integration** - Deep repository management
- **🎯 Project Templates** - Quick-start for common projects
- **📊 Advanced Analytics** - Detailed performance insights
- **🤖 Custom Agent Types** - Domain-specific coordinators
- **🔄 Workflow Automation** - Reusable task sequences
- **🛡️ Enhanced Security** - Safer command execution

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Examples: https://github.com/ruvnet/claude-flow/tree/main/examples

---

Remember: **Claude Flow coordinates, Claude Code creates!** Start with `mcp__claude-flow__swarm_init` to enhance your development workflow.
