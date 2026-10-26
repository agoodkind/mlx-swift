// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX

// The ownership bookkeeping these tests read is debug-only.
#if DEBUG

    /// The `evalLock` → `CompiledFunction.lock` ordering invariant.
    ///
    /// `Source/CompileLockRepro` shows that the reverse order deadlocks.  It
    /// lives in its own executable because a failure wedges the process, and a
    /// passing run of it only shows that the race did not fire -- not that the
    /// ordering is right.  These tests assert the ordering itself, cannot hang,
    /// and are safe to run in the shared test process.
    ///
    /// A violation trips an `assert` at the point of acquisition, so a
    /// regression traps there rather than reaching the counter assertions
    /// below.  The counters are what make a *pass* meaningful: they show the
    /// check was reached at all, rather than the code path never running.
    class CompileLockOrderingTests: XCTestCase {

        private var previousMemoryLimit = 0
        private var previousCacheLimit = 0

        override func setUp() {
            super.setUp()
            previousMemoryLimit = Memory.memoryLimit
            previousCacheLimit = Memory.cacheLimit
            Memory.memoryLimit = 64 * 1024 * 1024
            Memory.cacheLimit = 0
        }

        override func tearDown() {
            Memory.clearCache()
            Memory.cacheLimit = previousCacheLimit
            Memory.memoryLimit = previousMemoryLimit
            super.tearDown()
        }

        func testDeinitReturnsWhileTraceWaitsForCallerLock() {
            final class Outcome: @unchecked Sendable {
                let callerLock = NSLock()
                let insideTrace = DispatchSemaphore(value: 0)
                let finished = DispatchSemaphore(value: 0)
                var acquiredCallerLock = false
                var result: Float = .nan
            }
            let outcome = Outcome()
            let victim = Victim()
            victim.warm()
            outcome.callerLock.lock()

            let tracer = Thread {
                let compiled = MLX.compile { (input: MLXArray) in
                    outcome.insideTrace.signal()
                    outcome.acquiredCallerLock = outcome.callerLock.lock(
                        before: Date(timeIntervalSinceNow: 2))
                    if outcome.acquiredCallerLock {
                        outcome.callerLock.unlock()
                    }
                    return input + 2
                }
                let output = compiled(MLXArray(Float(5)))
                eval(output)
                outcome.result = output.item(Float.self)
                outcome.finished.signal()
            }
            tracer.start()
            let enteredTrace = outcome.insideTrace.wait(timeout: .now() + 5)
            if enteredTrace != .success {
                outcome.callerLock.unlock()
                XCTFail("The compiled function did not start tracing.")
                return
            }

            victim.release()
            outcome.callerLock.unlock()
            guard outcome.finished.wait(timeout: .now() + 5) == .success else {
                XCTFail("The compiled function did not finish after releasing the caller lock.")
                return
            }
            XCTAssertTrue(
                outcome.acquiredCallerLock,
                "CompiledFunction.deinit waited for evalLock while the trace waited for the caller lock."
            )
            XCTAssertEqual(outcome.result, 7)
        }

        /// The instrumentation must be able to report "not held", otherwise a
        /// violation count of zero would mean nothing.
        func testOwnershipTrackingIsNotVacuous() {
            XCTAssertFalse(EvalLockOwnership.isHeldByCurrentThread)
            XCTAssertEqual(EvalLockOwnership.depth, 0)

            withEvalLock {
                XCTAssertTrue(EvalLockOwnership.isHeldByCurrentThread)
                XCTAssertEqual(EvalLockOwnership.depth, 1)

                // evalLock is recursive; nesting is expected and must be counted
                withEvalLock {
                    XCTAssertEqual(EvalLockOwnership.depth, 2)
                }
                XCTAssertEqual(EvalLockOwnership.depth, 1)
            }

            XCTAssertFalse(EvalLockOwnership.isHeldByCurrentThread)
            XCTAssertEqual(EvalLockOwnership.depth, 0)
        }

        /// Ownership is per thread: another thread must not believe it holds the
        /// lock this one is holding.  (The other thread only reads its own
        /// bookkeeping -- it never waits on `evalLock` -- so this cannot hang.)
        func testOwnershipIsPerThread() {
            final class Result: @unchecked Sendable {
                var heldOnOtherThread = true
            }
            let result = Result()
            let done = DispatchSemaphore(value: 0)

            withEvalLock {
                let thread = Thread {
                    result.heldOnOtherThread = EvalLockOwnership.isHeldByCurrentThread
                    done.signal()
                }
                thread.start()
                XCTAssertEqual(done.wait(timeout: .now() + 10), .success)
            }

            XCTAssertFalse(result.heldOnOtherThread)
        }

        /// A compiled function with observed state really takes the per-instance
        /// lock -- a stateless one has nothing to guard and skips it -- so this
        /// exercises the checked acquisition.
        func testEvalLockIsHeldWhenTheInstanceLockIsTaken() {
            EvalLockOwnership.resetCounters()

            let state = MLXArray(2)
            func body(_ x: [MLXArray]) -> [MLXArray] {
                [x[0] + state]
            }
            let compiled = compile(inputs: [state], outputs: [state], body(_:))

            let r = compiled([MLXArray(5)])
            eval(r)
            XCTAssertEqual(r[0].item(Float.self), 7)

            let counters = EvalLockOwnership.counters
            XCTAssertGreaterThan(
                counters.checks, 0,
                "the ordering invariant was never checked, so this test proves nothing")
            XCTAssertEqual(
                counters.violations, 0,
                "CompiledFunction.lock was taken without evalLock held")
        }

        /// The deadlock shape, single threaded: one compiled function called
        /// from inside another's trace.  The nested acquisition happens while an
        /// outer `evalLock` is already held, which is exactly the ordering the
        /// invariant describes.
        func testNestedCompiledCallKeepsTheOrdering() {
            EvalLockOwnership.resetCounters()

            let innerState = MLXArray(1)
            func innerBody(_ x: [MLXArray]) -> [MLXArray] {
                [x[0] * 2 + innerState]
            }
            let inner = compile(inputs: [innerState], outputs: [innerState], innerBody(_:))

            let outerState = MLXArray(3)
            nonisolated(unsafe) var tracedNested = false
            func outerBody(_ x: [MLXArray]) -> [MLXArray] {
                tracedNested = true
                return [inner([x[0]])[0] + outerState]
            }
            let outer = compile(inputs: [outerState], outputs: [outerState], outerBody(_:))

            let r = outer([MLXArray(5)])
            eval(r)
            XCTAssertEqual(r[0].item(Float.self), 14)

            XCTAssertTrue(
                tracedNested,
                "the traced body never ran, so no nested compiled call was exercised")

            let counters = EvalLockOwnership.counters
            XCTAssertGreaterThan(counters.checks, 0)
            XCTAssertEqual(
                counters.violations, 0,
                "a nested CompiledFunction.lock was taken without evalLock held")
        }

        private final class Victim: @unchecked Sendable {
            private var compiled: ((MLXArray) -> MLXArray)?

            init() {
                let bias = MLXArray(Float(2))
                compiled = MLX.compile(inputs: [bias]) { (input: MLXArray) in input * 3 + bias }
            }

            func warm() {
                let output = compiled!(MLXArray(1))
                eval(output)
                XCTAssertEqual(output.item(Float.self), 5)
            }

            func release() {
                compiled = nil
            }
        }

        private func waitForCompileErases(_ count: Int) {
            let cleanup = XCTNSPredicateExpectation(
                predicate: NSPredicate { _, _ in EvalLockOwnership.compileErases >= count },
                object: nil)
            wait(for: [cleanup], timeout: 5)
        }

        func testDeferredEraseWaitsForEvalLock() {
            let victim = Victim()
            victim.warm()
            let released = DispatchSemaphore(value: 0)
            let erasesBeforeRelease = withEvalLock {
                let before = EvalLockOwnership.compileErases
                let releaser = Thread {
                    victim.release()
                    released.signal()
                }
                releaser.start()
                XCTAssertEqual(released.wait(timeout: .now() + 5), .success)
                XCTAssertEqual(EvalLockOwnership.compileErases, before)
                return before
            }
            waitForCompileErases(erasesBeforeRelease + 1)
            XCTAssertEqual(EvalLockOwnership.counters.violations, 0)
        }

        func testNewFunctionsDoNotReuseGraphsBeforeDeferredCleanup() {
            let rounds = 64
            let expectedErases = withEvalLock {
                let before = EvalLockOwnership.compileErases
                // Cleanup cannot acquire evalLock until every result has been checked.
                for round in 0 ..< rounds {
                    autoreleasepool {
                        let compiled = MLX.compile { (input: MLXArray) in
                            input + MLXArray(Float(round))
                        }
                        let result = compiled(MLXArray(Float(1)))
                        eval(result)
                        XCTAssertEqual(result.item(Float.self), Float(1 + round))
                    }
                }
                XCTAssertEqual(EvalLockOwnership.compileErases, before)
                return before + rounds
            }
            waitForCompileErases(expectedErases)
        }

        func testConcurrentDeinitErasesAndCompiledCalls() {
            let rounds = 64
            let start = DispatchSemaphore(value: 0)
            let ready = DispatchSemaphore(value: 0)
            let done = DispatchSemaphore(value: 0)
            let erasesBefore = EvalLockOwnership.compileErases

            final class Outcome: @unchecked Sendable {
                var callerValues = [Float]()
                var releases = 0
            }
            let outcome = Outcome()

            let caller = Thread {
                ready.signal()
                guard start.wait(timeout: .now() + 5) == .success else { return }
                var values = [Float]()
                for round in 0 ..< rounds {
                    let compiled = MLX.compile { (input: MLXArray) in
                        input + MLXArray(Float(round))
                    }
                    let result = compiled(MLXArray(Float(1)))
                    eval(result)
                    values.append(result.item(Float.self))
                }
                outcome.callerValues = values
                done.signal()
            }

            let releaser = Thread {
                ready.signal()
                guard start.wait(timeout: .now() + 5) == .success else { return }
                var released = 0
                for _ in 0 ..< rounds {
                    let victim = Victim()
                    victim.warm()
                    victim.release()
                    released += 1
                }
                outcome.releases = released
                done.signal()
            }

            caller.start()
            releaser.start()
            XCTAssertEqual(ready.wait(timeout: .now() + 5), .success)
            XCTAssertEqual(ready.wait(timeout: .now() + 5), .success)
            start.signal()
            start.signal()
            guard done.wait(timeout: .now() + 10) == .success else {
                XCTFail("A compiled call or function release did not finish.")
                return
            }
            guard done.wait(timeout: .now() + 10) == .success else {
                XCTFail("A compiled call or function release did not finish.")
                return
            }

            XCTAssertEqual(outcome.releases, rounds)
            XCTAssertEqual(outcome.callerValues.count, rounds)
            for (round, value) in outcome.callerValues.enumerated() {
                XCTAssertEqual(value, Float(1 + round))
            }
            waitForCompileErases(erasesBefore + rounds * 2)
            XCTAssertGreaterThan(EvalLockOwnership.counters.checks, 0)
            XCTAssertEqual(EvalLockOwnership.counters.violations, 0)
        }
    }

#endif
