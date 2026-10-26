// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Stress concurrent nested compiled-function calls.
///
/// A callback can execute during tracing or eager evaluation. Its execution
/// does not prove that a compile trace ran. A scheduling delay also does not
/// prove that another thread acquired a compiled function's instance lock.
/// CompileLockOrderingTests checks the production lock order with instrumentation.
///
/// Run with --device cpu|gpu. Metal resources must match the linked MLX build,
/// including when selecting the CPU device.
/// Exit 1 reports a completion timeout. Exit 2 reports an inconclusive run.
@main
struct CompileLockRepro {
    static let deadline: TimeInterval = 10
    static let widenMicroseconds: UInt32 = 200_000
    static let rounds = 20

    // This lock protects the flag without depending on MLX locks.
    final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false

        func set() {
            lock.lock()
            value = true
            lock.unlock()
        }

        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    static func die(_ code: Int32, _ message: String) -> Never {
        print(message)
        fflush(stdout)
        // Avoid waiting for worker cleanup after a completion timeout.
        _exit(code)
    }

    static func main() {
        var device = Device.cpu
        if let index = CommandLine.arguments.firstIndex(of: "--device") {
            guard index + 1 < CommandLine.arguments.count else {
                die(2, "--device requires cpu or gpu.")
            }
            switch CommandLine.arguments[index + 1].lowercased() {
            case "cpu": device = .cpu
            case "gpu": device = .gpu
            default: die(2, "Unknown --device \(CommandLine.arguments[index + 1]).")
            }
        }
        print(
            "compile-lock stress: device=\(device), \(rounds) rounds, \(Int(deadline))s deadline per round"
        )

        // Retain compiled functions until every round finishes.
        var keepAlive: [Any] = []
        var executedRounds = 0
        for round in 1 ... rounds {
            if runRound(round, device: device, keepAlive: &keepAlive) {
                executedRounds += 1
            }
        }
        die(
            2,
            """
            INCONCLUSIVE: \(rounds) stress rounds completed; the callback executed in \(executedRounds) rounds.
            Callback execution does not distinguish tracing from eager evaluation.
            The schedule does not prove instance-lock contention.
            Run CompileLockOrderingTests to verify the instrumented production lock order.
            """)
    }

    static func runRound(_ round: Int, device: Device, keepAlive: inout [Any]) -> Bool {
        let gate = DispatchSemaphore(value: 0)
        let hDone = DispatchSemaphore(value: 0)
        let wDone = DispatchSemaphore(value: 0)
        let bodyExecuted = Flag()
        let stop = Flag()

        let shared: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
            x * 2 + 1
        }
        let outer: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
            bodyExecuted.set()
            gate.signal()
            // This pause changes scheduling opportunities without proving contention.
            usleep(widenMicroseconds)
            return shared(x) + 3
        }
        keepAlive.append(shared)
        keepAlive.append(outer)

        let h = Thread {
            Device.withDefaultDevice(device) {
                let input = MLXArray([1.0, 2.0, 3.0] as [Float])
                let result = outer(input)
                eval(result)
                hDone.signal()
            }
        }
        h.name = "H-nested-compiled-call"

        let w = Thread {
            Device.withDefaultDevice(device) {
                gate.wait()
                let input = MLXArray([4.0, 5.0, 6.0] as [Float])
                var iterations = 0
                repeat {
                    let result = shared(input)
                    eval(result)
                    iterations += 1
                } while !stop.isSet && iterations < 1000
                wDone.signal()
            }
        }
        w.name = "W-plain-compiled-call"

        h.start()
        w.start()

        if hDone.wait(timeout: .now() + deadline) == .timedOut {
            die(
                1,
                """
                TIMEOUT: round \(round): thread H did not finish within \(Int(deadline))s.
                callbackExecuted=\(bodyExecuted.isSet). The blocking lock is unknown.
                """)
        }
        stop.set()
        if wDone.wait(timeout: .now() + deadline) == .timedOut {
            die(
                1,
                "TIMEOUT: round \(round): thread W did not finish within \(Int(deadline))s after H."
            )
        }

        print("  round \(round): completed (callbackExecuted=\(bodyExecuted.isSet))")
        return bodyExecuted.isSet
    }
}
