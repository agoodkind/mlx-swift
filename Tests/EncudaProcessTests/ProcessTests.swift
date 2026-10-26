import Foundation

#if canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#elseif canImport(Darwin)
    import Darwin
#endif

private struct TestFailure: Error, CustomStringConvertible {
    let description: String
}

// The lock protects the messages written by concurrent subprocess callers.
private final class Failures: @unchecked Sendable {
    private let lock = NSLock()
    private var messages = [String]()

    func append(_ message: String) {
        lock.withLock { messages.append(message) }
    }

    var all: [String] {
        lock.withLock { messages }
    }
}

@main
struct EncudaProcessTests {
    static func main() {
        do {
            try runChildIfRequested()
            try verifyProcesses()
            print(
                "Encuda process tests passed: exits, signals, arguments, environment, output, and 128 concurrent true/false children."
            )
        } catch {
            print("Encuda process test failed: \(error)")
            exit(1)
        }
    }

    private static func runChildIfRequested() throws {
        let arguments = CommandLine.arguments
        guard arguments.count > 1 else { return }
        switch arguments[1] {
        case "--child-exit":
            guard arguments.count == 3, let status = Int32(arguments[2]) else {
                throw TestFailure(description: "Provide one child exit status.")
            }
            _exit(status)
        case "--child-signal":
            raise(SIGTERM)
            _exit(99)
        case "--child-arguments":
            guard arguments.count == 6,
                arguments[2] == "value with spaces",
                arguments[3] == "quotes'\"",
                arguments[4] == "λ",
                arguments[5] == ProcessInfo.processInfo.environment["PATH"]
            else {
                _exit(98)
            }
            _exit(0)
        case "--child-output":
            guard arguments.count == 3 else {
                throw TestFailure(description: "Provide the child output path.")
            }
            try Data("completed subprocess output".utf8).write(
                to: URL(fileURLWithPath: arguments[2]))
            _exit(23)
        default:
            throw TestFailure(description: "Unknown subprocess test mode: \(arguments[1])")
        }
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw TestFailure(description: message) }
    }

    private static func verifyProcesses() throws {
        guard let trueCommand = searchForCommand("true"),
            let falseCommand = searchForCommand("false"),
            let path = ProcessInfo.processInfo.environment["PATH"]
        else {
            throw TestFailure(description: "Provide PATH with the true and false commands.")
        }
        let executable = URL(fileURLWithPath: CommandLine.arguments[0])
        for expected: Int32 in [0, 1, 7, 37, 127, 255] {
            let status = try runTool(
                executable: executable, arguments: ["--child-exit", String(expected)])
            try require(status == expected, "Expected exit \(expected), received \(status).")
        }
        let signalStatus = try runTool(executable: executable, arguments: ["--child-signal"])
        try require(signalStatus == 128 + SIGTERM, "A signal exit did not preserve SIGTERM.")
        let argumentStatus = try runTool(
            executable: executable,
            arguments: ["--child-arguments", "value with spaces", "quotes'\"", "λ", path])
        try require(argumentStatus == 0, "Subprocess arguments or environment changed.")

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("output.txt")
        let outputStatus = try runTool(
            executable: executable, arguments: ["--child-output", output.path])
        try require(outputStatus == 23, "The completed output child lost its exit status.")
        let contents = try String(contentsOf: output, encoding: .utf8)
        try require(
            contents == "completed subprocess output", "The child did not finish its output.")

        do {
            _ = try runTool(
                executable: directory.appendingPathComponent("missing-tool"), arguments: [])
            throw TestFailure(description: "A missing executable unexpectedly started.")
        } catch let error as NSError where error.domain == NSPOSIXErrorDomain {
            try require(error.code == Int(ENOENT), "A missing executable lost its POSIX error.")
        }

        let failures = Failures()
        DispatchQueue.concurrentPerform(iterations: 128) { iteration in
            let expected: Int32 = Int32(iteration % 2)
            let command = expected == 0 ? trueCommand : falseCommand
            do {
                let status = try runTool(executable: command, arguments: [])
                if status != expected {
                    failures.append("Child \(iteration): expected \(expected), received \(status).")
                }
            } catch {
                failures.append("Child \(iteration): \(error)")
            }
        }
        try require(failures.all.isEmpty, failures.all.joined(separator: "\n"))
    }
}
