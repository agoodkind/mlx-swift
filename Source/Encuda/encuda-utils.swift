import Foundation

#if canImport(Glibc)
    import Glibc
#elseif canImport(Musl)
    import Musl
#elseif canImport(Darwin)
    import Darwin
#endif

func searchForCommand(_ name: String) -> URL? {
    let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
    for folder in path.split(separator: ":") {
        let url = URL(fileURLWithPath: String(folder)).appendingPathComponent(name)
        if FileManager.default.isExecutableFile(atPath: url.path) {
            return url
        }
    }
    return nil
}

#if os(macOS) || os(Linux)
    /// Spawn and reap the same child without Foundation process bookkeeping.
    ///
    /// During parallel arm64 Linux CUDA builds, nvcc completed its output but
    /// remained a zombie while Foundation's waitUntilExit() and isRunning stalled.
    /// POSIX spawning gives this function exclusive ownership of the child.
    /// The tool inherits the current environment and standard file descriptors.
    func runTool(executable: URL, arguments: [String]) throws -> Int32 {
        let path = executable.path
        let environment = ProcessInfo.processInfo.environment.map { "\($0.key)=\($0.value)" }
        var processIdentifier: pid_t = 0
        let spawnStatus = try withCStringArray([path] + arguments) { argumentPointers in
            try withCStringArray(environment) { environmentPointers in
                posix_spawn(
                    &processIdentifier, path, nil, nil, argumentPointers, environmentPointers)
            }
        }
        guard spawnStatus == 0 else {
            throw processError("start \(path)", code: spawnStatus)
        }

        var status: Int32 = 0
        while true {
            let reaped = waitpid(processIdentifier, &status, 0)
            if reaped == processIdentifier {
                // POSIX wait status stores the signal in the low seven bits and
                // the normal exit code in the next eight bits.
                let terminatingSignal = status & 0x7F
                if terminatingSignal == 0 {
                    return (status >> 8) & 0xFF
                }
                return 128 + terminatingSignal
            }
            let waitError = errno
            if waitError == EINTR {
                continue
            }
            throw processError("wait for \(path) (PID \(processIdentifier))", code: waitError)
        }
    }

    private func processError(_ operation: String, code: Int32) -> NSError {
        NSError(
            domain: NSPOSIXErrorDomain, code: Int(code),
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Unable to \(operation): \(String(cString: strerror(code)))."
            ])
    }

    private func withCStringArray<Result>(
        _ values: [String],
        _ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) throws -> Result
    ) throws -> Result {
        var pointers = [UnsafeMutablePointer<CChar>?]()
        defer {
            for pointer in pointers {
                free(pointer)
            }
        }
        for value in values {
            guard let pointer = strdup(value) else {
                throw processError("allocate subprocess arguments", code: ENOMEM)
            }
            pointers.append(pointer)
        }
        pointers.append(nil)
        return try pointers.withUnsafeMutableBufferPointer { buffer in
            try body(buffer.baseAddress!)
        }
    }
#endif
