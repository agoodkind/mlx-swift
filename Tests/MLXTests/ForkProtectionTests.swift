import Foundation
import MLX
import XCTest

#if canImport(Metal)
    final class ForkProtectionTests: XCTestCase {
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
            MLXFast.setPrefetchEnabled(false)
            Memory.clearCache()
            Memory.cacheLimit = previousCacheLimit
            Memory.memoryLimit = previousMemoryLimit
            super.tearDown()
        }

        func testSSDExpertReadFeedsQuantizedMatmulAndMetrics() throws {
            let url = try makeExpertFile()
            let before = MLXFast.ssdMetricsSnapshot()
            let weights = MLXFast.streamedGatherMM(
                x: MLXArray(0), wShape: .zeros([2, 1, 4], dtype: .uint32),
                activeExpert: 1, safetensorsPath: url.path, tensorName: "weight", stream: .gpu)
            try checkedEval(weights)
            XCTAssertEqual(weights.shape, [1, 1, 4])
            XCTAssertEqual(weights.asArray(UInt32.self), Array(repeating: 0x2222_2222, count: 4))

            let output = quantizedMM(
                .ones([1, 32]), weights[0], scales: .ones([1, 1]), biases: .zeros([1, 1]),
                groupSize: 32, bits: 4, stream: .gpu)
            try checkedEval(output)
            XCTAssertEqual(output.asArray(Float.self), [64])
            let after = MLXFast.ssdMetricsSnapshot()
            XCTAssertGreaterThanOrEqual(after.totalBytesRead, before.totalBytesRead + 16)
            XCTAssertGreaterThanOrEqual(after.totalChunks, before.totalChunks + 1)
        }

        func testPrefetchAndDirectReadsPreserveExpertOffsets() throws {
            let url = try makeExpertFile()
            let stacked = MLXArray.zeros([2, 1, 4], dtype: .uint32)
            try checkedEval(stacked)
            MLXFast.setPrefetchEnabled(true)
            MLXFast.pappsPrefetch(safetensorsPath: url.path, tensorName: "weight", expertIndex: 1)
            XCTAssertEqual(
                MLXFast.preadIntoOffset(
                    stacked, safetensorsPath: url.path, tensorName: "weight", expertIndex: 1,
                    dstOffset: 16), 0)
            XCTAssertEqual(
                stacked.asArray(UInt32.self),
                Array(repeating: 0, count: 4) + Array(repeating: 0x2222_2222, count: 4))

            MLXFast.setPrefetchEnabled(false)
            let single = MLXArray.zeros([1, 4], dtype: .uint32)
            try checkedEval(single)
            XCTAssertEqual(
                MLXFast.preadInto(
                    single, safetensorsPath: url.path, tensorName: "weight", expertIndex: 0), 0)
            XCTAssertEqual(single.asArray(UInt32.self), Array(repeating: 0x1111_1111, count: 4))
        }

        func testTurboKVReconstructsBothSupportedHeadSizes() throws {
            for headSize in [128, 256] {
                let input = (0 ..< headSize * 2).map { Float(($0 % 17) - 8) / 8 }
                let keys = MLXArray(input, [1, 1, 2, headSize])
                let values = keys * 0.5
                let (keyStorage, valueStorage) = MLXFast.turboQuantEncode(
                    keys: keys, values: values, stream: .cpu)
                XCTAssertEqual(keyStorage.0.shape, [1, 1, 2, 68 * headSize / 128])
                XCTAssertEqual(valueStorage.0.shape, [1, 1, 2, 50 * headSize / 128])
                XCTAssertEqual(keyStorage.0.dtype, .uint8)
                XCTAssertEqual(valueStorage.0.dtype, .uint8)

                let decodedKeys = MLXFast.turboDecodeK(packed: keyStorage.0, stream: .cpu)
                let decodedValues = MLXFast.turboDecodeV(packed: valueStorage.0, stream: .cpu)
                try checkedEval(decodedKeys, decodedValues)
                XCTAssertEqual(decodedKeys.shape, keys.shape)
                XCTAssertEqual(decodedValues.shape, values.shape)
                for (original, decoded) in [(keys, decodedKeys), (values, decodedValues)] {
                    let relativeError = sum(square(original - decoded)) / sum(square(original))
                    XCTAssertLessThan(relativeError.item(Float.self), 0.25)
                }
            }
        }

        func testNearbyLargeAllocationsReuseOneBucket() throws {
            MLX.Stream.gpu.synchronize()
            Memory.clearCache()
            Memory.cacheLimit = 1024 * 1024
            func allocateAndRelease(_ count: Int) throws {
                try autoreleasepool {
                    let array = MLXArray.zeros([count], dtype: .uint8, stream: .gpu)
                    try checkedEval(array)
                    XCTAssertEqual(array.asArray(UInt8.self).last, 0)
                    MLX.Stream.gpu.synchronize()
                }
            }
            try allocateAndRelease(33_333)
            let cachedAfterFirstAllocation = Memory.cacheMemory
            XCTAssertGreaterThanOrEqual(cachedAfterFirstAllocation, 65_536)
            try allocateAndRelease(50_000)
            XCTAssertEqual(Memory.cacheMemory, cachedAfterFirstAllocation)
        }

        func testCompatibilityFP8DecoderPreservesKnownValues() throws {
            let bytes = MLXArray([UInt8(0x00), 0x80, 0x38, 0xC0, 0x20, 0x08, 0x7E, 0xFE])
            let decoded = MLXFast.fromFp8(bytes, dtype: .float32, stream: .cpu)
            try checkedEval(decoded)
            XCTAssertEqual(
                decoded.asArray(Float.self), [0, -0.0, 1, -2, 0.125, 0.015625, 448, -448])
        }

        @MainActor func testFloat16GPUMatmulMatchesCPU() throws {
            try assertGPUMatmulMatchesCPU(dtype: .float16)
        }

        @MainActor func testBfloat16GPUMatmulMatchesCPU() throws {
            try assertGPUMatmulMatchesCPU(dtype: .bfloat16)
        }

        @MainActor private func assertGPUMatmulMatchesCPU(dtype: DType) throws {
            Memory.peakMemory = 0
            let dimension = 512
            let leftValues = (0 ..< dimension * dimension).map { Float(($0 % 29) - 14) / 32 }
            let rightValues = (0 ..< dimension * dimension).map { Float(($0 % 31) - 15) / 32 }
            let left = MLXArray(leftValues, [dimension, dimension])
            let right = MLXArray(rightValues, [dimension, dimension])
            let reference = matmul(left, right, stream: .cpu)
            let result = matmul(
                left.asType(dtype, stream: .gpu), right.asType(dtype, stream: .gpu), stream: .gpu)
            try checkedEval(reference, result)
            let resultFloat = result.asType(.float32, stream: .cpu)
            XCTAssertTrue(
                allClose(reference, resultFloat, rtol: 0.015, atol: 0.015, stream: .cpu)
                    .item(Bool.self))
            let difference = subtract(reference, resultFloat, stream: .cpu)
            let maxError = MLX.max(abs(difference, stream: .cpu), stream: .cpu).item(Float.self)
            let peakBytes = Memory.peakMemory
            XCTAssertLessThanOrEqual(peakBytes, 64 * 1024 * 1024)
            let architecture = GPU.deviceInfo().architecture
            XCTContext.runActivity(named: "GPU matrix multiplication comparison") { activity in
                let attachment = XCTAttachment(
                    string: "architecture=\(architecture) dtype=\(dtype) dimension=\(dimension) "
                        + "max_absolute_error=\(maxError) peak_bytes=\(peakBytes)")
                attachment.lifetime = .keepAlways
                activity.add(attachment)
            }
        }

        private func makeExpertFile() throws -> URL {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            addTeardownBlock {
                try FileManager.default.removeItem(at: directory)
            }
            let url = directory.appendingPathComponent("experts.safetensors")
            let values =
                Array(repeating: UInt32(0x1111_1111), count: 4)
                + Array(repeating: UInt32(0x2222_2222), count: 4)
            try MLX.save(arrays: ["weight": MLXArray(values, [2, 1, 4])], url: url)
            return url
        }
    }
#endif
