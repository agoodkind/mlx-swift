// swift-tools-version: 6.2
import PackageDescription

#if os(Linux)
    let dependencies: [Package.Dependency] = [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.0.0")
    ]
    let helperTargets: [Target] = [
        .plugin(name: "CudaBuild", capability: .buildTool()),
        .plugin(name: "OtherBuild", capability: .buildTool()),
        .executableTarget(name: "encuda"),
    ]
#else
    let dependencies: [Package.Dependency] = []
    let helperTargets: [Target] = [
        .plugin(name: "__PLUGIN_NAME__", capability: .buildTool())
    ]
#endif

let package = Package(
    name: "ManifestValidationFixture",
    dependencies: dependencies,
    targets: [
        .target(name: "Cmlx", plugins: [.plugin(name: "__PLUGIN_NAME__")])
    ] + helperTargets
)
