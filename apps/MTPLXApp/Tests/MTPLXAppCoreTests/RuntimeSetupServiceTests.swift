import XCTest
@testable import MTPLXAppCore

// MARK: - RuntimeSetupServiceTests
//
// Exercises the onboarding "Setting up MTPLX" service with injected
// engine/fan-control/brew closures and fake executables under an
// isolated HOME — no real installs, no network, no real brew.

final class RuntimeSetupServiceTests: XCTestCase {
    private final class CallCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0

        func increment() {
            lock.lock()
            value += 1
            lock.unlock()
        }

        func count() -> Int {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    private struct SetupRun {
        var snapshots: [[RuntimeSetupRow]]
        var outcome: RuntimeSetupOutcome?

        func row(_ id: RuntimeSetupRowID) -> RuntimeSetupRow? {
            outcome?.rows.first { $0.id == id } ?? snapshots.last?.first { $0.id == id }
        }
    }

    private func run(_ service: RuntimeSetupService) async -> SetupRun {
        var snapshots: [[RuntimeSetupRow]] = []
        var outcome: RuntimeSetupOutcome?
        for await event in service.stream() {
            switch event {
            case .rows(let rows):
                snapshots.append(rows)
            case .finished(let finished):
                outcome = finished
            }
        }
        return SetupRun(snapshots: snapshots, outcome: outcome)
    }

    private func isolatedEnvironment(home: URL, pathDir: URL) -> [String: String] {
        [
            "HOME": home.path,
            "PATH": pathDir.path,
            "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
        ]
    }

    private func makeFakeCLI(in directory: URL, version: String) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("mtplx")
        try """
        #!/bin/sh
        if [ "$1" = "--version" ]; then
          echo "mtplx \(version) (\(version))"
          exit 0
        fi
        echo ok
        """.data(using: .utf8)!.write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func temporaryDirectory() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("runtime-setup-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func fanControlOK() -> RuntimeSetupService.FanControlEnsurer {
        { _, status in
            status("Fan control ready")
            return FanControlSetupResult(ok: true, exitCode: 0, message: "Fan control ready")
        }
    }

    // MARK: Engine

    func testEngineFailureBlocksSetupAndLeavesLaterRowsPending() async throws {
        struct InstallError: LocalizedError {
            var errorDescription: String? { "Python 3.11 or newer was not found." }
        }
        let home = temporaryDirectory()
        let pathDir = home.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: pathDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: pathDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in throw InstallError() },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        XCTAssertEqual(result.outcome?.engineReady, false)
        XCTAssertNil(result.outcome?.executablePath)
        XCTAssertEqual(result.row(.engine)?.state, .failed)
        XCTAssertEqual(result.row(.engine)?.detail, "Python 3.11 or newer was not found.")
        XCTAssertEqual(result.row(.fanControl)?.state, .pending)
        XCTAssertEqual(result.row(.globalCLI)?.state, .pending)
    }

    func testFanControlFailureDegradesToWarningAndSetupCompletes() async throws {
        let home = temporaryDirectory()
        let engineDir = home.appendingPathComponent("engine", isDirectory: true)
        let engine = try makeFakeCLI(in: engineDir, version: "1.0.0")
        let pathDir = home.appendingPathComponent("bin", isDirectory: true)
        try FileManager.default.createDirectory(at: pathDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: pathDir),
            appVersion: "1.0.0",
            engineInstaller: { status in
                status("Installing MTPLX runtime")
                return engine
            },
            fanControlEnsurer: { _, _ in
                FanControlSetupResult(ok: false, exitCode: 1, message: "No supported fan tool")
            }
        )
        let result = await run(service)

        XCTAssertEqual(result.outcome?.engineReady, true)
        XCTAssertEqual(result.outcome?.executablePath, engine.path)
        XCTAssertEqual(result.row(.engine)?.state, .done)
        XCTAssertEqual(result.row(.engine)?.detail, "MTPLX 1.0.0 ready")
        XCTAssertEqual(result.row(.fanControl)?.state, .warning)
        XCTAssertTrue(result.row(.fanControl)?.detail.contains("safe defaults") == true)
    }

    // MARK: Global CLI sync

    func testUpgradesOldHomebrewCLIExactlyOnce() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        let staleCLI = try makeFakeCLI(in: globalDir, version: "0.3.7")
        let upgraded = try makeFakeCLI(in: home.appendingPathComponent("brew-upgraded"), version: "1.0.0")
        _ = staleCLI

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"
        let upgrades = CallCounter()

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: {
                upgrades.increment()
                return upgraded
            }
        )
        let result = await run(service)

        XCTAssertEqual(upgrades.count(), 1, "brew upgrade should run exactly once")
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertTrue(
            result.row(.globalCLI)?.detail.contains("updated to 1.0.0") == true,
            result.row(.globalCLI)?.detail ?? "nil"
        )
        XCTAssertEqual(result.outcome?.engineReady, true)
    }

    func testHomebrewUpgradeFailureFallsBackToShim() async throws {
        struct BrewError: LocalizedError {
            var errorDescription: String? { "brew upgrade exited 1" }
        }
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "0.3.7")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: { throw BrewError() }
        )
        let result = await run(service)

        XCTAssertEqual(result.outcome?.engineReady, true, "CLI sync must never block setup")
        // brew failing must not leave the user on a stale CLI — the
        // shim shadows it so the terminal still serves the engine.
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertNil(result.row(.globalCLI)?.command)
        let shim = home.appendingPathComponent(".mtplx/bin/mtplx")
        XCTAssertEqual(
            try FileManager.default.destinationOfSymbolicLink(atPath: shim.path),
            engine.path
        )
    }

    /// The app is not polite about stale CLIs: anything older than the
    /// app gets the shim put in front of it on PATH, automatically.
    /// The old install is shadowed, never modified.
    func testStalePipCLIIsUpdatedAutomaticallyViaShim() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        let stale = try makeFakeCLI(in: globalDir, version: "0.3.7")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "pipLike"
        let upgrades = CallCounter()

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: {
                upgrades.increment()
                return engine
            }
        )
        let result = await run(service)

        XCTAssertEqual(upgrades.count(), 0, "pip installs never go through brew")
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertNil(result.row(.globalCLI)?.command, "no manual command — the app already fixed it")
        XCTAssertEqual(result.outcome?.engineReady, true)

        let shim = home.appendingPathComponent(".mtplx/bin/mtplx")
        XCTAssertEqual(
            try FileManager.default.destinationOfSymbolicLink(atPath: shim.path),
            engine.path
        )
        let zshrc = try String(
            contentsOf: home.appendingPathComponent(".zshrc"),
            encoding: .utf8
        )
        XCTAssertTrue(zshrc.contains(".mtplx/bin"), "PATH line must put the shim in front")
        XCTAssertTrue(
            FileManager.default.isExecutableFile(atPath: stale.path),
            "the user's old CLI file is shadowed, not deleted"
        )
    }

    /// The founder's edge case: a CLI newer than the app is the user's
    /// business — no shim, no downgrade, no nagging.
    func testNewerThanAppCLIIsLeftAlone() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "1.1.0")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "pipLike"

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: home.appendingPathComponent(".mtplx/bin/mtplx").path
            ),
            "newer CLI must not be shadowed"
        )
    }

    func testMissingGlobalCLIInstallsTerminalShimAndPATHLine() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertTrue(
            result.row(.globalCLI)?.detail.contains("Installed the mtplx command") == true,
            result.row(.globalCLI)?.detail ?? "nil"
        )
        XCTAssertEqual(result.outcome?.engineReady, true)

        let shim = home.appendingPathComponent(".mtplx/bin/mtplx")
        XCTAssertEqual(
            try FileManager.default.destinationOfSymbolicLink(atPath: shim.path),
            engine.path,
            "Shim must symlink to the app-owned engine"
        )
        let zshrc = try String(
            contentsOf: home.appendingPathComponent(".zshrc"),
            encoding: .utf8
        )
        XCTAssertTrue(zshrc.contains(#"export PATH="$HOME/.mtplx/bin:$PATH""#), zshrc)
    }

    func testTerminalShimInstallIsIdempotent() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        _ = await run(service)
        let second = await run(service)

        XCTAssertEqual(second.row(.globalCLI)?.state, .done)
        XCTAssertEqual(second.row(.globalCLI)?.detail, "mtplx command ready.")
        let zshrc = try String(
            contentsOf: home.appendingPathComponent(".zshrc"),
            encoding: .utf8
        )
        let occurrences = zshrc.components(separatedBy: ".mtplx/bin:").count - 1
        XCTAssertEqual(occurrences, 1, "PATH line must not be duplicated:\n\(zshrc)")
    }

    func testExistingBrewCLIIsNotShadowedByShim() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "1.0.0")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        _ = await run(service)

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: home.appendingPathComponent(".mtplx/bin/mtplx").path),
            "An up-to-date CLI must never be shadowed by the shim"
        )
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: home.appendingPathComponent(".zshrc").path),
            "The shell profile must not be touched when the CLI is already current"
        )
    }

    func testUpToDateGlobalCLIIsLeftAlone() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let globalDir = home.appendingPathComponent("global-bin", isDirectory: true)
        _ = try makeFakeCLI(in: globalDir, version: "1.0.0")

        var environment = isolatedEnvironment(home: home, pathDir: globalDir)
        environment["MTPLX_APP_FAKE_INSTALL_KIND"] = "homebrew"
        let upgrades = CallCounter()

        let service = RuntimeSetupService(
            processEnvironment: environment,
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK(),
            homebrewUpgrader: {
                upgrades.increment()
                return engine
            }
        )
        let result = await run(service)

        XCTAssertEqual(upgrades.count(), 0)
        XCTAssertEqual(result.row(.globalCLI)?.state, .done)
        XCTAssertTrue(
            result.row(.globalCLI)?.detail.contains("Up to date (1.0.0)") == true,
            result.row(.globalCLI)?.detail ?? "nil"
        )
    }

    func testRowsArePublishedInCanonicalOrder() async throws {
        let home = temporaryDirectory()
        let engine = try makeFakeCLI(in: home.appendingPathComponent("engine"), version: "1.0.0")
        let emptyDir = home.appendingPathComponent("empty-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)

        let service = RuntimeSetupService(
            processEnvironment: isolatedEnvironment(home: home, pathDir: emptyDir),
            appVersion: "1.0.0",
            engineInstaller: { _ in engine },
            fanControlEnsurer: fanControlOK()
        )
        let result = await run(service)

        for snapshot in result.snapshots {
            XCTAssertEqual(snapshot.map(\.id), RuntimeSetupRowID.allCases)
        }
        XCTAssertEqual(result.outcome?.rows.map(\.id), RuntimeSetupRowID.allCases)
    }
}
