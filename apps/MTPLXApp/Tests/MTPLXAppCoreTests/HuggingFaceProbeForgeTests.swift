import XCTest
@testable import MTPLXAppCore

final class HuggingFaceProbeForgeTests: XCTestCase {
    // MARK: - HTTPRunner harness

    /// Routes URLs to canned JSON responses for the three endpoints
    /// HuggingFaceProbe touches:
    ///
    ///   /<repo>/resolve/main/mtplx_runtime.json
    ///   /<repo>/resolve/main/config.json
    ///   /api/models/<repo>/tree/main
    ///
    /// Missing entries return 404; throwing entries simulate network
    /// errors (URLError-style). Comparisons are case-insensitive on
    /// path so cases like cyankiwi vs CyanKiwi don't break tests.
    private final class FakeRunner: @unchecked Sendable {
        var responses: [String: (Int, Data)] = [:]
        var errors: Set<String> = []

        func install(url: String, status: Int = 200, body: String) {
            responses[url] = (status, body.data(using: .utf8) ?? Data())
        }

        func runner() -> HuggingFaceProbe.HTTPRunner {
            let snapshot = responses
            let snapshotErrors = errors
            return { url, _ in
                let key = url.absoluteString
                if snapshotErrors.contains(key) {
                    throw URLError(.notConnectedToInternet)
                }
                if let (status, data) = snapshot[key] {
                    return (status, data)
                }
                return (404, Data())
            }
        }
    }

    // MARK: - Already-MTPLX short-circuit

    func testForgeProbeAlreadyMTPLXShortCircuitsBeforeFetchingConfig() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed/resolve/main/mtplx_runtime.json",
            body: """
            {
              "mtplx_version": "0.1.0-preview",
              "arch_id": "qwen3-next-mtp",
              "mtp_depth_max": 3,
              "mtp_sidecar": "Qwen3.6-27B-MTPLX-CyanKiwi-Packed-BF16-INT4-v3"
            }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")
        XCTAssertEqual(result.verdict, .alreadyMTPLX)
        XCTAssertEqual(result.sourceFormat, .mlxAffineWithMtp)
        XCTAssertTrue(result.message.contains("depth 3"), "Verdict surfaces the verified MTP depth")
    }

    func testForgeProbeAlreadyMTPLXWithoutSidecarFlagsMlxAffine() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/foo/Qwen3.6-MTPLX-NoSidecar/resolve/main/mtplx_runtime.json",
            body: """
            { "mtplx_version": "1.0.0", "arch_id": "qwen3-next-mtp", "mtp_depth_max": 1 }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "foo/Qwen3.6-MTPLX-NoSidecar")
        XCTAssertEqual(result.verdict, .alreadyMTPLX)
        XCTAssertEqual(result.sourceFormat, .mlxAffine,
                       "No mtp_sidecar key → can't claim mlxAffineWithMtp")
    }

    // MARK: - Forgeable verdicts (no mtplx_runtime.json present)

    func testForgeProbeCompressedTensorsAwqDetected() async {
        let fake = FakeRunner()
        // mtplx_runtime.json absent → fall through to config.json
        fake.install(
            url: "https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit/resolve/main/config.json",
            body: """
            {
              "architectures": ["Qwen3MoeForCausalLM"],
              "num_nextn_predict_layers": 1,
              "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized"
              }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit/tree/main",
            body: "[]"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.sourceFormat, .compressedTensorsAwq)
        XCTAssertTrue(result.message.contains("AWQ"))
    }

    func testForgeProbeAcceptsHuggingFaceModelURL() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4/resolve/main/config.json",
            body: """
            {
              "architectures": ["Qwen3MoeForCausalLM"],
              "num_nextn_predict_layers": 1,
              "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized"
              }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/cyankiwi/Qwen3.6-27B-AWQ-INT4/tree/main",
            body: "[]"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4")

        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.hfRepo, "cyankiwi/Qwen3.6-27B-AWQ-INT4")
        XCTAssertEqual(result.sourceFormat, .compressedTensorsAwq)
        XCTAssertNil(result.diagnostic)
    }

    func testForgeProbeBf16NativeDetectedWhenNoQuantBlock() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json",
            body: """
            {
              "architectures": ["Qwen3NextForCausalLM"],
              "num_nextn_predict_layers": 1
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/Qwen/Qwen3.6-27B/tree/main",
            body: "[]"
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "Qwen/Qwen3.6-27B")
        XCTAssertEqual(result.verdict, .forgeable)
        XCTAssertEqual(result.sourceFormat, .bf16Native)
        XCTAssertTrue(result.message.contains("BF16"))
    }

    func testForgeProbeMlxAffineWithMtpDetectedFromExtraTensors() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/mlx-community/Qwen3.6-27B-MTP/resolve/main/config.json",
            body: """
            {
              "num_nextn_predict_layers": 1,
              "mlx_lm_extra_tensors": { "mtp_file": "mtp.safetensors", "mtp_tensor_count": 29 }
            }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/mlx-community/Qwen3.6-27B-MTP/tree/main",
            body: """
            [ { "path": "mtp.safetensors" } ]
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "mlx-community/Qwen3.6-27B-MTP")
        XCTAssertEqual(result.sourceFormat, .mlxAffineWithMtp)
        XCTAssertTrue(result.hasMtpWeights)
    }

    func testForgeProbeNoMtpHeadsRefusedEvenWhenSourceIsClean() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/mlx-community/Llama-3-8B/resolve/main/config.json",
            body: """
            { "architectures": ["LlamaForCausalLM"] }
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "mlx-community/Llama-3-8B")
        XCTAssertEqual(result.verdict, .noMtpHeads)
        XCTAssertTrue(result.message.contains("Forge cannot synthesize"))
    }

    // MARK: - Error paths

    func testForgeProbeInvalidRepoIdShortCircuits() async {
        let probe = HuggingFaceProbe(runner: FakeRunner().runner())
        let result = await probe.forgeProbe(repo: "not-a-valid-repo")
        XCTAssertEqual(result.verdict, .probeFailed)
        XCTAssertEqual(result.diagnostic, "invalid_repo_id")
    }

    func testForgeProbe404OnConfigJsonReturnsProbeFailed() async {
        let fake = FakeRunner()
        // Nothing installed → all requests 404
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.forgeProbe(repo: "nonexistent/repo")
        XCTAssertEqual(result.verdict, .probeFailed)
    }

    // MARK: - classifySourceFormat unit (config-only, no IO)

    func testClassifySourceFormatPrefersCompressedTensorsOverMlxHints() {
        let config: [String: Any] = [
            "quantization_config": ["quant_method": "compressed-tensors"],
            "mlx_lm_extra_tensors": ["mtp_file": "mtp.safetensors"]
        ]
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: config, hasMTP: true),
            .compressedTensorsAwq,
            "compressed-tensors is the dominant signal — those repos can't be loaded as MLX even if they ship an mtp marker"
        )
    }

    func testClassifySourceFormatDetectsAwqMethod() {
        let config: [String: Any] = [
            "quantization_config": ["quant_method": "awq"]
        ]
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: config, hasMTP: true),
            .compressedTensorsAwq
        )
    }

    func testClassifySourceFormatFallsBackToBf16NativeWhenMtpPresent() {
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: [:], hasMTP: true),
            .bf16Native
        )
    }

    func testClassifySourceFormatFallsBackToHfVllmWhenNoMtp() {
        XCTAssertEqual(
            HuggingFaceProbe.classifySourceFormat(config: [:], hasMTP: false),
            .hfVllm
        )
    }

    // MARK: - Onboarding flow is untouched

    func testOnboardingProbeStillReturnsReadyForVanillaMtpRepo() async {
        // Sanity check that the extension didn't break the existing
        // onboarding contract.
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json",
            body: """
            { "num_nextn_predict_layers": 1 }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/Qwen/Qwen3.6-27B/tree/main",
            body: """
            [ { "path": "mtp.safetensors" } ]
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "Qwen/Qwen3.6-27B")
        XCTAssertEqual(result.verdict, .ready)
    }

    func testOnboardingProbeAlsoAcceptsHuggingFaceModelURL() async {
        let fake = FakeRunner()
        fake.install(
            url: "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/main/config.json",
            body: """
            { "num_nextn_predict_layers": 1 }
            """
        )
        fake.install(
            url: "https://huggingface.co/api/models/Qwen/Qwen3.6-27B/tree/main",
            body: """
            [ { "path": "mtp.safetensors" } ]
            """
        )
        let probe = HuggingFaceProbe(runner: fake.runner())
        let result = await probe.probe(repo: "https://huggingface.co/Qwen/Qwen3.6-27B/tree/main")
        XCTAssertEqual(result.verdict, .ready)
        XCTAssertEqual(result.hfRepo, "Qwen/Qwen3.6-27B")
    }
}
