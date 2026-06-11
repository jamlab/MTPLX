import Foundation

// MARK: - HuggingFaceProbe
//
// Pre-download verification for the `Other` model-pick path. Reads
// the public `config.json` first for the MTP arch fields, then checks
// the Hugging Face tree listing for the `mtp.safetensors` sidecar file
// when needed. The probe deliberately avoids a separate auth/existence
// preflight: public model downloads should not be blocked by a brittle
// metadata endpoint.
//
// Verdict rules mirror the daemon's `_classify_scanned_model` at
// `mtplx/ui/onboarding.py:314-496` but for a remote repository:
//   .ready          arch supports MTP AND mtp.safetensors is published
//   .missingSidecar arch supports MTP but no sidecar weights in tree
//   .noMTP          architecture does not declare MTP at all
//   .probeFailed    network / 404 / private-or-gated / malformed config

public struct HuggingFaceProbe: Sendable {
    /// Injectable for tests. Returns (httpStatusCode, body).
    public typealias HTTPRunner = @Sendable (URL, String) async throws -> (Int, Data)

    private let runner: HTTPRunner

    public init(runner: @escaping HTTPRunner = Self.defaultRunner) {
        self.runner = runner
    }

    // MARK: - Onboarding flow (unchanged)

    public func probe(repo rawRepo: String) async -> OtherModelProbe {
        guard let repo = Self.normalizedRepoID(rawRepo) else {
            return OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: rawRepo.trimmingCharacters(in: .whitespacesAndNewlines),
                message: "Paste a Hugging Face repo or link.",
                diagnostic: "invalid_repo_id"
            )
        }

        let configOutcome = await fetchConfig(repo: repo)
        let config: [String: Any]
        switch configOutcome {
        case .failed(let probe):
            return probe
        case .ok(let dict):
            config = dict
        }

        if !Self.configDeclaresMTP(config) {
            return OtherModelProbe(
                verdict: .noMTP,
                hfRepo: repo,
                message: "No MTP heads. You'll lose speculative decoding (~2-3× slower)."
            )
        }

        let hasSidecar = await checkSidecarPublished(repo: repo)
        if hasSidecar {
            return OtherModelProbe(
                verdict: .ready,
                hfRepo: repo,
                message: "MTP heads detected. Ready to download."
            )
        }
        return OtherModelProbe(
            verdict: .missingSidecar,
            hfRepo: repo,
            message: "Model declares MTP but mtp.safetensors isn't published. Speed will drop to standard decoding."
        )
    }

    // MARK: - Forge flow
    //
    // Returns a Forge-shaped verdict (`ForgeSourceProbe`) rather than
    // the onboarding-shaped `OtherModelProbe`. Forge has different
    // routing needs:
    //
    //   .alreadyMTPLX   repo already carries an mtplx_runtime.json,
    //                    so SourceStage swaps the primary CTA from
    //                    "Forge" to "Install instead"
    //   .forgeable      ready to forge, with the detected source
    //                    format threaded through so PlanStage can
    //                    auto-pick the right recipe
    //   .noMtpHeads     refuse — Forge cannot synthesize MTP heads
    //                    from nothing
    //   .probeFailed    network/404/private/malformed
    //
    // Implementation reuses the same fetchConfig +
    // checkSidecarPublished + configDeclaresMTP helpers as the
    // onboarding flow; the extra work is one HEAD-ish GET for the
    // mtplx_runtime.json marker and source-format classification
    // against `quantization_config` + `mlx_lm_extra_tensors` hints.
    // We deliberately do NOT pollute OtherModelProbe with Forge-
    // specific verdict cases — onboarding's switch statements stay
    // untouched and the two flows evolve independently.

    public func forgeProbe(repo rawRepo: String) async -> ForgeSourceProbe {
        guard let repo = Self.normalizedRepoID(rawRepo) else {
            return ForgeSourceProbe(
                verdict: .probeFailed,
                hfRepo: rawRepo.trimmingCharacters(in: .whitespacesAndNewlines),
                sourceFormat: .unknown,
                hasMtpWeights: false,
                message: "Paste a Hugging Face repo or link.",
                diagnostic: "invalid_repo_id"
            )
        }

        // Cheapest signal first: is this an already-MTPLX-branded
        // artifact? If yes, short-circuit before doing the heavier
        // config + tree fetches.
        if let runtimeJSON = await fetchMtplxRuntimeJSON(repo: repo) {
            let format: ForgeSourceFormat = runtimeJSON["mtp_sidecar"] != nil
                ? .mlxAffineWithMtp
                : .mlxAffine
            let depth = (runtimeJSON["mtp_depth_max"] as? Int) ?? 1
            return ForgeSourceProbe(
                verdict: .alreadyMTPLX,
                hfRepo: repo,
                sourceFormat: format,
                hasMtpWeights: true,
                message: "Already MTPLX-branded — depth \(depth) verified. Install instead of rebuilding.",
                diagnostic: nil
            )
        }

        let configOutcome = await fetchConfig(repo: repo)
        let config: [String: Any]
        switch configOutcome {
        case .failed(let probe):
            return ForgeSourceProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                sourceFormat: .unknown,
                hasMtpWeights: false,
                message: probe.message,
                diagnostic: probe.diagnostic
            )
        case .ok(let dict):
            config = dict
        }

        let hasMTP = Self.configDeclaresMTP(config)
        if !hasMTP {
            return ForgeSourceProbe(
                verdict: .noMtpHeads,
                hfRepo: repo,
                sourceFormat: Self.classifySourceFormat(config: config, hasMTP: false),
                hasMtpWeights: false,
                message: "Architecture has no MTP heads. Forge cannot synthesize them — pick a model with `mtp_num_hidden_layers > 0` in config.json."
            )
        }

        let hasSidecar = await checkSidecarPublished(repo: repo)
        let format = Self.classifySourceFormat(config: config, hasMTP: true)
        let hasMtpWeights = hasSidecar || format == .mlxAffineWithMtp
        return ForgeSourceProbe(
            verdict: .forgeable,
            hfRepo: repo,
            sourceFormat: format,
            hasMtpWeights: hasMtpWeights,
            message: Self.forgeableMessage(format: format, hasSidecar: hasSidecar)
        )
    }

    private func fetchMtplxRuntimeJSON(repo: String) async -> [String: Any]? {
        guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/mtplx_runtime.json") else {
            return nil
        }
        do {
            let (status, body) = try await runner(url, "GET")
            guard status == 200 else { return nil }
            return try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        } catch {
            return nil
        }
    }

    /// Classify the source artifact's quantization layout from the
    /// `config.json` we already fetched. Detection priority:
    ///
    /// 1. `quantization_config.quant_method == "compressed-tensors"`
    ///    or weight format hints → compressed-tensors AWQ (vLLM/SGLang
    ///    target; cyankiwi pattern). Needs the AWQ → MLX-affine path.
    /// 2. `mlx_lm_extra_tensors.mtp_file` present → already an MLX
    ///    artifact with packed MTP sidecar. Verify-only.
    /// 3. `quantization` block with MLX-style fields → mlxAffine.
    /// 4. Default: BF16 native if MTP, otherwise `.hfVllm`.
    ///
    /// `unknown` is reserved for genuinely uncategorisable cases —
    /// PlanStage warns the user and defaults to the BF16-native
    /// recipe.
    static func classifySourceFormat(config: [String: Any], hasMTP: Bool) -> ForgeSourceFormat {
        if let quantConfig = config["quantization_config"] as? [String: Any] {
            let method = (quantConfig["quant_method"] as? String)?.lowercased() ?? ""
            let format = (quantConfig["format"] as? String)?.lowercased() ?? ""
            let isCompressedTensors = method == "compressed-tensors"
                || method == "compressed_tensors"
                || format.contains("compressed-tensors")
                || format.contains("compressed_tensors")
            let isAWQ = method == "awq" || method.contains("awq")
            if isCompressedTensors || isAWQ {
                return .compressedTensorsAwq
            }
        }
        if let extras = config["mlx_lm_extra_tensors"] as? [String: Any],
           extras["mtp_file"] != nil {
            return .mlxAffineWithMtp
        }
        if config["quantization"] is [String: Any] {
            return .mlxAffine
        }
        return hasMTP ? .bf16Native : .hfVllm
    }

    private static func forgeableMessage(format: ForgeSourceFormat, hasSidecar: Bool) -> String {
        switch format {
        case .compressedTensorsAwq:
            return "AWQ source detected (vLLM/SGLang format). Forge will convert the body to MLX-affine and extract the MTP sidecar."
        case .mlxAffineWithMtp:
            return "MLX-affine artifact with packed MTP sidecar. Forge will requantize the body and re-pack."
        case .mlxAffine:
            return "MLX-affine source. Forge will package the MTP sidecar from " + (hasSidecar ? "mtp.safetensors." : "the main shards.")
        case .bf16Native:
            return "BF16 source with MTP heads. Forge will quantize the body and keep MTP weights at BF16 (safest)."
        case .hfVllm:
            return "Hugging Face source. Forge will convert to MLX and pack the MTP sidecar."
        case .unknown:
            return "Source detected but format is unfamiliar. Forge will attempt a BF16-native recipe; review the Plan step carefully."
        }
    }

    private enum ConfigOutcome {
        case ok([String: Any])
        case failed(OtherModelProbe)
    }

    // MARK: - Step 1: GET https://huggingface.co/<repo>/resolve/main/config.json

    private func fetchConfig(repo: String) async -> ConfigOutcome {
        guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/config.json") else {
            return .failed(OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "Could not build URL for config.json.",
                diagnostic: "url_build_failed"
            ))
        }
        do {
            let (status, body) = try await runner(url, "GET")
            switch status {
            case 200:
                break
            case 401, 403:
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "This repo is private or gated. Public Hugging Face models download without a login.",
                    diagnostic: "http_\(status)"
                ))
            case 404:
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "Repository or config.json not found on huggingface.co.",
                    diagnostic: "http_404"
                ))
            default:
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "config.json is unavailable (HTTP \(status)).",
                    diagnostic: "http_\(status)"
                ))
            }
            guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
                return .failed(OtherModelProbe(
                    verdict: .probeFailed,
                    hfRepo: repo,
                    message: "config.json was malformed.",
                    diagnostic: "config_decode_failed"
                ))
            }
            return .ok(json)
        } catch {
            return .failed(OtherModelProbe(
                verdict: .probeFailed,
                hfRepo: repo,
                message: "Couldn't fetch config.json.",
                diagnostic: error.localizedDescription
            ))
        }
    }

    // MARK: - Step 2: GET https://huggingface.co/api/models/<repo>/tree/main

    private func checkSidecarPublished(repo: String) async -> Bool {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main") else {
            return false
        }
        guard let (status, body) = try? await runner(url, "GET"), status == 200 else {
            // On any tree-listing failure we conservatively assume the
            // sidecar is missing — the user-visible verdict drops to
            // amber `.missingSidecar` rather than crashing the probe.
            return false
        }
        guard let entries = try? JSONSerialization.jsonObject(with: body) as? [[String: Any]] else {
            return false
        }
        for entry in entries {
            let path = (entry["path"] as? String)?.lowercased() ?? ""
            if path == "mtp.safetensors"
                || path == "model-mtp.safetensors"
                || path == "mtp/weights.safetensors"
                || path.hasSuffix("/mtp.safetensors")
                || (path.contains("/mtp/") && path.hasSuffix(".safetensors"))
            {
                return true
            }
        }
        return false
    }

    // MARK: - Architecture detection (config-only, no weight inspection)

    static func configDeclaresMTP(_ config: [String: Any]) -> Bool {
        // Three indicator fields the daemon recognises, in priority
        // order. Each is an Int > 0 on MTP-capable model configs.
        if let n = configInt(config, ["text_config", "mtp_num_hidden_layers"]), n > 0 {
            return true
        }
        if let n = configInt(config, ["num_nextn_predict_layers"]), n > 0 {
            return true
        }
        if let n = configInt(config, ["num_mtp_modules"]), n > 0 {
            return true
        }
        return false
    }

    /// Walks a key path inside a JSON dict and coerces the leaf to Int.
    /// Accepts both `Int` and `Double` leaf values because HF configs
    /// sometimes serialise integers as floats.
    private static func configInt(_ config: [String: Any], _ keys: [String]) -> Int? {
        var cursor: Any = config
        for key in keys {
            guard let dict = cursor as? [String: Any], let next = dict[key] else {
                return nil
            }
            cursor = next
        }
        if let int = cursor as? Int { return int }
        if let double = cursor as? Double { return Int(double) }
        return nil
    }

    // MARK: - Repo id sanity

    static func normalizedRepoID(_ rawValue: String) -> String? {
        MTPLXModelOption.normalizedHuggingFaceRepoID(rawValue)
    }

    static func looksLikeRepoID(_ s: String) -> Bool {
        normalizedRepoID(s) != nil
    }

    // MARK: - Default URLSession runner

    @Sendable
    public static func defaultRunner(_ url: URL, _ method: String) async throws -> (Int, Data) {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 12
        // Hugging Face honors a User-Agent and uses it to gate scraping
        // heuristics. Identify ourselves clearly so a probe failure is
        // traceable in HF logs back to MTPLX.
        request.setValue("MTPLXApp/1 OnboardingProbe", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (status, data)
    }
}
