# Model Compatibility

MTPLX separates detection from support.

| Tier | Meaning | Default behavior |
|---|---|---|
| Verified | `mtplx_runtime.json` exists and matches the expected contract | Run |
| Architecture-compatible, unverified | Qwen3-Next MTP markers exist, but no MTPLX contract | Refuse unless explicitly forced |
| Incompatible architecture | MTP markers exist for an unsupported architecture | Exit with roadmap pointer |
| No MTP | No MTP head detected | Exit with a clear message |

Only verified Qwen3-Next-MTP models are supported for v0.1.0-preview.1 product runs.
