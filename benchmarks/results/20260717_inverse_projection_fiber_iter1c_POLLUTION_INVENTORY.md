# Iteration 1c development-receipt pollution inventory

Recorded: 2026-07-17. Evidence is retained under `/tmp`; it was not deleted, pooled, or used as
scientific output.

An exact search found 18 files containing the retired official namespace
`rtgs.inverse-projection-fiber.iter1c.v1`:

| SHA-256 | Retained path |
| --- | --- |
| `b2960041e551c08daf3f3ef584950cf62b81ef9037d593a325ccb0eff814edb6` | `/tmp/pytest-of-alex/pytest-1943/test_iter1c_invalid_publisher_0/artifacts/invalid-publication-fallback.json` |
| `502d647d6272143238df06eba642433ee543799635bc7e6ac676c9b4987ab90d` | `/tmp/pytest-of-alex/pytest-1943/test_iter1c_invalid_publisher_0/artifacts/lifecycle.json` |
| `24c2144361a9011769256a41da1b8999cfc1f7878b9ae4f2f15d28c802c81d20` | `/tmp/pytest-of-alex/pytest-1943/test_iter1c_invalid_receipt_is0/artifacts/lifecycle.json` |
| `fd53dd185c0e7467eb93a606b5da8470c3862c869ae6acf9ffc45a23d3a238f3` | `/tmp/pytest-of-alex/pytest-1943/test_iter1c_invalid_receipt_is0/artifacts/terminal.json` |
| `fd53dd185c0e7467eb93a606b5da8470c3862c869ae6acf9ffc45a23d3a238f3` | `/tmp/pytest-of-alex/pytest-1943/test_iter1c_invalid_receipt_is0/result.json` |
| `9221ed4c060c0b0bd461485022212dcb02861e15992bd52d51aebad5d6b8729c` | `/tmp/pytest-of-alex/pytest-1943/test_iter1c_preflight_rejects_1/verification.json` |
| `944255ba9e344b2448dc121c717658e626648b8fd72c63e06e42772fc74d0618` | `/tmp/pytest-of-alex/pytest-1944/test_iter1c_invalid_publisher_0/artifacts/invalid-publication-fallback.json` |
| `480619c0915d33f4f37afd3e9a735780f119e15817c2bf00ccd531ed8c108327` | `/tmp/pytest-of-alex/pytest-1944/test_iter1c_invalid_publisher_0/artifacts/lifecycle.json` |
| `ac47de0b31ace8b7b0bef9b07d0944457c036f97fd4a82bb188c83ed2fd45eae` | `/tmp/pytest-of-alex/pytest-1944/test_iter1c_invalid_receipt_is0/artifacts/lifecycle.json` |
| `6a706b977bcc196794194fdc99b9e2019246680985c7a7980fd4d157a30adeb1` | `/tmp/pytest-of-alex/pytest-1944/test_iter1c_invalid_receipt_is0/artifacts/terminal.json` |
| `6a706b977bcc196794194fdc99b9e2019246680985c7a7980fd4d157a30adeb1` | `/tmp/pytest-of-alex/pytest-1944/test_iter1c_invalid_receipt_is0/result.json` |
| `9221ed4c060c0b0bd461485022212dcb02861e15992bd52d51aebad5d6b8729c` | `/tmp/pytest-of-alex/pytest-1944/test_iter1c_preflight_rejects_1/verification.json` |
| `af7aad720e01234dd219ca5c9ad6f3da2e5c496b66ba4b31c33cd17cd16fb0ed` | `/tmp/pytest-of-alex/pytest-1945/test_iter1c_invalid_publisher_0/artifacts/invalid-publication-fallback.json` |
| `b29c97748cd6aeda8a56539fda0d4be4d46abeaa502c86ac7e41c2da6f5f79b6` | `/tmp/pytest-of-alex/pytest-1945/test_iter1c_invalid_publisher_0/artifacts/lifecycle.json` |
| `ac584612ea22813eb1413f779af74c801e2d4f4cca0964e5116ee0708c867b35` | `/tmp/pytest-of-alex/pytest-1945/test_iter1c_invalid_receipt_is0/artifacts/lifecycle.json` |
| `a69a8c4c969d90187c8d5c8ba8e9bd512f4b47d98b38b5e4198959f053b233b1` | `/tmp/pytest-of-alex/pytest-1945/test_iter1c_invalid_receipt_is0/artifacts/terminal.json` |
| `a69a8c4c969d90187c8d5c8ba8e9bd512f4b47d98b38b5e4198959f053b233b1` | `/tmp/pytest-of-alex/pytest-1945/test_iter1c_invalid_receipt_is0/result.json` |
| `9221ed4c060c0b0bd461485022212dcb02861e15992bd52d51aebad5d6b8729c` | `/tmp/pytest-of-alex/pytest-1945/test_iter1c_preflight_rejects_1/verification.json` |

The `invalid_receipt` and `invalid_publisher` files are the outcome-determining pollution: they
carry official terminal/lifecycle schemas and false official root-consumption status despite
development phases. The three `verification.json` files are deliberately malformed negative
fixtures but also contain the official namespace literal. All 18 paths are excluded from every
future source, input, result, metric, gate, and aggregate.
