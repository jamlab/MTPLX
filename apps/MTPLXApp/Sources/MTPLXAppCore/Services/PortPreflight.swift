import Foundation

// MARK: - PortPreflight
//
// Classifies who owns the configured daemon port BEFORE the app spawns a
// server. Without this, a foreign occupant (some other dev server on 8000)
// surfaced as a raw "Degraded: Port is already used…" failure after a full
// launch attempt. With it, the store can fall back to the next free port
// and tell the user in one humane sentence.
//
// SYNC PAIR: mtplx/daemon_client.py classify_port_occupant implements the
// same classification for the CLI. Update both sides together.

public enum PortOccupantKind: Equatable, Sendable {
    /// Nothing is listening; safe to bind.
    case free
    /// A healthy MTPLX daemon answered `/health`. The supervisor decides
    /// separately whether it is adoptable (app-owned, same model).
    case mtplxServer(HealthPayload)
    /// Something is listening but does not speak MTPLX health.
    case foreign
}

public enum PortPreflight {
    /// Classify the occupant of `baseURL`'s port with a short timeout.
    public static func classify(
        baseURL: URL,
        apiKey: String?,
        timeoutSeconds: TimeInterval = 2
    ) async -> PortOccupantKind {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeoutSeconds
        configuration.timeoutIntervalForResource = timeoutSeconds
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        let client = MTPLXAPIClient(baseURL: baseURL, apiKey: apiKey, session: session)
        do {
            let health = try await client.health()
            return health.ok ? .mtplxServer(health) : .foreign
        } catch let error as URLError {
            switch error.code {
            case .cannotConnectToHost, .cannotFindHost, .networkConnectionLost:
                return .free
            default:
                // Timeouts and protocol garbage both mean "occupied by
                // something we cannot use".
                return .foreign
            }
        } catch {
            // Decode failures / non-2xx statuses: a listener that is not an
            // MTPLX daemon.
            return .foreign
        }
    }

    /// First bindable loopback port strictly after `port`.
    public static func nextFreePort(
        after port: Int,
        attempts: Int = 50
    ) -> Int? {
        guard port < 65_535 else { return nil }
        let upperBound = min(port + max(1, attempts), 65_535)
        for candidate in (port + 1)...upperBound where portIsBindable(candidate) {
            return candidate
        }
        return nil
    }

    static func portIsBindable(_ port: Int) -> Bool {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        var reuse: Int32 = 1
        setsockopt(
            descriptor,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse,
            socklen_t(MemoryLayout<Int32>.size)
        )
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { rebound in
                bind(descriptor, rebound, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }
}
