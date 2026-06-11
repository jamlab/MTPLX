import Foundation
import SwiftData

// MARK: - MessageRole
//
// OpenAI/Anthropic-compatible role taxonomy as it actually appears on
// the daemon's `/v1/chat/completions` wire surface. Stored as a raw
// String so SwiftData schema stays primitive-only; the typed enum is a
// computed property on `ChatMessage`.

public enum MessageRole: String, Codable, Hashable, Sendable, CaseIterable {
    case user
    case assistant
    case tool
    /// System messages are not persisted today (the daemon owns the
    /// system prompt). Reserved here so the schema does not need a
    /// migration if a future feature surfaces them.
    case system
}

// MARK: - ChatConversation

/// One conversation == one SessionBank session on the daemon side.
/// `id` is sent as `X-MTPLX-Session-Id` on every request in this
/// conversation, so in-conversation prefix reuse stays warm in RAM.
@Model
public final class ChatConversation {
    /// Stable identity; persisted across app restarts and used as the
    /// session id for the daemon.
    @Attribute(.unique) public var id: UUID
    public var title: String
    public var createdAt: Date
    /// Bumped whenever a message is added; the sidebar sorts by this
    /// so the most-recent conversation floats to the top.
    public var updatedAt: Date
    /// Whether the web-search toggle is on for this conversation. Lives
    /// on the conversation (not globally) so a user can leave search on
    /// for a research thread and off for everyday chat without paging
    /// through Settings.
    public var webSearchEnabled: Bool

    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    public var messages: [ChatMessage]

    public init(
        id: UUID = UUID(),
        title: String = "New Chat",
        createdAt: Date = Date(),
        updatedAt: Date? = nil,
        webSearchEnabled: Bool = false,
        messages: [ChatMessage] = []
    ) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt ?? createdAt
        self.webSearchEnabled = webSearchEnabled
        self.messages = messages
    }
}

// MARK: - ChatMessage

/// One turn in the conversation. Stores enough to round-trip the
/// rendered UI exactly (visible text + reasoning + tool calls + the
/// attachments the user added). Tool calls and per-turn stats are
/// stored as JSON blobs to avoid SwiftData relationship complexity for
/// short-lived debug fields.
@Model
public final class ChatMessage {
    @Attribute(.unique) public var id: UUID
    /// Stored as raw String because SwiftData rejects enum types in
    /// some Xcode 26 builds; the typed accessor below is the API.
    public var roleRaw: String
    public var visibleContent: String
    public var reasoningContent: String?
    /// Filled when role == .tool so the daemon can match this tool
    /// result back to the assistant turn that requested it.
    public var toolCallId: String?
    /// Filled when role == .assistant and the model emitted tool calls.
    /// Encoded as `[ToolCallRecord]` JSON; decoded only when needed.
    public var toolCallsJSON: String?
    /// JSON-encoded `[ChatTurnStats]` for the assistant turn (decode
    /// TPS, accepted/drafted, verify time). Optional; assistant turns
    /// before the stats wiring stayed nil.
    public var statsJSON: String?
    /// Finish state for assistant turns. Completed turns use the server's
    /// finish reason (`stop`, `length`, etc.); interrupted local turns use
    /// app reasons such as `cancelled` or `error`.
    public var finishReason: String?
    public var createdAt: Date
    /// Denormalized conversation identity for resilient transcript fetches.
    /// SwiftData relationship queries can lag after rapid inserts/reloads;
    /// this keeps the selected chat from rendering blank while the
    /// relationship graph catches up.
    public var conversationID: UUID?
    public var conversation: ChatConversation?

    @Relationship(deleteRule: .cascade, inverse: \ChatAttachment.message)
    public var attachments: [ChatAttachment]

    @Relationship(deleteRule: .cascade, inverse: \ToolTraceRecord.message)
    public var toolTraces: [ToolTraceRecord]

    public var role: MessageRole {
        get { MessageRole(rawValue: roleRaw) ?? .user }
        set { roleRaw = newValue.rawValue }
    }

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        visibleContent: String,
        reasoningContent: String? = nil,
        toolCallId: String? = nil,
        toolCallsJSON: String? = nil,
        statsJSON: String? = nil,
        finishReason: String? = nil,
        createdAt: Date = Date(),
        conversation: ChatConversation? = nil,
        attachments: [ChatAttachment] = [],
        toolTraces: [ToolTraceRecord] = []
    ) {
        self.id = id
        self.roleRaw = role.rawValue
        self.visibleContent = visibleContent
        self.reasoningContent = reasoningContent
        self.toolCallId = toolCallId
        self.toolCallsJSON = toolCallsJSON
        self.statsJSON = statsJSON
        self.finishReason = finishReason
        self.createdAt = createdAt
        self.conversationID = conversation?.id
        self.conversation = conversation
        self.attachments = attachments
        self.toolTraces = toolTraces
    }
}

// MARK: - ChatAttachment

@Model
public final class ChatAttachment {
    @Attribute(.unique) public var id: UUID
    public var filename: String
    public var mimeType: String
    public var sizeBytes: Int
    /// Plain-text content extracted client-side by FileExtractor. Always
    /// present (failed extractions are not persisted — the user gets a
    /// red-dot composer chip and can send without the attachment).
    public var extractedText: String
    public var createdAt: Date
    public var message: ChatMessage?

    public init(
        id: UUID = UUID(),
        filename: String,
        mimeType: String,
        sizeBytes: Int,
        extractedText: String,
        createdAt: Date = Date(),
        message: ChatMessage? = nil
    ) {
        self.id = id
        self.filename = filename
        self.mimeType = mimeType
        self.sizeBytes = sizeBytes
        self.extractedText = extractedText
        self.createdAt = createdAt
        self.message = message
    }
}

// MARK: - ToolTraceRecord

/// One tool call's lifecycle, persisted alongside the assistant
/// message that triggered it. `name` is the OpenAI tool name (e.g.
/// `web_search`, `fetch_url`); arguments and result are JSON strings
/// so the trace surface can re-hydrate them on render without paying a
/// schema-migration cost when tool shapes change.
@Model
public final class ToolTraceRecord {
    @Attribute(.unique) public var id: UUID
    public var name: String
    public var statusRaw: String
    public var argumentsJSON: String?
    public var resultJSON: String?
    public var activityLog: [String]
    public var startedAt: Date
    public var completedAt: Date?
    public var message: ChatMessage?

    public var status: ToolTraceStatus {
        get { ToolTraceStatus(rawValue: statusRaw) ?? .pending }
        set { statusRaw = newValue.rawValue }
    }

    public init(
        id: UUID = UUID(),
        name: String,
        status: ToolTraceStatus = .pending,
        argumentsJSON: String? = nil,
        resultJSON: String? = nil,
        activityLog: [String] = [],
        startedAt: Date = Date(),
        completedAt: Date? = nil,
        message: ChatMessage? = nil
    ) {
        self.id = id
        self.name = name
        self.statusRaw = status.rawValue
        self.argumentsJSON = argumentsJSON
        self.resultJSON = resultJSON
        self.activityLog = activityLog
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.message = message
    }
}

public enum ToolTraceStatus: String, Codable, Sendable, CaseIterable {
    case pending
    case success
    case failed
}

// MARK: - Decoded helpers

/// Decoded shape of one tool call as it appears on the assistant turn's
/// `tool_calls` array. Persisted as JSON inside `ChatMessage.toolCallsJSON`
/// so the schema stays primitive.
public struct ToolCallRecord: Codable, Hashable, Sendable {
    public var id: String
    public var name: String
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// Decoded shape of the assistant turn's per-request stats. Persisted as
/// JSON inside `ChatMessage.statsJSON` so we don't need a schema bump
/// every time a new metric is exposed in `/v1/chat/completions`'s
/// `mtplx_stats` envelope.
public struct ChatTurnStats: Codable, Hashable, Sendable {
    public var rawDecodeTokS: Double?
    public var displayDecodeTokS: Double?
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var ttftS: Double?
    public var acceptedByDepth: [Int]?
    public var draftedByDepth: [Int]?
    public var verifyCalls: Int?
    public var verifyTimeS: Double?

    public init(
        rawDecodeTokS: Double? = nil,
        displayDecodeTokS: Double? = nil,
        promptTokens: Int? = nil,
        completionTokens: Int? = nil,
        ttftS: Double? = nil,
        acceptedByDepth: [Int]? = nil,
        draftedByDepth: [Int]? = nil,
        verifyCalls: Int? = nil,
        verifyTimeS: Double? = nil
    ) {
        self.rawDecodeTokS = rawDecodeTokS
        self.displayDecodeTokS = displayDecodeTokS
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.ttftS = ttftS
        self.acceptedByDepth = acceptedByDepth
        self.draftedByDepth = draftedByDepth
        self.verifyCalls = verifyCalls
        self.verifyTimeS = verifyTimeS
    }
}
