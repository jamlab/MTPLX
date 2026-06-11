import Combine
import Foundation
import SwiftData

// MARK: - StreamingPhase

/// What the assistant is doing right now. Drives the inline activity caption
/// before visible answer tokens, tool traces, or reasoning tokens arrive.
/// above the streaming bubble.
public enum StreamingPhase: String, Sendable, Equatable {
    case idle
    case thinking
    case generating
    case searching
    case reading
    case answering
    case finalizing
}

// MARK: - ChatError

public enum ChatError: LocalizedError, Equatable {
    case streamLost
    case unauthorized
    case http(Int, String)
    case malformedRequest
    case daemonStopped
    case unknown(String)

    public var errorDescription: String? {
        switch self {
        case .streamLost: return "Connection dropped mid-reply. Try again."
        case .unauthorized: return "The model rejected the request. Set an API key in Settings."
        case .http(let code, let body):
            let truncated = body.prefix(160)
            return "HTTP \(code): \(truncated)"
        case .malformedRequest: return "Couldn't send the message."
        case .daemonStopped: return "MTPLX isn't running. Hit the play button to start a model."
        case .unknown(let detail): return detail
        }
    }
}

// MARK: - Pending tool trace
//
// Lightweight in-flight trace that drives the live `AssistantTraceSurface`
// before the tool call completes and we persist a `ToolTraceRecord`.

public struct PendingToolTrace: Identifiable, Equatable, Sendable {
    public let id: String
    public var name: String
    public var subtitle: String
    public var detail: String
    public var activityLog: [String]
    public var status: ToolTraceStatus

    public init(
        id: String,
        name: String,
        subtitle: String = "",
        detail: String = "",
        activityLog: [String] = [],
        status: ToolTraceStatus = .pending
    ) {
        self.id = id
        self.name = name
        self.subtitle = subtitle
        self.detail = detail
        self.activityLog = activityLog
        self.status = status
    }
}

// MARK: - ChatViewModel
//
// One per app session. Owns the chat surface's published state plus the
// multi-round tool loop that drives a single user turn:
//   1. Persist the user message (with attachment-extracted text inlined
//      as a fenced block) + bump conversation.updatedAt.
//   2. Build a `ChatRequest` from `visibleMessages`. Include
//      `factory.toolDefinitions()` if `webSearchEnabled` is on.
//   3. Stream tokens via `MTPLXChatClient.stream(...)` and fold events
//      into published state.
//   4. On `finished` with `finishReason == "tool_calls"`, dispatch each
//      tool call through the factory and append `role: "tool"` messages,
//      then loop back to (2). The default product path allows one tool
//      round, then forces `tool_choice: "none"` so web chat answers from
//      the sources it already gathered instead of over-searching.
//   5. On any other finish_reason (`stop`, `length`), persist the
//      assistant turn + stats and clear streaming state.

@MainActor
public final class ChatViewModel: ObservableObject {
    // Published UI state
    @Published public private(set) var conversations: [ChatConversation] = []
    @Published public private(set) var current: ChatConversation?
    @Published public private(set) var visibleMessages: [ChatMessage] = []
    @Published public private(set) var isStreaming: Bool = false
    @Published public private(set) var streamingPhase: StreamingPhase = .idle
    public let streamingReasoningDocument = StreamingDocumentStore(mode: .plainLines)
    public let streamingContentDocument = StreamingDocumentStore(mode: .plainLines)
    @Published public private(set) var hasStreamingReasoning: Bool = false
    @Published public private(set) var hasStreamingContent: Bool = false
    @Published public private(set) var handoffAssistantMessageID: UUID?
    public var streamingReasoning: String { streamingReasoningDocument.rawText + streamingReasoningBuffer }
    public var streamingContent: String { streamingContentDocument.rawText + streamingContentBuffer }
    public var shouldRenderStreamingAssistant: Bool {
        guard isStreaming else { return false }
        guard let handoffAssistantMessageID else { return true }
        return !visibleMessages.contains { $0.id == handoffAssistantMessageID }
    }
    @Published public private(set) var pendingToolTraces: [PendingToolTrace] = []
    @Published public private(set) var chatDecodeReading: HeadlineDecodeReading = .absent
    @Published public var pendingAttachments: [ChatAttachment] = []
    @Published public var lastError: ChatError?

    // Public knobs
    public var webSearchEnabled: Bool {
        get { current?.webSearchEnabled ?? false }
        set {
            guard let current else { return }
            current.webSearchEnabled = newValue
            saveContext()
            objectWillChange.send()
        }
    }

    // Internals
    private let container: ModelContainer
    private let chatClientProvider: @MainActor () -> MTPLXChatClient
    private let toolFactory: MTPLXChatToolFactory
    private let modelName: () -> String?
    private let reasoningEnabledProvider: @MainActor () -> Bool?
    private let onDaemonUnreachable: @MainActor () -> Void
    private let maxToolRounds: Int

    private var context: ModelContext { container.mainContext }
    private var streamTask: Task<Void, Never>?
    private var currentRequestId: String?
    /// Monotonic turn token. Bumped when a turn starts and again on
    /// cancel, so a cancelled stream task that is still draining can be
    /// recognized as superseded and ignored — it must not fold tokens
    /// into, or persist a turn over, the next message.
    private var streamGeneration: Int = 0
    /// Per-conversation server session id override. Normally the session
    /// id is the stable `conversation.id` (so SessionBank warm-prefix
    /// reuse works across turns); after a cancel we rotate it to a fresh
    /// UUID so the daemon can't resume the cancelled prompt's committed
    /// prefix into the next turn.
    private var sessionOverrides: [UUID: UUID] = [:]
    /// Per-round accumulator. Lives on the viewmodel (which is
    /// @MainActor) rather than as a captured local so the SSE event
    /// closure can mutate it without crossing a Sendable boundary.
    private var roundToolCalls: [Int: AccumulatingToolCall] = [:]
    private var roundFinishReason: String = "stop"
    private var roundUsage: ChatUsage?
    private var roundStats: ChatStreamStats?
    private var turnStartedAt: Date?
    private var reasoningStartedAt: Date?
    private var streamingReasoningBuffer = ""
    private var streamingContentBuffer = ""
    private var streamFlushTask: Task<Void, Never>?
    private var lastLiveDecodeUpdateAt: Date = .distantPast
    // Paint token-sized SSE deltas near display refresh. Live chat stays plain
    // text, so this can feel token-by-token without invoking markdown/layout
    // work for every raw network event.
    private static let streamFlushInterval: Duration = .milliseconds(16)
    private static let liveDecodeUpdateInterval: TimeInterval = 0.20
    private static let requestContextCharacterBudget = 64_000
    private static let requestRecentVerbatimMessageCount = 8
    private static let requestHistoricalContentLimit = 1_600
    static let requestToolResultContentLimit = 20_000
    private static let requestToolResultMaxResults = 5
    static let requestToolResultExcerptLimit = 2_400
    private var leakedThinkingSplitter = ChatThinkingTagSplitter()

    public init(
        container: ModelContainer,
        chatClientProvider: @escaping @MainActor () -> MTPLXChatClient,
        toolFactory: MTPLXChatToolFactory = MTPLXChatToolFactory(),
        modelName: @escaping () -> String? = { nil },
        reasoningEnabledProvider: @escaping @MainActor () -> Bool? = { nil },
        onDaemonUnreachable: @escaping @MainActor () -> Void = {},
        maxToolRounds: Int = 1
    ) {
        self.container = container
        self.chatClientProvider = chatClientProvider
        self.toolFactory = toolFactory
        self.modelName = modelName
        self.reasoningEnabledProvider = reasoningEnabledProvider
        self.onDaemonUnreachable = onDaemonUnreachable
        self.maxToolRounds = maxToolRounds
        refreshConversations()
        if let first = conversations.first {
            select(first)
        }
    }

    // MARK: - Conversation lifecycle

    public func refreshConversations() {
        let descriptor = FetchDescriptor<ChatConversation>(
            sortBy: [SortDescriptor(\.updatedAt, order: .reverse)]
        )
        conversations = (try? context.fetch(descriptor)) ?? []
    }

    @discardableResult
    public func createNewConversation() -> ChatConversation {
        let convo = ChatConversation(title: "New Chat")
        context.insert(convo)
        saveContext()
        refreshConversations()
        select(convo)
        return convo
    }

    public func select(_ conversation: ChatConversation) {
        current = conversation
        visibleMessages = loadMessages(for: conversation)
        clearStreamingState()
    }

    public func delete(_ conversation: ChatConversation) async {
        if current?.id == conversation.id {
            await cancel()
            clearStreamingState()
            current = nil
            visibleMessages = []
        }
        context.delete(conversation)
        saveContext()
        refreshConversations()
        if current == nil, let next = conversations.first {
            select(next)
        }
    }

    // MARK: - Attachments

    public func attach(_ urls: [URL]) async {
        var added: [ChatAttachment] = []
        for url in urls {
            do {
                let extracted = try FileExtractor.extract(from: url)
                let attachment = ChatAttachment(
                    filename: extracted.filename,
                    mimeType: extracted.mimeType,
                    sizeBytes: extracted.sizeBytes,
                    extractedText: extracted.combinedText
                )
                added.append(attachment)
            } catch let error as FileExtractorError {
                let placeholder = ChatAttachment(
                    filename: url.lastPathComponent,
                    mimeType: FileExtractor.mimeType(for: url.pathExtension),
                    sizeBytes: 0,
                    extractedText: ""
                )
                lastError = .unknown(error.localizedDescription)
                added.append(placeholder)
            } catch {
                lastError = .unknown(error.localizedDescription)
            }
        }
        pendingAttachments.append(contentsOf: added)
    }

    public var hasSendablePendingAttachments: Bool {
        pendingAttachments.contains(where: Self.isSendableAttachment)
    }

    public var canRetryLastUserMessage: Bool {
        guard lastError != nil, !isStreaming else { return false }
        return visibleMessages.last(where: { $0.role == .user }) != nil
    }

    public func removeAttachment(_ attachment: ChatAttachment) {
        pendingAttachments.removeAll { $0.id == attachment.id }
    }

    // MARK: - Send / cancel

    public func send(_ rawText: String) {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || hasSendablePendingAttachments else { return }
        guard !isStreaming else { return }

        let conversation = current ?? createNewConversation()
        let attachments = pendingAttachments.filter(Self.isSendableAttachment)
        pendingAttachments.removeAll(where: Self.isSendableAttachment)

        let fencedAttachmentText = Self.buildAttachmentContext(attachments: attachments)
        let visibleUserContent = text
        let fullUserContent: String =
            attachments.isEmpty
            ? visibleUserContent
            : (visibleUserContent.isEmpty
                ? fencedAttachmentText
                : "\(visibleUserContent)\n\n\(fencedAttachmentText)")

        let userMessage = ChatMessage(
            role: .user,
            visibleContent: visibleUserContent,
            createdAt: Date(),
            conversation: conversation,
            attachments: attachments
        )
        for attachment in attachments {
            attachment.message = userMessage
        }
        context.insert(userMessage)
        conversation.messages.append(userMessage)
        conversation.updatedAt = userMessage.createdAt
        if conversation.title == "New Chat", !visibleUserContent.isEmpty {
            conversation.title = Self.firstNWords(visibleUserContent, n: 5)
        }
        saveContext()
        publishVisibleMessages(for: conversation, ensuring: userMessage)
        refreshConversations()

        startStream(fullUserContent: fullUserContent, conversation: conversation)
    }

    public func retryLastUserMessage() {
        guard !isStreaming, let conversation = current else { return }
        let messages = loadMessages(for: conversation)
        guard let userMessage = messages.last(where: { $0.role == .user }) else { return }
        let fullUserContent = Self.fullUserContent(for: userMessage)
        let retryMessages = Self.buildRetryRequestMessages(
            from: messages,
            retrying: userMessage,
            fullUserContent: fullUserContent
        )
        guard !retryMessages.isEmpty else { return }
        startStream(
            fullUserContent: fullUserContent,
            conversation: conversation,
            requestMessages: retryMessages
        )
    }

    public func cancel() async {
        guard isStreaming else { return }
        flushStreamingBuffers()
        stopStreamFlushLoop()
        // Invalidate the in-flight task's writes BEFORE awaiting it, so any
        // late SSE events it emits while tearing down are recognized as
        // superseded (generation mismatch) and dropped instead of bleeding
        // into the next turn.
        streamGeneration &+= 1
        let task = streamTask
        streamTask = nil
        task?.cancel()
        if let requestId = currentRequestId {
            await chatClientProvider().cancel(requestId: requestId)
        }
        // Wait for the stream task to actually stop before resetting state,
        // so a new send() can't race a still-draining cancelled task.
        await task?.value
        // Rotate the server session so the cancelled prompt's committed
        // prefix can't be resumed into the next message.
        if let conversation = current {
            sessionOverrides[conversation.id] = UUID()
        }
        finalizePartialAssistantTurn(reason: "cancelled")
    }

    /// Server session id for a conversation. Stable (== conversation.id)
    /// across normal turns so warm-prefix reuse works; rotated after a
    /// cancel so the daemon starts a clean session for the next turn.
    private func liveSessionId(for conversation: ChatConversation) -> UUID {
        sessionOverrides[conversation.id] ?? conversation.id
    }

    // MARK: - Streaming

    private func startStream(
        fullUserContent: String,
        conversation: ChatConversation,
        requestMessages: [ChatRequestMessage]? = nil
    ) {
        streamGeneration &+= 1
        let generation = streamGeneration
        isStreaming = true
        streamingPhase = reasoningEnabledProvider() == false ? .generating : .thinking
        streamingReasoningDocument.reset()
        streamingContentDocument.reset()
        hasStreamingReasoning = false
        hasStreamingContent = false
        handoffAssistantMessageID = nil
        pendingToolTraces = []
        chatDecodeReading = .absent
        roundToolCalls = [:]
        turnStartedAt = Date()
        reasoningStartedAt = nil
        currentRequestId = nil
        lastError = nil
        streamingReasoningBuffer = ""
        streamingContentBuffer = ""
        leakedThinkingSplitter.reset()
        lastLiveDecodeUpdateAt = .distantPast
        startStreamFlushLoop(generation: generation)

        // Take a snapshot of the request shape so the loop is reentrant.
        let initialMessages = requestMessages ?? Self.buildRequestMessages(
            from: visibleMessages,
            overrideLastUserContent: fullUserContent
        )
        let sessionId = liveSessionId(for: conversation)
        let useTools = conversation.webSearchEnabled
        let tools = useTools ? toolFactory.toolDefinitions() : nil
        let toolChoice: String? = useTools ? "auto" : nil
        let model = modelName()

        let client = chatClientProvider()
        streamTask = Task { [weak self] in
            guard let self else { return }
            await self.toolFactory.beginTurn()
            await self.runToolLoop(
                generation: generation,
                client: client,
                conversation: conversation,
                sessionId: sessionId,
                messages: initialMessages,
                model: model,
                tools: tools,
                toolChoice: toolChoice
            )
        }
    }

    private func runToolLoop(
        generation: Int,
        client: MTPLXChatClient,
        conversation: ChatConversation,
        sessionId: UUID,
        messages initial: [ChatRequestMessage],
        model: String?,
        tools: [ChatRequestTool]?,
        toolChoice initialToolChoice: String?
    ) async {
        var messages = initial
        var toolChoice = initialToolChoice
        var round = 0

        loop: while !Task.isCancelled {
            round += 1
            let request = ChatRequest(
                model: model,
                messages: messages,
                stream: true,
                tools: tools,
                toolChoice: toolChoice
            )
            roundToolCalls.removeAll(keepingCapacity: true)
            roundFinishReason = "stop"
            roundUsage = nil
            roundStats = nil
            var streamError: Error?
            do {
                try await client.stream(
                    request: request,
                    sessionId: sessionId
                ) { [weak self] event in
                    await self?.handleEvent(event, generation: generation)
                }
            } catch is CancellationError {
                // User Stop: cancel() owns teardown/finalization. Do not
                // persist a turn or report an error.
                return
            } catch let error as MTPLXChatClientError {
                streamError = error
            } catch {
                // Transport-level cancellation (URLError.cancelled) arrives
                // here, not as CancellationError; recognize it via the
                // generation bump that cancel() performs.
                if generation != streamGeneration { return }
                streamError = error
            }

            // Superseded by a cancel() (which bumps the generation and owns
            // finalization) — don't fall through to persistence. Keyed on
            // the generation token, not Task.isCancelled, because the
            // latter can read true transiently and would wrongly drop a
            // normal finish.
            if generation != streamGeneration {
                return
            }

            flushLeakedThinkingSplitter()
            flushStreamingBuffers()

            if let streamError {
                handleStreamError(streamError, conversation: conversation)
                return
            }

            let accumulatedToolCalls = roundToolCalls
            let finishReason = roundFinishReason
            let finalUsage = roundUsage
            let finalStats = roundStats

            if finishReason == "tool_calls", round <= maxToolRounds {
                // Persist the assistant turn that requested the tool
                // calls, then dispatch each call and append role:"tool"
                // responses, then continue the loop.
                let assistantMessage = persistAssistantTurn(
                    conversation: conversation,
                    finishReason: finishReason,
                    usage: finalUsage,
                    stats: finalStats,
                    toolCalls: Array(accumulatedToolCalls.values),
                    traces: pendingToolTraces
                )
                messages.append(
                    Self.assistantRequestMessage(from: assistantMessage)
                )
                pendingToolTraces.removeAll()

                for call in accumulatedToolCalls.values {
                    if Task.isCancelled { break }
                    let traceId = call.id
                    pendingToolTraces.append(
                        PendingToolTrace(
                            id: traceId,
                            name: call.name,
                            subtitle: Self.shortArgsSubtitle(for: call),
                            detail: Self.liveDetail(for: call.name),
                            activityLog: [],
                            status: .pending
                        )
                    )
                    streamingPhase = Self.streamingPhase(forTool: call.name)
                    let result = await toolFactory.dispatch(
                        name: call.name,
                        argumentsJSON: call.arguments
                    )
                    updatePendingTrace(id: traceId) { trace in
                        trace.status = .success
                        trace.detail = Self.shortResultDetail(for: call.name, json: result)
                    }
                    persistToolTrace(
                        on: assistantMessage,
                        id: call.id,
                        name: call.name,
                        argumentsJSON: call.arguments,
                        resultJSON: result,
                        status: .success
                    )
                    let requestResult = Self.compactToolResultContent(result)
                    messages.append(
                        ChatRequestMessage(
                            role: "tool",
                            content: requestResult,
                            toolCallId: call.id
                        )
                    )
                    let toolStorageMessage = ChatMessage(
                        role: .tool,
                        visibleContent: result,
                        toolCallId: call.id,
                        createdAt: Date(),
                        conversation: conversation
                    )
                    context.insert(toolStorageMessage)
                    conversation.messages.append(toolStorageMessage)
                }
                saveContext()
                refreshVisibleMessages()

                streamingContentDocument.reset()
                streamingReasoningDocument.reset()
                hasStreamingContent = false
                hasStreamingReasoning = false
                streamingPhase = .answering
                if round == maxToolRounds {
                    // Final pass: stop the model from issuing more tool
                    // calls so the user always gets a concrete answer.
                    toolChoice = "none"
                }
                continue loop
            }

            // Plain finish (stop / length / unknown). Persist and stop.
            let assistantMessage = persistAssistantTurn(
                conversation: conversation,
                finishReason: finishReason,
                usage: finalUsage,
                stats: finalStats,
                toolCalls: Array(accumulatedToolCalls.values),
                traces: pendingToolTraces,
                publishImmediately: false
            )
            updateChatDecodeReading(from: finalStats)
            publishVisibleMessages(for: conversation, ensuring: assistantMessage)
            refreshConversations()
            handoffAssistantMessageID = assistantMessage.id
            finalizeAssistantTurnUI()
            return
        }
        // Reached only if the task was cancelled between rounds. If cancel()
        // bumped the generation it owns finalization; otherwise (e.g. the
        // task was cancelled by teardown) finalize the partial turn here.
        if Task.isCancelled, generation == streamGeneration {
            finalizePartialAssistantTurn(reason: "cancelled")
        }
    }

    // MARK: - Event folding

    private func handleEvent(_ event: ChatStreamEvent, generation: Int) async {
        // Drop events from a superseded (cancelled / replaced) turn so a
        // still-draining task can't fold tokens into the next message.
        guard generation == streamGeneration else { return }
        switch event {
        case .requestId(let id):
            currentRequestId = id
        case .role:
            break
        case .reasoningDelta(let fragment):
            appendStreamingReasoning(fragment)
        case .contentDelta(let fragment):
            let split = leakedThinkingSplitter.feed(fragment)
            appendStreamingReasoning(split.reasoning)
            appendStreamingContent(split.content)
        case .toolCallStart(let index, let id, let name):
            roundToolCalls[index] = AccumulatingToolCall(id: id, name: name, arguments: "")
        case .toolCallArgumentsDelta(let index, let fragment):
            roundToolCalls[index, default: AccumulatingToolCall(
                id: "call_\(index)", name: "", arguments: ""
            )].arguments.append(fragment)
        case .progress(let frame):
            updateChatDecodeReading(from: frame)
        case .finished(let reason, let usage, let stats):
            roundFinishReason = reason
            roundUsage = usage
            roundStats = stats
        }
    }

    private func appendStreamingReasoning(_ fragment: String) {
        guard !fragment.isEmpty else { return }
        let wasEmpty = streamingReasoning.isEmpty
        if reasoningStartedAt == nil {
            reasoningStartedAt = Date()
        }
        streamingReasoningBuffer.append(fragment)
        if wasEmpty {
            hasStreamingReasoning = true
            flushStreamingBuffers()
        }
        if streamingContent.isEmpty, streamingPhase != .thinking {
            streamingPhase = .thinking
        }
    }

    private func appendStreamingContent(_ fragment: String) {
        guard !fragment.isEmpty else { return }
        let wasEmpty = streamingContent.isEmpty
        streamingContentBuffer.append(fragment)
        if wasEmpty {
            hasStreamingContent = true
        }
        if streamingPhase != .answering {
            streamingPhase = .answering
        }
        if wasEmpty {
            flushStreamingBuffers()
        }
    }

    private func updateChatDecodeReading(from frame: ChatProgressFrame) {
        guard let value = Self.chatDecodeTokS(from: frame) else { return }
        let now = Date()
        guard chatDecodeReading == .absent
            || now.timeIntervalSince(lastLiveDecodeUpdateAt) >= Self.liveDecodeUpdateInterval
        else { return }
        lastLiveDecodeUpdateAt = now
        chatDecodeReading = .live(value)
    }

    private func updateChatDecodeReading(from stats: ChatStreamStats?) {
        if let value = Self.chatDecodeTokS(from: stats) {
            chatDecodeReading = .held(value: value, completedAt: Date())
        } else if case .live(let value) = chatDecodeReading {
            chatDecodeReading = .held(value: value, completedAt: Date())
        }
    }

    private static func chatDecodeTokS(from frame: ChatProgressFrame) -> Double? {
        // Open WebUI-style cumulative decode rate: generated tokens /
        // decode time. A single converging value, never the raw-vs-window
        // flip that made the chip strobe between two integers.
        let tokens = frame.completionTokens.map(Double.init)
            ?? frame.raw.values["completion_tokens"]?.doubleValue
        if let tokens, tokens > 0,
           let decodeElapsed = frame.raw.values["decode_elapsed_s"]?.doubleValue,
           decodeElapsed > 0.05 {
            let tps = tokens / decodeElapsed
            if tps.isFinite, tps > 0 { return tps }
        }
        // Early-frame fallback: raw rate only (never the sliding window).
        return firstPositiveFinite(frame.decodeTokS)
    }

    private static func chatDecodeTokS(from stats: ChatStreamStats?) -> Double? {
        guard let stats else { return nil }
        // Completed chat TPS is the full-request cumulative average.
        if let tokens = stats.raw.values["completion_tokens"]?.doubleValue, tokens > 0,
           let decodeElapsed = stats.raw.values["decode_elapsed_s"]?.doubleValue,
           decodeElapsed > 0.05 {
            let tps = tokens / decodeElapsed
            if tps.isFinite, tps > 0 { return tps }
        }
        return firstPositiveFinite(stats.rawDecodeTokS)
    }

    private static func firstPositiveFinite(_ values: Double?...) -> Double? {
        for value in values {
            guard let value, value.isFinite, value > 0 else { continue }
            return value
        }
        return nil
    }

    private func updatePendingTrace(
        id: String,
        _ mutate: (inout PendingToolTrace) -> Void
    ) {
        guard let index = pendingToolTraces.firstIndex(where: { $0.id == id }) else { return }
        var trace = pendingToolTraces[index]
        mutate(&trace)
        pendingToolTraces[index] = trace
    }

    // MARK: - Stream UI coalescing

    private func startStreamFlushLoop(generation: Int) {
        stopStreamFlushLoop()
        streamFlushTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: Self.streamFlushInterval)
                } catch {
                    return
                }
                self?.flushStreamingBuffersIfCurrent(generation: generation)
            }
        }
    }

    private func stopStreamFlushLoop() {
        streamFlushTask?.cancel()
        streamFlushTask = nil
    }

    private func flushStreamingBuffersIfCurrent(generation: Int) {
        guard generation == streamGeneration else { return }
        flushStreamingBuffers()
    }

    private func flushStreamingBuffers() {
        if !streamingReasoningBuffer.isEmpty {
            let delta = streamingReasoningBuffer
            streamingReasoningBuffer = ""
            streamingReasoningDocument.append(delta)
        }
        if !streamingContentBuffer.isEmpty {
            let delta = streamingContentBuffer
            streamingContentBuffer = ""
            streamingContentDocument.append(delta)
        }
    }

    private func flushLeakedThinkingSplitter() {
        let split = leakedThinkingSplitter.finish()
        appendStreamingReasoning(split.reasoning)
        appendStreamingContent(split.content)
    }

    // MARK: - Persistence helpers

    @discardableResult
    private func persistAssistantTurn(
        conversation: ChatConversation,
        finishReason: String,
        usage: ChatUsage?,
        stats: ChatStreamStats?,
        toolCalls: [AccumulatingToolCall],
        traces: [PendingToolTrace],
        publishImmediately: Bool = true
    ) -> ChatMessage {
        let toolCallRecords = toolCalls.map { call in
            ToolCallRecord(id: call.id, name: call.name, arguments: call.arguments)
        }
        let toolCallsJSON: String?
        if toolCallRecords.isEmpty {
            toolCallsJSON = nil
        } else if let data = try? JSONEncoder().encode(toolCallRecords),
            let str = String(data: data, encoding: .utf8)
        {
            toolCallsJSON = str
        } else {
            toolCallsJSON = nil
        }

        let chatStats = ChatTurnStats(
            rawDecodeTokS: stats?.rawDecodeTokS,
            displayDecodeTokS: stats?.displayDecodeTokS,
            promptTokens: usage?.promptTokens,
            completionTokens: usage?.completionTokens,
            ttftS: stats?.ttftS,
            acceptedByDepth: stats?.acceptedByDepth,
            draftedByDepth: stats?.draftedByDepth,
            verifyCalls: stats?.verifyCalls,
            verifyTimeS: stats?.verifyTimeS
        )
        let statsJSON: String? = {
            guard let data = try? JSONEncoder().encode(chatStats),
                let str = String(data: data, encoding: .utf8)
            else { return nil }
            return str
        }()

        let message = ChatMessage(
            role: .assistant,
            visibleContent: streamingContent,
            reasoningContent: streamingReasoning.isEmpty ? nil : streamingReasoning,
            toolCallsJSON: toolCallsJSON,
            statsJSON: statsJSON,
            finishReason: finishReason,
            createdAt: Date(),
            conversation: conversation
        )
        context.insert(message)
        conversation.messages.append(message)
        conversation.updatedAt = message.createdAt
        for trace in traces {
            persistToolTrace(
                on: message,
                id: trace.id,
                name: trace.name,
                argumentsJSON: nil,
                resultJSON: nil,
                status: trace.status
            )
        }
        saveContext()
        if publishImmediately {
            refreshVisibleMessages()
            refreshConversations()
        }
        return message
    }

    private func persistToolTrace(
        on message: ChatMessage,
        id: String,
        name: String,
        argumentsJSON: String?,
        resultJSON: String?,
        status: ToolTraceStatus
    ) {
        let trace = ToolTraceRecord(
            id: UUID(),
            name: name,
            status: status,
            argumentsJSON: argumentsJSON,
            resultJSON: resultJSON,
            startedAt: Date(),
            completedAt: status == .pending ? nil : Date(),
            message: message
        )
        context.insert(trace)
        message.toolTraces.append(trace)
    }

    private func finalizeAssistantTurnUI() {
        flushStreamingBuffers()
        stopStreamFlushLoop()
        isStreaming = false
        streamingPhase = .idle
        currentRequestId = nil
        turnStartedAt = nil
        reasoningStartedAt = nil
        lastLiveDecodeUpdateAt = .distantPast
        pendingToolTraces = []
        streamingContentDocument.reset()
        streamingReasoningDocument.reset()
        hasStreamingContent = false
        hasStreamingReasoning = false
        handoffAssistantMessageID = nil
        streamingContentBuffer = ""
        streamingReasoningBuffer = ""
    }

    private func finalizePartialAssistantTurn(reason: String) {
        guard isStreaming, let conversation = current else { return }
        flushLeakedThinkingSplitter()
        flushStreamingBuffers()
        var partialMessage: ChatMessage?
        if !streamingContent.isEmpty || !streamingReasoning.isEmpty {
            let message = ChatMessage(
                role: .assistant,
                visibleContent: streamingContent,
                reasoningContent: streamingReasoning.isEmpty ? nil : streamingReasoning,
                finishReason: reason,
                createdAt: Date(),
                conversation: conversation
            )
            context.insert(message)
            conversation.messages.append(message)
            conversation.updatedAt = message.createdAt
            saveContext()
            partialMessage = message
        }
        if let partialMessage {
            publishVisibleMessages(for: conversation, ensuring: partialMessage)
        } else {
            refreshVisibleMessages(preferRelationshipFirst: true)
        }
        refreshConversations()
        finalizeAssistantTurnUI()
    }

    private func handleStreamError(_ error: Error, conversation: ChatConversation) {
        switch error {
        case let chatError as MTPLXChatClientError:
            switch chatError {
            case .unauthorized: lastError = .unauthorized
            case .daemonUnreachable:
                onDaemonUnreachable()
                lastError = .daemonStopped
            case .httpStatus(let code, let body): lastError = .http(code, body)
            case .bodyEncodingFailed: lastError = .malformedRequest
            case .invalidResponse: lastError = .streamLost
            }
        default:
            lastError = .unknown(error.localizedDescription)
        }
        finalizePartialAssistantTurn(reason: "error")
    }

    // MARK: - Glue

    private func clearStreamingState() {
        isStreaming = false
        streamingPhase = .idle
        stopStreamFlushLoop()
        streamingReasoningDocument.reset()
        streamingContentDocument.reset()
        hasStreamingReasoning = false
        hasStreamingContent = false
        handoffAssistantMessageID = nil
        streamingReasoningBuffer = ""
        streamingContentBuffer = ""
        leakedThinkingSplitter.reset()
        pendingToolTraces = []
        currentRequestId = nil
        chatDecodeReading = .absent
        lastError = nil
        lastLiveDecodeUpdateAt = .distantPast
    }

    private func refreshVisibleMessages(preferRelationshipFirst: Bool = false) {
        guard let current else {
            visibleMessages = []
            return
        }
        visibleMessages = loadMessages(for: current, preferRelationshipFirst: preferRelationshipFirst)
    }

    private func publishVisibleMessages(
        for conversation: ChatConversation,
        ensuring message: ChatMessage
    ) {
        guard current?.id == conversation.id else { return }
        var loaded = loadMessages(for: conversation, preferRelationshipFirst: true)
        if !loaded.contains(where: { $0.id == message.id }) {
            loaded.append(message)
            loaded.sort { $0.createdAt < $1.createdAt }
        }
        visibleMessages = loaded
    }

    private func loadMessages(
        for conversation: ChatConversation,
        preferRelationshipFirst: Bool = false
    ) -> [ChatMessage] {
        let conversationID = conversation.id
        let relationshipMessages = conversation.messages.sorted { $0.createdAt < $1.createdAt }
        if preferRelationshipFirst, !relationshipMessages.isEmpty {
            return uniqueSortedMessages(relationshipMessages)
        }

        var candidates = relationshipMessages

        let idDescriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate<ChatMessage> { message in
                message.conversationID == conversationID
            },
            sortBy: [SortDescriptor(\.createdAt)]
        )
        if let fetchedByID = try? context.fetch(idDescriptor) {
            candidates.append(contentsOf: fetchedByID)
        }

        let relationshipDescriptor = FetchDescriptor<ChatMessage>(
            predicate: #Predicate<ChatMessage> { message in
                message.conversation?.id == conversationID
            },
            sortBy: [SortDescriptor(\.createdAt)]
        )
        if let fetchedByRelationship = try? context.fetch(relationshipDescriptor) {
            candidates.append(contentsOf: fetchedByRelationship)
        }

        let merged = uniqueSortedMessages(candidates)
        if !merged.isEmpty {
            return merged
        }

        let allDescriptor = FetchDescriptor<ChatMessage>(
            sortBy: [SortDescriptor(\.createdAt)]
        )
        let allMessages = (try? context.fetch(allDescriptor)) ?? []
        return uniqueSortedMessages(
            allMessages.filter { message in
                message.conversationID == conversationID
                    || message.conversation?.id == conversationID
            }
        )
    }

    private func uniqueSortedMessages(_ messages: [ChatMessage]) -> [ChatMessage] {
        var seen = Set<UUID>()
        return messages
            .sorted {
                if $0.createdAt == $1.createdAt {
                    return $0.id.uuidString < $1.id.uuidString
                }
                return $0.createdAt < $1.createdAt
            }
            .filter { message in
                if seen.contains(message.id) { return false }
                seen.insert(message.id)
                return true
            }
    }

    private func saveContext() {
        do {
            try context.save()
        } catch {
            lastError = .unknown("Persist failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Static helpers

    private static func buildAttachmentContext(attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return "" }
        return attachments
            .filter(isSendableAttachment)
            .map { attachment in
                "[Attached file: \(attachment.filename)]\n\(attachment.extractedText)\n[End of attachment]"
            }
            .joined(separator: "\n\n")
    }

    private static func fullUserContent(for message: ChatMessage) -> String {
        let attachmentText = buildAttachmentContext(attachments: message.attachments)
        guard !attachmentText.isEmpty else { return message.visibleContent }
        let text = message.visibleContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? attachmentText : "\(message.visibleContent)\n\n\(attachmentText)"
    }

    private static func isSendableAttachment(_ attachment: ChatAttachment) -> Bool {
        !attachment.extractedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func firstNWords(_ text: String, n: Int) -> String {
        let words = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .prefix(n)
            .map { String($0) }
        let joined = words.joined(separator: " ")
        return joined.isEmpty ? "New Chat" : joined
    }

    static func buildRequestMessages(
        from persisted: [ChatMessage],
        overrideLastUserContent: String?
    ) -> [ChatRequestMessage] {
        var output: [ChatRequestMessage] = []
        for (index, message) in persisted.enumerated() {
            switch message.role {
            case .user:
                let isLast = (index == persisted.count - 1)
                let content =
                    (isLast && overrideLastUserContent != nil)
                    ? overrideLastUserContent
                    : message.visibleContent
                output.append(ChatRequestMessage(role: "user", content: content))
            case .assistant:
                output.append(assistantRequestMessage(from: message))
            case .tool:
                output.append(
                    ChatRequestMessage(
                        role: "tool",
                        content: compactToolResultContent(message.visibleContent),
                        toolCallId: message.toolCallId
                    )
                )
            case .system:
                output.append(ChatRequestMessage(role: "system", content: message.visibleContent))
            }
        }
        return compactRequestMessagesIfNeeded(output)
    }

    static func buildRetryRequestMessages(
        from persisted: [ChatMessage],
        retrying userMessage: ChatMessage,
        fullUserContent: String
    ) -> [ChatRequestMessage] {
        guard let retryIndex = persisted.lastIndex(where: { $0.id == userMessage.id }) else {
            return []
        }
        let retryWindow = Array(persisted.prefix(through: retryIndex))
        return buildRequestMessages(
            from: retryWindow,
            overrideLastUserContent: fullUserContent
        )
    }

    private static func compactRequestMessagesIfNeeded(
        _ messages: [ChatRequestMessage]
    ) -> [ChatRequestMessage] {
        guard requestCharacterCount(messages) > requestContextCharacterBudget else {
            return messages
        }

        let verbatimStart = max(0, messages.count - requestRecentVerbatimMessageCount)
        var compacted = messages.enumerated().map { index, message in
            index < verbatimStart ? compactHistoricalRequestMessage(message) : message
        }
        var omittedCount = 0

        while requestCharacterCount(compacted) > requestContextCharacterBudget,
              compacted.count > requestRecentVerbatimMessageCount {
            guard let removalIndex = compacted.indices.first(where: { index in
                index < compacted.count - requestRecentVerbatimMessageCount
                    && compacted[index].role != "system"
            }) else {
                break
            }
            compacted.remove(at: removalIndex)
            omittedCount += 1
        }

        if omittedCount > 0 {
            compacted.insert(
                ChatRequestMessage(
                    role: "system",
                    content: "Earlier conversation turns were omitted from this request to keep local chat responsive. The full transcript remains visible in the app."
                ),
                at: 0
            )
        }
        return compacted
    }

    private static func compactHistoricalRequestMessage(
        _ message: ChatRequestMessage
    ) -> ChatRequestMessage {
        var copy = message
        guard let content = message.content, !content.isEmpty else {
            return copy
        }
        let withoutLargeCode: String
        if message.role == "assistant" {
            withoutLargeCode = compactCodeFences(in: content)
        } else {
            withoutLargeCode = content
        }
        copy.content = clampHistoricalContent(withoutLargeCode)
        return copy
    }

    private static func requestCharacterCount(_ messages: [ChatRequestMessage]) -> Int {
        messages.reduce(0) { total, message in
            let contentCount = message.content?.count ?? 0
            let toolCallCount = message.toolCalls?.reduce(0) { partial, call in
                partial + call.function.arguments.count + call.function.name.count
            } ?? 0
            return total + contentCount + toolCallCount
        }
    }

    private static func clampHistoricalContent(_ content: String) -> String {
        guard content.count > requestHistoricalContentLimit else { return content }
        let headCount = requestHistoricalContentLimit / 2
        let tailCount = requestHistoricalContentLimit - headCount
        let omitted = max(0, content.count - requestHistoricalContentLimit)
        return """
        \(content.prefix(headCount))

        [omitted \(omitted) historical characters]

        \(content.suffix(tailCount))
        """
    }

    static func compactToolResultContent(_ content: String) -> String {
        guard content.count > requestToolResultContentLimit else { return content }
        if let compactedJSON = compactToolResultJSON(content) {
            return compactedJSON.count > requestToolResultContentLimit
                ? clampToolResultContent(compactedJSON)
                : compactedJSON
        }
        return clampToolResultContent(content)
    }

    private static func compactToolResultJSON(_ content: String) -> String? {
        guard let data = content.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dictionary = object as? [String: Any]
        else {
            return nil
        }

        var compacted: [String: Any] = [
            "compact_notice": "Tool result compacted from \(content.count) characters to keep local chat responsive."
        ]
        for key in ["query", "url", "title", "host"] {
            if let value = dictionary[key] as? String, !value.isEmpty {
                compacted[key] = value
            }
        }

        if let results = dictionary["results"] as? [[String: Any]], !results.isEmpty {
            compacted["results"] = results.prefix(requestToolResultMaxResults).map(compactToolSearchResult)
            if results.count > requestToolResultMaxResults {
                compacted["omitted_results"] = results.count - requestToolResultMaxResults
            }
        }

        if let content = dictionary["content"] as? String, !content.isEmpty {
            compacted["content_excerpt"] = excerptToolText(content)
            let omitted = max(0, content.count - requestToolResultExcerptLimit)
            if omitted > 0 {
                compacted["content_omitted_chars"] = omitted
            }
        }

        guard compacted.count > 1,
              let compactedData = try? JSONSerialization.data(
                withJSONObject: compacted,
                options: [.sortedKeys]
              )
        else {
            return nil
        }
        return String(data: compactedData, encoding: .utf8)
    }

    private static func compactToolSearchResult(_ result: [String: Any]) -> [String: Any] {
        var compacted: [String: Any] = [:]
        for key in ["title", "url", "host", "snippet"] {
            if let value = result[key] as? String, !value.isEmpty {
                compacted[key] = value
            }
        }
        if let pageContent = result["page_content"] as? String, !pageContent.isEmpty {
            compacted["page_excerpt"] = excerptToolText(pageContent)
            let omitted = max(0, pageContent.count - requestToolResultExcerptLimit)
            if omitted > 0 {
                compacted["page_content_omitted_chars"] = omitted
            }
        }
        return compacted
    }

    private static func excerptToolText(_ content: String) -> String {
        let cleanedLines = content
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { line in
                guard !line.isEmpty else { return false }
                if line.hasPrefix("--") || line.hasPrefix(".") || line.hasPrefix("#") {
                    return false
                }
                if line.contains("{") || line.contains("}") || line.contains("var(") {
                    return false
                }
                return true
            }
        let cleaned = cleanedLines.isEmpty
            ? content.trimmingCharacters(in: .whitespacesAndNewlines)
            : cleanedLines.joined(separator: "\n")
        guard cleaned.count > requestToolResultExcerptLimit else {
            return cleaned
        }
        return String(cleaned.prefix(requestToolResultExcerptLimit))
    }

    private static func clampToolResultContent(_ content: String) -> String {
        guard content.count > requestToolResultContentLimit else { return content }
        let headCount = requestToolResultContentLimit / 2
        let tailCount = requestToolResultContentLimit - headCount
        let omitted = max(0, content.count - requestToolResultContentLimit)
        return """
        \(content.prefix(headCount))

        [omitted \(omitted) tool result characters to keep local chat responsive]

        \(content.suffix(tailCount))
        """
    }

    private static func compactCodeFences(in source: String) -> String {
        guard source.contains("```") else { return source }
        var result = ""
        var cursor = source.startIndex

        while cursor < source.endIndex {
            guard let openingFence = source.range(of: "```", range: cursor..<source.endIndex) else {
                result.append(contentsOf: source[cursor..<source.endIndex])
                break
            }

            result.append(contentsOf: source[cursor..<openingFence.upperBound])
            let languageStart = openingFence.upperBound
            guard let languageEnd = source[languageStart...].firstIndex(of: "\n") else {
                result.append(contentsOf: source[languageStart..<source.endIndex])
                break
            }

            let rawLanguage = String(source[languageStart..<languageEnd])
            result.append(rawLanguage)
            result.append("\n")
            let bodyStart = source.index(after: languageEnd)
            let closingFence = source.range(of: "```", range: bodyStart..<source.endIndex)
            let bodyEnd = closingFence?.lowerBound ?? source.endIndex
            let code = source[bodyStart..<bodyEnd]
            let languageLabel = rawLanguage.trimmingCharacters(in: .whitespacesAndNewlines)
            let label = languageLabel.isEmpty ? "code" : languageLabel

            if code.count > requestHistoricalContentLimit {
                result.append("[omitted historical \(label) code block, \(code.count) characters]\n")
            } else {
                result.append(contentsOf: code)
            }

            if let closingFence {
                result.append("```")
                cursor = closingFence.upperBound
            } else {
                cursor = source.endIndex
            }
        }

        return result
    }

    private static func assistantRequestMessage(from message: ChatMessage) -> ChatRequestMessage {
        var toolCalls: [ChatRequestToolCall]? = nil
        if let json = message.toolCallsJSON,
            let data = json.data(using: .utf8),
            let records = try? JSONDecoder().decode([ToolCallRecord].self, from: data),
            !records.isEmpty
        {
            toolCalls = records.map { record in
                ChatRequestToolCall(
                    id: record.id,
                    function: ChatRequestToolCallFunction(
                        name: record.name,
                        arguments: record.arguments
                    )
                )
            }
        }
        return ChatRequestMessage(
            role: "assistant",
            content: message.visibleContent.isEmpty ? nil : message.visibleContent,
            toolCalls: toolCalls
        )
    }

    private static func streamingPhase(forTool name: String) -> StreamingPhase {
        switch name {
        case "web_search": return .searching
        case "fetch_url": return .reading
        default: return .answering
        }
    }

    private static func shortArgsSubtitle(for call: AccumulatingToolCall) -> String {
        switch call.name {
        case "web_search":
            if let data = call.arguments.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let query = dict["query"] as? String, !query.isEmpty
            {
                return "Searching: \(query)"
            }
            return "Searching"
        case "fetch_url":
            if let data = call.arguments.data(using: .utf8),
                let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let url = dict["url"] as? String, !url.isEmpty
            {
                return url
            }
            return "Reading URL"
        default:
            return call.name.replacingOccurrences(of: "_", with: " ")
        }
    }

    private static func liveDetail(for toolName: String) -> String {
        switch toolName {
        case "web_search": return "Querying DuckDuckGo + Brave…"
        case "fetch_url": return "Fetching page content…"
        default: return "Running tool…"
        }
    }

    private static func shortResultDetail(for toolName: String, json: String) -> String {
        guard let data = json.data(using: .utf8),
            let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return "Done" }
        if let error = dict["error"] as? String {
            return "Error: \(error)"
        }
        switch toolName {
        case "web_search":
            if let results = dict["results"] as? [[String: Any]] {
                let titles = results.prefix(3)
                    .compactMap { $0["title"] as? String }
                    .joined(separator: " · ")
                return "Found \(results.count) results — \(titles)"
            }
            return "Done"
        case "fetch_url":
            if let title = dict["title"] as? String, !title.isEmpty {
                return "Read: \(title)"
            }
            return "Read"
        default:
            return "Done"
        }
    }
}

// MARK: - Internal accumulator

private struct AccumulatingToolCall: Sendable {
    var id: String
    var name: String
    var arguments: String
}

private struct ChatThinkingTagSplitter {
    struct Split {
        var reasoning = ""
        var content = ""
    }

    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    private var pending = ""
    private var insideThinking = false

    mutating func reset() {
        pending = ""
        insideThinking = false
    }

    mutating func feed(_ fragment: String) -> Split {
        guard !fragment.isEmpty else { return Split() }
        pending.append(fragment)
        return drain(flush: false)
    }

    mutating func finish() -> Split {
        drain(flush: true)
    }

    private mutating func drain(flush: Bool) -> Split {
        var split = Split()
        while !pending.isEmpty {
            if insideThinking {
                if let close = pending.range(of: Self.closeTag, options: [.caseInsensitive]) {
                    split.reasoning.append(contentsOf: pending[..<close.lowerBound])
                    pending.removeSubrange(..<close.upperBound)
                    insideThinking = false
                    continue
                }
                if flush {
                    split.reasoning.append(pending)
                    pending.removeAll()
                    break
                }
                Self.emitPrefix(
                    of: &pending,
                    holdingPossiblePrefixOf: Self.closeTag,
                    into: &split.reasoning
                )
            } else {
                if let open = pending.range(of: Self.openTag, options: [.caseInsensitive]) {
                    split.content.append(contentsOf: pending[..<open.lowerBound])
                    pending.removeSubrange(..<open.upperBound)
                    insideThinking = true
                    continue
                }
                if flush {
                    split.content.append(pending)
                    pending.removeAll()
                    break
                }
                Self.emitPrefix(
                    of: &pending,
                    holdingPossiblePrefixOf: Self.openTag,
                    into: &split.content
                )
            }
            break
        }
        return split
    }

    private static func emitPrefix(
        of buffer: inout String,
        holdingPossiblePrefixOf tag: String,
        into output: inout String
    ) {
        let hold = holdCount(in: buffer, tag: tag)
        guard hold < buffer.count else { return }
        let emitEnd = buffer.index(buffer.endIndex, offsetBy: -hold)
        output.append(contentsOf: buffer[..<emitEnd])
        buffer.removeSubrange(..<emitEnd)
    }

    private static func holdCount(in text: String, tag: String) -> Int {
        let lowerText = text.lowercased()
        let lowerTag = tag.lowercased()
        let maxHold = min(max(lowerTag.count - 1, 0), lowerText.count)
        guard maxHold > 0 else { return 0 }
        for count in stride(from: maxHold, through: 1, by: -1) {
            if lowerTag.hasPrefix(String(lowerText.suffix(count))) {
                return count
            }
        }
        return 0
    }
}
