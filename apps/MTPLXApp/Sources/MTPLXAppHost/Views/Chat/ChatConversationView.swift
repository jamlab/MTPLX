import AppKit
import QuartzCore
import SwiftUI
import MTPLXAppCore

// MARK: - ChatConversationView
//
// Scrollable column of messages. Persisted user / assistant / tool
// turns render as bubbles; the in-flight assistant turn renders as
// `StreamingAssistantView` at the bottom. Empty state takes over when
// the conversation has no messages.
//
// Auto-scroll: every render scrolls to the bottom anchor if streaming
// and `policy.shouldAutoScrollForStreamingUpdate` is true. The user
// can scroll up to detach (>120pt) and back to the bottom to reattach
// (<28pt), matching Aphanes' tuning.

struct ChatConversationView: View {
    @ObservedObject var viewModel: ChatViewModel
    @State private var policy = ConversationAutoScrollPolicy()
    @State private var autoScrollTask: Task<Void, Never>?
    @State private var deferredScrollTask: Task<Void, Never>?
    @State private var finishScrollRepairTask: Task<Void, Never>?
    @State private var lastAutoScrollAt: ContinuousClock.Instant?
    @State private var scrollDriver = ChatConversationScrollDriver()
    @State private var showFullHeavyTranscript = false
    @State private var renderPlan = ChatConversationRenderPlan(
        messages: [],
        showFullHeavyTranscript: false
    )

    var body: some View {
        let plan = activeRenderPlan
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 16) {
                if let hiddenTranscriptSummary = plan.hiddenTranscriptSummary {
                    HiddenTranscriptSummaryView(
                        summary: hiddenTranscriptSummary,
                        onShow: revealFullHeavyTranscript
                    )
                    .id("hidden-heavy-transcript-summary")
                }
                ForEach(plan.renderedMessages, id: \.id) { message in
                    switch message.role {
                    case .user:
                        UserBubbleView(message: message)
                            .id(message.id)
                    case .assistant:
                        AssistantBubbleView(message: message)
                            .id(message.id)
                    case .tool:
                        EmptyView()
                            .id(message.id)
                    case .system:
                        EmptyView()
                            .id(message.id)
                    }
                }
                if viewModel.shouldRenderStreamingAssistant {
                    StreamingAssistantView(viewModel: viewModel)
                        .id("streaming-bubble")
                }
                if let error = viewModel.lastError {
                    errorCard(error)
                        .id("error-card")
                }
            }
            .frame(maxWidth: 768)
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
            .frame(maxWidth: .infinity)
            .transaction { transaction in
                if viewModel.isStreaming {
                    transaction.animation = nil
                }
            }
            .background(
                ChatConversationScrollObserverView(
                    onScrollViewResolved: { scrollView in
                        scrollDriver.updateScrollView(scrollView)
                        if scrollView != nil, policy.shouldAutoScrollForStreamingUpdate {
                            scheduleDeferredBottomScroll(delays: [.milliseconds(40)])
                        }
                    },
                    onScroll: { distanceToBottom, isUserInitiated in
                        performScrollActions(
                            policy.didScroll(
                                distanceToBottom: distanceToBottom,
                                isUserInitiated: isUserInitiated
                            )
                        )
                    }
                )
            )
        }
        .overlay(alignment: .center) {
            if plan.renderableMessages.isEmpty && !viewModel.isStreaming {
                ChatConversationEmptyStateView()
            }
        }
        .background(Brand.bgOuter)
        .onReceive(viewModel.streamingContentDocument.revisionPublisher) { _ in
            scrollToBottom()
        }
        .onReceive(viewModel.streamingReasoningDocument.revisionPublisher) { _ in
            scrollToBottom()
        }
        .onChange(of: viewModel.current?.id) { _, _ in
            handleConversationChange()
        }
        .onChange(of: viewModel.visibleMessages.count) { _, _ in
            updateRenderPlan()
            if viewModel.visibleMessages.last?.role == .user {
                performScrollActions(policy.didSendUserMessage())
            } else if !viewModel.isStreaming && policy.shouldAutoScrollForStreamingUpdate {
                performScrollActions([.immediate, .deferred])
                scheduleFinishScrollRepair()
            } else {
                scrollToBottom()
            }
        }
        .onChange(of: viewModel.isStreaming) { _, streaming in
            if streaming {
                finishScrollRepairTask?.cancel()
                finishScrollRepairTask = nil
                performScrollActions(policy.didStartStreaming())
            } else {
                performScrollActions(policy.didFinishStreaming())
                scheduleFinishScrollRepair()
            }
        }
        .onAppear {
            updateRenderPlan()
            performScrollActions(policy.didAppear())
        }
        .onDisappear {
            autoScrollTask?.cancel()
            deferredScrollTask?.cancel()
            finishScrollRepairTask?.cancel()
            autoScrollTask = nil
            deferredScrollTask = nil
            finishScrollRepairTask = nil
        }
    }

    private func scrollToBottom(force: Bool = false) {
        guard force || policy.shouldAutoScrollForStreamingUpdate else { return }
        if force {
            autoScrollTask?.cancel()
            autoScrollTask = nil
            performAutoScroll(animated: false)
            return
        }

        guard autoScrollTask == nil else { return }

        let minimumCadence: Duration
        if viewModel.isStreaming {
            minimumCadence = usesHeavyTranscriptScrollGuard
                ? .milliseconds(50)
                : .milliseconds(24)
        } else {
            minimumCadence = usesHeavyTranscriptScrollGuard
                ? .milliseconds(120)
                : .milliseconds(50)
        }
        let now = ContinuousClock.now
        let delay: Duration
        if let lastAutoScrollAt {
            let elapsed = now - lastAutoScrollAt
            delay = elapsed >= minimumCadence ? .zero : minimumCadence - elapsed
        } else {
            delay = .zero
        }

        autoScrollTask = Task { @MainActor in
            if delay > .zero {
                try? await Task.sleep(for: delay)
            } else {
                await Task.yield()
            }
            guard !Task.isCancelled else { return }
            autoScrollTask = nil
            guard policy.shouldAutoScrollForStreamingUpdate else { return }
            performAutoScroll(animated: false)
        }
    }

    private func performScrollActions(_ actions: [ConversationAutoScrollAction]) {
        guard !actions.isEmpty else { return }
        let guarded = usesHeavyTranscriptScrollGuard
        if actions.contains(.immediate) {
            if guarded {
                scheduleDeferredBottomScroll(delays: [.milliseconds(140)])
            } else {
                scrollToBottom(force: true)
            }
        }
        if actions.contains(.deferred) {
            scheduleDeferredBottomScroll(
                delays: guarded
                    ? [.milliseconds(260)]
                    : [.milliseconds(60), .milliseconds(120), .milliseconds(240)]
            )
        }
    }

    private func scheduleDeferredBottomScroll(delays: [Duration]) {
        deferredScrollTask?.cancel()
        deferredScrollTask = Task { @MainActor in
            for delay in delays {
                try? await Task.sleep(for: delay)
                guard !Task.isCancelled, policy.shouldAutoScrollForStreamingUpdate else { return }
                performAutoScroll(animated: false)
            }
        }
    }

    private func performAutoScroll(animated: Bool) {
        lastAutoScrollAt = ContinuousClock.now
        _ = scrollDriver.scrollToBottom(animated: animated)
    }

    private func scheduleFinishScrollRepair() {
        finishScrollRepairTask?.cancel()
        finishScrollRepairTask = Task { @MainActor in
            await Task.yield()
            guard !Task.isCancelled else { return }
            performFinishScrollRepairTick()
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled else { return }
            performFinishScrollRepairTick()
        }
    }

    private func performFinishScrollRepairTick() {
        guard !viewModel.isStreaming, policy.shouldAutoScrollForStreamingUpdate else { return }
        lastAutoScrollAt = ContinuousClock.now
        _ = scrollDriver.clampToValidOffset()
        _ = scrollDriver.scrollToBottom(animated: false)
    }

    private func handleConversationChange() {
        autoScrollTask?.cancel()
        deferredScrollTask?.cancel()
        finishScrollRepairTask?.cancel()
        autoScrollTask = nil
        deferredScrollTask = nil
        finishScrollRepairTask = nil
        performScrollActions(policy.didOpenConversation())
        showFullHeavyTranscript = false
        updateRenderPlan(showFullHeavyTranscript: false)
    }

    private var usesHeavyTranscriptScrollGuard: Bool {
        activeRenderPlan.usesHeavyTranscriptScrollGuard
    }

    private var activeRenderPlan: ChatConversationRenderPlan {
        if renderPlan.matches(
            messages: viewModel.visibleMessages,
            showFullHeavyTranscript: showFullHeavyTranscript
        ) {
            return renderPlan
        }
        return ChatConversationRenderPlan(
            messages: viewModel.visibleMessages,
            showFullHeavyTranscript: showFullHeavyTranscript
        )
    }

    private func updateRenderPlan(showFullHeavyTranscript override: Bool? = nil) {
        renderPlan = ChatConversationRenderPlan(
            messages: viewModel.visibleMessages,
            showFullHeavyTranscript: override ?? showFullHeavyTranscript
        )
    }

    private func revealFullHeavyTranscript() {
        showFullHeavyTranscript = true
        updateRenderPlan(showFullHeavyTranscript: true)
    }

    private func errorCard(_ error: ChatError) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Brand.warning)
            VStack(alignment: .leading, spacing: 4) {
                Text(error.errorDescription ?? "Something went wrong.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Brand.typeHi)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if case .daemonStopped = error {
                    Text("Hit the play button to start a model, then send again.")
                        .font(.system(size: 11))
                        .foregroundStyle(Brand.typeSecondary)
                }
            }
            if offersRetry(for: error) {
                Button(action: { viewModel.retryLastUserMessage() }) {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .font(.system(size: 11, weight: .semibold))
                        .labelStyle(.titleAndIcon)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(Brand.warning.opacity(0.14))
                        )
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Brand.warning.opacity(0.38), lineWidth: 0.5)
                        )
                }
                .buttonStyle(.plain)
                .foregroundStyle(Brand.typeHi)
                .disabled(!viewModel.canRetryLastUserMessage)
                .help("Retry the last message")
                .accessibilityLabel("Retry last message")
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.warning.opacity(0.10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.warning.opacity(0.40), lineWidth: 0.5)
                )
        )
        .frame(maxWidth: 768)
    }

    private func offersRetry(for error: ChatError) -> Bool {
        if case .daemonStopped = error {
            return false
        }
        return true
    }
}

private struct ChatConversationEmptyStateView: View {
    @EnvironmentObject private var backend: MTPLXBackendStore

    var body: some View {
        if let startupState {
            ChatStartupStatusView(state: startupState)
        } else {
            ChatEmptyView()
        }
    }

    private var startupState: ChatStartupStatusView.State? {
        switch backend.daemonState.kind {
        case .starting, .warming:
            return ChatStartupStatusView.State(
                title: startupTitle,
                detail: startupDetail
            )
        case .stopping:
            return ChatStartupStatusView.State(
                title: "Stopping MTPLX",
                detail: "The app is closing the model server and restoring fans."
            )
        default:
            return nil
        }
    }

    private var startupTitle: String {
        switch backend.startupPhase {
        case .launching:
            return "Starting \(selectedModelName)"
        case .waitingForOwnedHealth:
            return "Loading \(selectedModelName)"
        case .rampingFans:
            return "Preparing cooling"
        case .warming:
            return "Warming up \(selectedModelName)"
        case .ready:
            return "\(selectedModelName) is ready"
        case .failed:
            return "Startup failed"
        case .idle:
            return "Starting \(selectedModelName)"
        }
    }

    private var startupDetail: String {
        switch backend.startupPhase {
        case .launching:
            return "Starting the local model…"
        case .waitingForOwnedHealth:
            return "Mapping weights and building the draft head. Large Step loads can take a minute or two cold."
        case .rampingFans:
            return "Waiting for the requested fan profile."
        case .warming:
            return "Running the first warmup tokens before chat opens."
        case .failed(let message):
            return message
        case .ready:
            return "You can send now."
        case .idle:
            return "Preparing the local engine."
        }
    }

    private var selectedModelName: String {
        if let option = MTPLXModelOption.option(matching: backend.configuration.model) {
            return option.shortName
        }
        let expanded = NSString(string: backend.configuration.model).expandingTildeInPath
        let last = URL(fileURLWithPath: expanded).lastPathComponent
        return last.isEmpty ? backend.configuration.model : last
    }
}

private struct HiddenTranscriptSummary: Equatable {
    let messageCount: Int
    let characterCount: Int
}

private struct ChatConversationRenderPlan {
    private static let heavyMessageCharacterThreshold = 3_000
    private static let heavyTranscriptCharacterThreshold = 18_000
    private static let heavyTranscriptTailMessageCount = 4

    let renderableMessages: [ChatMessage]
    let renderedMessages: [ChatMessage]
    let hiddenTranscriptSummary: HiddenTranscriptSummary?
    let usesHeavyTranscriptScrollGuard: Bool

    private let sourceMessageCount: Int
    private let firstSourceMessageID: UUID?
    private let lastSourceMessageID: UUID?
    private let showFullHeavyTranscript: Bool

    init(messages: [ChatMessage], showFullHeavyTranscript: Bool) {
        self.sourceMessageCount = messages.count
        self.firstSourceMessageID = messages.first?.id
        self.lastSourceMessageID = messages.last?.id
        self.showFullHeavyTranscript = showFullHeavyTranscript

        var totalCharacters = 0
        var heavy = false
        var renderable: [ChatMessage] = []
        renderable.reserveCapacity(messages.count)

        for message in messages {
            if Self.isRenderableMessage(message) {
                renderable.append(message)
            }

            if !heavy {
                let messageCharacters = message.visibleContent.count + (message.reasoningContent?.count ?? 0)
                if messageCharacters >= Self.heavyMessageCharacterThreshold {
                    heavy = true
                } else {
                    totalCharacters += messageCharacters
                    if totalCharacters >= Self.heavyTranscriptCharacterThreshold {
                        heavy = true
                    }
                }
            }
        }

        self.renderableMessages = renderable
        self.usesHeavyTranscriptScrollGuard = heavy

        guard !showFullHeavyTranscript,
              heavy,
              renderable.count > Self.heavyTranscriptTailMessageCount
        else {
            self.renderedMessages = renderable
            self.hiddenTranscriptSummary = nil
            return
        }

        let hidden = renderable.dropLast(Self.heavyTranscriptTailMessageCount)
        let hiddenCharacters = hidden.reduce(0) { total, message in
            total + message.visibleContent.count + (message.reasoningContent?.count ?? 0)
        }
        self.renderedMessages = Array(renderable.suffix(Self.heavyTranscriptTailMessageCount))
        self.hiddenTranscriptSummary = HiddenTranscriptSummary(
            messageCount: hidden.count,
            characterCount: hiddenCharacters
        )
    }

    func matches(messages: [ChatMessage], showFullHeavyTranscript: Bool) -> Bool {
        sourceMessageCount == messages.count
            && firstSourceMessageID == messages.first?.id
            && lastSourceMessageID == messages.last?.id
            && self.showFullHeavyTranscript == showFullHeavyTranscript
    }

    private static func isRenderableMessage(_ message: ChatMessage) -> Bool {
        switch message.role {
        case .user, .assistant:
            return true
        case .tool, .system:
            return false
        }
    }
}

private struct HiddenTranscriptSummaryView: View {
    let summary: HiddenTranscriptSummary
    let onShow: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Brand.typeTertiary)
            VStack(alignment: .leading, spacing: 3) {
                Text("Earlier heavy history hidden")
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(0.6)
                    .foregroundStyle(Brand.typeSecondary)
                Text("\(summary.messageCount) messages · \(Self.formatCount(summary.characterCount))")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Brand.typeTertiary)
            }
            Spacer(minLength: 8)
            Button(action: onShow) {
                Label("Show", systemImage: "arrow.down.right.and.arrow.up.left")
                    .font(.system(size: 10, weight: .semibold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(Brand.typeSecondary)
            .help("Show older messages")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Brand.cardSurface.opacity(0.74))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Brand.separator, lineWidth: 0.5)
                )
        )
        .frame(maxWidth: 576, alignment: .leading)
    }

    private static func formatCount(_ value: Int) -> String {
        if value >= 1000 {
            return String(format: "%.1fk chars", Double(value) / 1000.0)
        }
        return "\(value) chars"
    }
}

private struct ChatConversationScrollObserverView: NSViewRepresentable {
    let onScrollViewResolved: @MainActor (NSScrollView?) -> Void
    let onScroll: @MainActor (CGFloat, Bool) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onScrollViewResolved: onScrollViewResolved, onScroll: onScroll)
    }

    func makeNSView(context: Context) -> HostView {
        let view = HostView()
        view.coordinator = context.coordinator
        return view
    }

    func updateNSView(_ nsView: HostView, context: Context) {
        context.coordinator.onScrollViewResolved = onScrollViewResolved
        context.coordinator.onScroll = onScroll
        nsView.coordinator = context.coordinator
        DispatchQueue.main.async {
            context.coordinator.attachIfNeeded(from: nsView)
        }
    }

    static func dismantleNSView(_ nsView: HostView, coordinator: Coordinator) {
        coordinator.detach()
    }

    @MainActor
    final class Coordinator: NSObject {
        var onScrollViewResolved: @MainActor (NSScrollView?) -> Void
        var onScroll: @MainActor (CGFloat, Bool) -> Void
        weak var scrollView: NSScrollView?
        private var boundsObserver: NSObjectProtocol?
        private var liveScrollStartObserver: NSObjectProtocol?
        private var liveScrollEndObserver: NSObjectProtocol?
        private var isUserLiveScrolling = false

        init(
            onScrollViewResolved: @escaping @MainActor (NSScrollView?) -> Void,
            onScroll: @escaping @MainActor (CGFloat, Bool) -> Void
        ) {
            self.onScrollViewResolved = onScrollViewResolved
            self.onScroll = onScroll
        }

        func attachIfNeeded(from hostView: HostView) {
            guard let resolvedScrollView = Self.findEnclosingScrollView(from: hostView) else {
                return
            }

            guard scrollView !== resolvedScrollView else { return }
            detach()

            scrollView = resolvedScrollView
            onScrollViewResolved(resolvedScrollView)
            resolvedScrollView.contentView.postsBoundsChangedNotifications = true
            liveScrollStartObserver = NotificationCenter.default.addObserver(
                forName: NSScrollView.willStartLiveScrollNotification,
                object: resolvedScrollView,
                queue: .main
            ) { [weak hostView, weak resolvedScrollView] _ in
                Task { @MainActor [weak hostView, weak resolvedScrollView] in
                    guard let hostView, let resolvedScrollView, let coordinator = hostView.coordinator else { return }
                    coordinator.isUserLiveScrolling = true
                    coordinator.onScroll(Self.distanceToBottom(for: resolvedScrollView), true)
                }
            }
            liveScrollEndObserver = NotificationCenter.default.addObserver(
                forName: NSScrollView.didEndLiveScrollNotification,
                object: resolvedScrollView,
                queue: .main
            ) { [weak hostView, weak resolvedScrollView] _ in
                Task { @MainActor [weak hostView, weak resolvedScrollView] in
                    guard let hostView, let resolvedScrollView, let coordinator = hostView.coordinator else { return }
                    coordinator.onScroll(Self.distanceToBottom(for: resolvedScrollView), true)
                    coordinator.isUserLiveScrolling = false
                }
            }
            boundsObserver = NotificationCenter.default.addObserver(
                forName: NSView.boundsDidChangeNotification,
                object: resolvedScrollView.contentView,
                queue: .main
            ) { [weak hostView, weak resolvedScrollView] _ in
                Task { @MainActor [weak hostView, weak resolvedScrollView] in
                    guard let hostView, let resolvedScrollView, let coordinator = hostView.coordinator else { return }
                    coordinator.onScroll(
                        Self.distanceToBottom(for: resolvedScrollView),
                        coordinator.isUserLiveScrolling
                    )
                }
            }
        }

        func detach() {
            if let boundsObserver {
                NotificationCenter.default.removeObserver(boundsObserver)
            }
            if let liveScrollStartObserver {
                NotificationCenter.default.removeObserver(liveScrollStartObserver)
            }
            if let liveScrollEndObserver {
                NotificationCenter.default.removeObserver(liveScrollEndObserver)
            }
            boundsObserver = nil
            liveScrollStartObserver = nil
            liveScrollEndObserver = nil
            isUserLiveScrolling = false
            scrollView = nil
            onScrollViewResolved(nil)
        }

        private static func findEnclosingScrollView(from view: NSView?) -> NSScrollView? {
            var candidate = view?.superview
            while let current = candidate {
                if let scrollView = current as? NSScrollView {
                    return scrollView
                }
                candidate = current.superview
            }
            return nil
        }

        private static func distanceToBottom(for scrollView: NSScrollView) -> CGFloat {
            guard let documentView = scrollView.documentView else { return 0 }
            let visibleRect = scrollView.contentView.documentVisibleRect
            return max(0, documentView.bounds.maxY - visibleRect.maxY)
        }
    }

    final class HostView: NSView {
        weak var coordinator: Coordinator?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            DispatchQueue.main.async { [weak self] in
                guard let self, let coordinator else { return }
                coordinator.attachIfNeeded(from: self)
            }
        }

        override func viewDidMoveToSuperview() {
            super.viewDidMoveToSuperview()
            DispatchQueue.main.async { [weak self] in
                guard let self, let coordinator else { return }
                coordinator.attachIfNeeded(from: self)
            }
        }
    }
}

@MainActor
private final class ChatConversationScrollDriver {
    weak var scrollView: NSScrollView?

    func updateScrollView(_ scrollView: NSScrollView?) {
        self.scrollView = scrollView
    }

    @discardableResult
    func scrollToBottom(animated: Bool) -> Bool {
        _ = clampToValidOffset()
        guard let scrollView,
              let documentView = scrollView.documentView else { return false }

        let clipView = scrollView.contentView
        let targetY: CGFloat
        if documentView.isFlipped {
            targetY = max(documentView.bounds.minY, documentView.bounds.maxY - clipView.bounds.height)
        } else {
            targetY = documentView.bounds.minY
        }
        let targetRect = clipView.constrainBoundsRect(
            NSRect(
                x: clipView.bounds.origin.x,
                y: targetY,
                width: clipView.bounds.width,
                height: clipView.bounds.height
            )
        )
        let targetPoint = targetRect.origin

        if abs(clipView.bounds.origin.y - targetPoint.y) < 0.5 {
            return true
        }

        if animated {
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.14
                context.timingFunction = CAMediaTimingFunction(name: .easeOut)
                clipView.animator().setBoundsOrigin(targetPoint)
            } completionHandler: {
                Task { @MainActor in
                    scrollView.reflectScrolledClipView(clipView)
                }
            }
        } else {
            clipView.scroll(to: targetPoint)
            scrollView.reflectScrolledClipView(clipView)
        }

        return true
    }

    @discardableResult
    func clampToValidOffset() -> Bool {
        guard let scrollView,
              scrollView.documentView != nil else { return false }

        let clipView = scrollView.contentView
        let currentRect = NSRect(
            x: clipView.bounds.origin.x,
            y: clipView.bounds.origin.y,
            width: clipView.bounds.width,
            height: clipView.bounds.height
        )
        let constrainedPoint = clipView.constrainBoundsRect(currentRect).origin

        if abs(clipView.bounds.origin.x - constrainedPoint.x) < 0.5,
           abs(clipView.bounds.origin.y - constrainedPoint.y) < 0.5 {
            return true
        }

        clipView.scroll(to: constrainedPoint)
        scrollView.reflectScrolledClipView(clipView)
        return true
    }
}

private struct ChatStartupStatusView: View {
    struct State: Equatable {
        var title: String
        var detail: String
    }

    let state: State

    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
                .controlSize(.regular)
                .tint(Brand.warning)
                .accessibilityHidden(true)
            Text(state.title)
                .font(BrandFont.subtitle())
                .foregroundStyle(Brand.typeHi)
                .multilineTextAlignment(.center)
            Text(state.detail)
                .font(.system(size: 11))
                .foregroundStyle(Brand.typeSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 40)
    }
}
