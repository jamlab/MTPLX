import SwiftUI
import MTPLXAppCore

// MARK: - StreamingAssistantView
//
// Same visual shell as `AssistantBubbleView` but reads live from
// `ChatViewModel`. Renders streamingReasoning inside a live ThinkingCard,
// streamingContent through stable document blocks, and any live
// `pendingToolTraces` as live AssistantTraceSurfaces at the bottom.

struct StreamingAssistantView: View {
    @ObservedObject var viewModel: ChatViewModel

    /// The reasoning card auto-collapses the moment any visible content
    /// has streamed. While the model is still in pure-thinking mode we
    /// show the expanded card with the line viewport; as soon as the
    /// first answer token arrives the card morphs to the compact
    /// "Thought" chip and the bubble appears below.
    private var contentHasStarted: Bool {
        viewModel.hasStreamingContent
    }

    private var reasoningHasStarted: Bool {
        viewModel.hasStreamingReasoning
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Reasoning lives OUTSIDE the bubble. Same compact/full
            // pattern Aphanes V2 uses. When content starts streaming
            // (`contentHasStarted`) the card shrinks to the inline
            // chip immediately.
            if reasoningHasStarted {
                StreamingThinkingCard(
                    document: viewModel.streamingReasoningDocument,
                    contentOverride: viewModel.streamingReasoning,
                    isStreaming: true,
                    isCompact: contentHasStarted
                )
                .frame(maxWidth: 576, alignment: .leading)
            }

            // Tool traces ALSO live OUTSIDE the bubble — they are
            // process metadata, not the assistant's spoken reply.
            // Treated identically to ThinkingCard: standalone cards
            // stacked above the eventual text bubble.
            if !viewModel.pendingToolTraces.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(viewModel.pendingToolTraces) { trace in
                        AssistantTraceSurface(
                            title: Self.traceTitle(for: trace.name),
                            subtitle: trace.subtitle,
                            detail: trace.detail,
                            activityLog: trace.activityLog,
                            systemName: Self.traceIcon(for: trace.name),
                            isCompact: false,
                            isLive: trace.status == .pending,
                            defaultExpanded: true
                        )
                    }
                }
                .frame(maxWidth: 576, alignment: .leading)
            }

            // The assistant bubble itself. Only rendered once there is
            // actually visible content to show. Otherwise we show the
            // pre-first-token pulse (unless reasoning or a tool trace
            // is already providing activity feedback).
            if contentHasStarted {
                HStack(alignment: .top, spacing: 0) {
                    StreamingAssistantMarkdownView(
                        document: viewModel.streamingContentDocument,
                        fallbackText: viewModel.streamingContent
                    )
                    .frame(maxWidth: 576, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 11)
                    .background(assistantBubbleBackground)
                    Spacer(minLength: 60)
                }
            } else if !reasoningHasStarted,
                viewModel.pendingToolTraces.isEmpty {
                // Pre-first-token: show pulse so the user sees activity
                // before either reasoning, a tool trace, or content
                // arrives.
                HStack(spacing: 8) {
                    ThinkingIndicatorDots()
                    Text(Self.phaseCaption(for: viewModel.streamingPhase))
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(Brand.typeSecondary)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(assistantBubbleBackground)
                .frame(maxWidth: 576, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var assistantBubbleBackground: some View {
        UnevenRoundedRectangle(
            topLeadingRadius: 14,
            bottomLeadingRadius: 4,
            bottomTrailingRadius: 14,
            topTrailingRadius: 14,
            style: .continuous
        )
        .fill(Brand.cardSurface)
        .overlay(
            UnevenRoundedRectangle(
                topLeadingRadius: 14,
                bottomLeadingRadius: 4,
                bottomTrailingRadius: 14,
                topTrailingRadius: 14,
                style: .continuous
            )
            .stroke(Brand.separator, lineWidth: 1)
        )
    }

    private static func phaseCaption(for phase: StreamingPhase) -> String {
        switch phase {
        case .idle: return ""
        case .thinking: return "Thinking…"
        case .generating: return "Generating…"
        case .searching: return "Searching the web…"
        case .reading: return "Reading sources…"
        case .answering: return "Answering…"
        case .finalizing: return "Finalising…"
        }
    }

    private static func traceTitle(for toolName: String) -> String {
        switch toolName {
        case "web_search": return "Web Search"
        case "fetch_url": return "Fetched Page"
        default: return toolName.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private static func traceIcon(for toolName: String) -> String {
        switch toolName {
        case "web_search": return "globe"
        case "fetch_url": return "link"
        default: return "wrench.and.screwdriver"
        }
    }
}
