import SwiftUI
import MTPLXAppCore

// MARK: - ChatView
//
// Root of the in-app chat primary mode. ContentView swaps the dashboard
// surface for this view when `router.primaryMode == .chat`. The chrome
// (TopChromeStrip + ConnectionIssueBanner) sits above; ChatView owns
// everything below.
//
// Layout:
//   HStack {
//     ChatSidebarView   // left rail, collapsible
//     VStack {
//       ChatHeaderView          // title + live TPS chip
//       ChatConversationView    // messages + streaming bubble
//       ChatComposerView        // composer pill at bottom
//     }
//   }

struct ChatView: View {
    @EnvironmentObject private var chatViewModel: ChatViewModel
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        HStack(spacing: 0) {
            ChatSidebarView(
                viewModel: chatViewModel,
                collapsed: $router.chatSidebarCollapsed
            )
            VStack(spacing: 0) {
                ChatHeaderView(
                    viewModel: chatViewModel,
                    sidebarCollapsed: $router.chatSidebarCollapsed
                )
                ChatConversationView(viewModel: chatViewModel)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                ChatComposerView(viewModel: chatViewModel)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 16)
                    .padding(.top, 8)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Brand.bgOuter)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            // Ensure there is a conversation to send into, so the user
            // can type immediately without having to click "+".
            if chatViewModel.current == nil {
                _ = chatViewModel.createNewConversation()
            }
        }
    }
}
