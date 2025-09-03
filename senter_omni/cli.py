#!/usr/bin/env python3
"""
Senter-Omni CLI Interface

Command-line interface for the Senter-Omni chat model.
"""

import sys
import json
from typing import Dict, Any, Optional
from .core import SenterOmniChat


def interactive_chat():
    """Run interactive chat interface"""
    print("🎭 Senter-Omni Interactive Chat")
    print("=" * 50)
    print("Features:")
    print("• XML tags: <system>, <user>, <assistant>")
    print("• Multimodal: <image>, <audio>, <video>")
    print("• Streaming responses")
    print("• Conversation history")
    print("=" * 50)

    try:
        chat = SenterOmniChat()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure the model is available at models/huggingface/senter-omni-lora")
        return

    print("\n💡 Example usage:")
    print("<user>Hello! Tell me about yourself.</user>")
    print("<system>You are a helpful AI assistant.</system>")
    print("<user>I have this image: <image>sunset over mountains</image> What do you see?</user>")

    conversation_history = []

    while True:
        try:
            user_input = input("\n🎤 Your message (or 'quit' to exit): ").strip()

            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'help':
                print("\n📚 Commands:")
                print("  help     - Show this help")
                print("  clear    - Clear conversation")
                print("  history  - View conversation history")
                print("  quit     - Exit")
                print("\n📝 XML Tags:")
                print("  <system>content</system>     - System prompt")
                print("  <user>content</user>         - User message")
                print("  <assistant>content</assistant> - Assistant response")
                print("  <image>content</image>       - Image description")
                print("  <audio>content</audio>       - Audio description")
                print("  <video>content</video>       - Video description")
                continue

            if user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation cleared!")
                continue

            if user_input.lower() == 'history':
                if conversation_history:
                    print("\n📜 Conversation History:")
                    for i, msg in enumerate(conversation_history, 1):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:100]
                        print(f"  {i}. {role}: {content}...")
                else:
                    print("📜 No conversation history yet.")
                continue

            if not user_input:
                continue

            # Add user message to history
            user_msg = chat.parse_xml_message(user_input)
            conversation_history.append(user_msg)

            # Generate response
            print("\n🤖 Senter-Omni:", end=" ")
            response = chat.generate_streaming(conversation_history)
            print()

            # Add assistant response to history
            assistant_msg = {
                "role": "assistant",
                "content": response
            }
            conversation_history.append(assistant_msg)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def api_chat(messages, **kwargs):
    """API-style chat completion"""
    try:
        chat = SenterOmniChat()
        return chat.chat_completion(messages, kwargs)
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main CLI entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--api":
            # API mode - expect JSON input
            try:
                if len(sys.argv) > 2:
                    # JSON file input
                    with open(sys.argv[2], 'r') as f:
                        data = json.load(f)
                else:
                    # JSON stdin input
                    data = json.load(sys.stdin)

                messages = data.get('messages', [])
                params = data.get('parameters', {})

                result = api_chat(messages, **params)
                print(json.dumps(result, indent=2))

            except Exception as e:
                print(json.dumps({"error": str(e)}, indent=2))

        elif sys.argv[1] == "--help":
            print("Senter-Omni CLI")
            print("Usage:")
            print("  senter-omni                # Interactive chat")
            print("  senter-omni --api          # API mode (JSON stdin)")
            print("  senter-omni --api file.json # API mode (JSON file)")
            print("  senter-omni --help         # Show this help")

        else:
            print("Unknown option. Use --help for usage information.")
    else:
        # Default to interactive mode
        interactive_chat()


if __name__ == "__main__":
    main()
