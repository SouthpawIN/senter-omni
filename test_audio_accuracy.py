#!/usr/bin/env python3
"""
Test AI Audio Recognition Accuracy

Tests the AI's ability to correctly identify different types of audio.
"""

import os
import sys
sys.path.insert(0, '.')

from senter_omni import SenterOmniChat

def test_audio_recognition():
    """Test AI's audio recognition accuracy"""
    print('🧪 Testing AI Audio Recognition Accuracy:')
    print('='*60)

    try:
        chat = SenterOmniChat()
        print('✅ Chat model loaded')
    except Exception as e:
        print(f'❌ Failed to load chat model: {e}')
        return

    test_files = [
        ('test_assets/pure_tone_440hz.wav', 'Pure 440Hz sine wave (musical tone)'),
        ('test_assets/white_noise.wav', 'White noise (random static)'),
        ('test_assets/silence.wav', 'Complete silence (no sound)'),
        ('test_assets/square_wave_440hz.wav', '440Hz square wave (digital tone)')
    ]

    for audio_file, description in test_files:
        if not os.path.exists(audio_file):
            print(f'⚠️ {audio_file} not found, skipping')
            continue

        print(f'\n🎵 Testing: {description}')
        print(f'File: {audio_file}')

        query = f'<user>Listen to this audio: <audio>{audio_file}</audio> What do you hear? Describe the sound precisely and be specific about what type of sound it is.</user>'

        print('🤖 AI Response:', end=' ')
        try:
            response = chat.generate_streaming([query])
            print(f'\n[Preview: {response[:120]}...]')

            print(f'\n🎯 Expected: {description}')
            print(f'❓ AI Said: {response[:100].replace(chr(10), " ")}'[:80])

            # Check if response matches expectation
            response_lower = response.lower()
            success = False

            if 'silence' in audio_file and ('silence' in response_lower or 'quiet' in response_lower or 'no sound' in response_lower):
                print('✅ Correctly identified silence!')
                success = True
            elif 'noise' in audio_file and ('noise' in response_lower or 'static' in response_lower or 'random' in response_lower):
                print('✅ Correctly identified noise!')
                success = True
            elif 'tone' in audio_file and ('tone' in response_lower or 'musical' in response_lower or 'note' in response_lower or 'pitch' in response_lower):
                print('✅ Correctly identified tone!')
                success = True

            if not success:
                print('❌ Response does not match audio content')
                print(f'   AI Response: {response[:200]}...')

        except Exception as e:
            print(f'❌ Error: {e}')

        print('-' * 50)

    print('\n📊 SUMMARY:')
    print('If the AI consistently describes all audio files as "voice speaking",')
    print('then the audio processing is not working correctly.')
    print('The audio files are clearly different: tone, noise, silence, square wave.')

if __name__ == "__main__":
    test_audio_recognition()
