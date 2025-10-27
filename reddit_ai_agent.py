import requests
import textwrap
import whisper_timestamped as whisper
import torch
import soundfile as sf
import re 
import os 
import random 
from moviepy.editor import (AudioFileClip, ImageClip, TextClip, 
                            CompositeVideoClip, CompositeAudioClip, 
                            VideoFileClip, vfx)
import moviepy.audio.fx.all as afx

# --- NEW GEMINI IMPORTS ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Warning: 'google-genai' not installed. Rewriting feature will be disabled.")
    # Define placeholder to prevent crash if library is missing
    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    genai = MockClient
    types = object
# --- END GEMINI IMPORTS ---

# --- Configuration ---
SUBREDDIT_LIST = [
    "nosleep", 
    "shortscarystories", 
    "UnresolvedMysteries", 
    "creepyencounters", 
    "letsnotmeet" 
]

OUTPUT_FILENAME = "voice.wav"
BACKGROUND_VIDEO = "minecraft_background.mp4" 
BACKGROUND_MUSIC = "background.mp3"
FINAL_VIDEO_FILENAME = "reddit_story.mp4"
MUSIC_VOLUME = 0.1 

# --- Goldilocks Filter Configuration (ADJUSTED FOR TTS STABILITY) ---
STORY_MIN_LENGTH = 300 
STORY_MAX_LENGTH = 1000 
REWRITE_TARGET_CHARS = 900 # Target length for the LLM to aim for

# --- VIDEO CONFIGURATION FOR 9:16 (YouTube Shorts) ---
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
TEXT_WIDTH = 900 # Reduced max text width for vertical orientation

print(f"--- Starting AI Agent ---")

# --- GEMINI REWRITING FUNCTION ---
def get_story_summary(title, full_text):
    """
    Uses the Gemini API to summarize a story to fit within the character limit.
    Requires GEMINI_API_KEY environment variable.
    """
    print("-> Connecting to Gemini API for story summarization...")
    
    client = None
    try:
        client = genai.Client()
        
        if not client:
             raise Exception("Client failed to initialize. GEMINI_API_KEY might be missing or invalid.")
        
        prompt = textwrap.dedent(f"""
            The following story titled '{title}' is too long for a short video.
            Your task is to **rewrite and summarize the entire story** into a compelling, 
            complete version that is a maximum of {REWRITE_TARGET_CHARS} characters long.
            **Maintain the original scary/creepy tone and all critical plot points.**
            Do not include any introductory phrases like 'Here is the summary' or 'The story is about'.
            
            STORY:
            {full_text}
        """)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        rewritten_story = response.text.strip()
        
        if len(rewritten_story) > STORY_MAX_LENGTH:
            print(f"-> WARNING: Rewritten story is still too long ({len(rewritten_story)} chars). Returning it.")
        
        return title, rewritten_story
    
    except Exception as e:
        print(f"-> ERROR during Gemini API call: {e}")
        return None, None
        
    finally:
        if client and hasattr(client, 'close'):
            try:
                client.close()
            except Exception as e:
                pass


# --- REDDIT FETCHING FUNCTION ---
def get_reddit_story(subreddit):
    """
    Fetches 'hot' stories from Reddit, filtering for length and rewriting if necessary.
    """
    print(f"Connecting to Reddit... (fetching r/{subreddit})")
    try:
        headers = {'User-Agent': 'MyRedditAgent/1.0'}
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error: Failed to connect to Reddit. Status Code: {response.status_code}")
            return None, None

        data = response.json()
        
        print(f"Filtering for stories between {STORY_MIN_LENGTH} and {STORY_MAX_LENGTH} chars...")
        
        for post_data in data['data']['children']:
            post = post_data['data']
            title = post['title']
            story_text = post['selftext']
            story_len = len(story_text)
            is_stickied = post['stickied']
            
            print(f"Checking post: '{title[:40]}...' (Length: {story_len}, Stickied: {is_stickied})")
            
            if is_stickied or "Monthly Open Forum" in title or "AITAOP" in title:
                print("--> Skipping (Sticky or unwanted content).")
                continue

            if STORY_MIN_LENGTH < story_len < STORY_MAX_LENGTH:
                print("Found a perfect length story!")
                return title, story_text
            
            elif story_len >= STORY_MAX_LENGTH:
                print(f"--> Story is too long ({story_len} chars). Attempting rewrite...")
                
                if story_len > 10000:
                    print("--> Skipping extremely long post (>10k chars) to save API calls.")
                    continue
                    
                rewritten_title, rewritten_text = get_story_summary(title, story_text)
                
                if rewritten_text:
                    print(f"-> AI rewrite successful! New length: {len(rewritten_text)} chars.")
                    return rewritten_title, rewritten_text
                else:
                    print("-> AI rewrite failed. Skipping post.")
            
            else:
                print("--> Skipping (Too short).")

        print(f"Error: Could not find a suitable story (checked top 25 posts and attempted rewrite).")
        return None, None

    except Exception as e:
        print(f"An error occurred while fetching/rewriting from Reddit: {e}")
        return None, None

# --- Main Execution ---
chosen_subreddit = random.choice(SUBREDDIT_LIST)
print(f"Targeting random source: reddit (r/{chosen_subreddit})")

title, self_text = get_reddit_story(chosen_subreddit)

if title and self_text:
    print(f"\nTitle: {title}")

    # --- TEXT-TO-SPEECH (Silero) ---
    print("\nStarting Text-to-Speech generation...")
    full_story = f"{title}. {self_text}"

    print("Cleaning text...")
    full_story = re.sub(r'http\S+', '', full_story)
    full_story = re.sub(r'[*_]', '', full_story)
    full_story = re.sub(r'&amp;', 'and', full_story)
    full_story = re.sub(r'\n{2,}', ' \n', full_story).strip()

    print("Loading AI model (this may take a moment)...")
    language = 'en'
    model_id = 'v3_en'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                             model='silero_tts',
                                             language=language,
                                             speaker=model_id,
                                             trust_repo=True)
        model.to(device)
    except Exception as e:
        print(f"Error loading Silero model: {e}")
        exit()

    print("Generating audio...")
    sample_rate = 48000
    speaker = 'en_42'

    audio = model.apply_tts(text=full_story,
                            speaker=speaker,
                            sample_rate=sample_rate)

    print(f"Saving audio to {OUTPUT_FILENAME}...")
    sf.write(OUTPUT_FILENAME, audio.numpy(), sample_rate)

    # --- GETTING TIMESTAMPS (Whisper) ---
    print("\nStarting AI transcription to get timestamps...")
    result = None
    try:
        print("Loading Whisper AI model...")
        whisper_model = whisper.load_model("small.en") 
        print("Transcribing audio (this may take a few moments)...")
        result = whisper.transcribe(whisper_model, OUTPUT_FILENAME, language="en")
        print("Transcription complete!")

    except Exception as e:
        print(f"--- ERROR DURING WHISPER TRANSCRIPTION ---")
        print(f"Error: {e}")
        print("Ensure 'ffmpeg' and the 'whisper_timestamped' package are installed correctly.")
    
    if result:
        print("Starting final video creation...")

        try:
            # 1. Load Audio Clips
            narration_clip = AudioFileClip(OUTPUT_FILENAME)
            print("Loading background music...")
            music_clip = AudioFileClip(BACKGROUND_MUSIC)
            
            # 2. Sweeten the Music
            print("Adjusting music (volume and loop)...")
            music_clip = music_clip.fx(afx.volumex, MUSIC_VOLUME).fx(afx.audio_loop, duration=narration_clip.duration)
            
            # 3. Combine Audio Tracks
            print("Mixing audio tracks...")
            final_audio = CompositeAudioClip([narration_clip, music_clip])
            video_duration = final_audio.duration

            # 4. Load Background Video (Uses BACKGROUND_VIDEO from config)
            print("Loading and resizing background video for 9:16 aspect ratio...")
            background_video_clip = VideoFileClip(BACKGROUND_VIDEO)
            background_video_clip = background_video_clip.without_audio()
            
            # --- START: MODIFIED 9:16 CROP LOGIC ---
            # Resize the clip so its HEIGHT matches the target height (1920)
            # This scales the video up, making it (for example) 3413x1920
            clip_resized = background_video_clip.fx(vfx.resize, height=VIDEO_HEIGHT)
            
            # Crop the horizontal center of the scaled-up clip
            # This takes the 1080px middle section, resulting in a 1080x1920 clip
            clip_cropped = clip_resized.fx(
                vfx.crop, 
                x_center=clip_resized.w / 2, 
                width=VIDEO_WIDTH
            )
            
            # Set this correctly cropped clip as the background
            background_video_clip = clip_cropped.set_opacity(0.8)
            # --- END: MODIFIED 9:16 CROP LOGIC ---

            # Loop or Trim the background video to match the audio duration
            if background_video_clip.duration < video_duration:
                background_video_clip = background_video_clip.fx(vfx.loop, duration=video_duration)
            else:
                background_video_clip = background_video_clip.set_duration(video_duration)


            # 5. Create STATIC Title Clip - Positioned near the top
            print("Creating title text...")
            title_clip = TextClip(
                title,
                fontsize=50,
                color='white',
                font='Arial-Bold',
                size=(TEXT_WIDTH, 200), 
                method='caption'
            )
            title_clip = title_clip.set_position(('center', VIDEO_HEIGHT * 0.05))
            title_clip = title_clip.set_duration(video_duration)

            # 6. Create DYNAMIC "Karaoke" Word Clips - Positioned centrally/lower
            print("Creating karaoke-style text clips (3-word chunks)...")
            all_word_clips = []
            
            for segment in result['segments']:
                words_list = segment['words']
                
                for i in range(0, len(words_list), 3):
                    chunk_data = words_list[i : i + 3] 
                    
                    if not chunk_data:
                        continue
                        
                    chunk_text = " ".join([word['text'] for word in chunk_data])
                    start_time = chunk_data[0]['start']
                    end_time = chunk_data[-1]['end']
                    duration = end_time - start_time
                    
                    chunk_clip = TextClip(
                        chunk_text,
                        fontsize=45,
                        color='white',
                        font='Arial-Bold',
                        size=(TEXT_WIDTH, 200),
                        method='caption' # Use caption for word wrapping
                    )
                    
                    chunk_clip = chunk_clip.set_start(start_time).set_duration(duration)
                    # Positioned lower in the vertical frame
                    chunk_clip = chunk_clip.set_position(('center', VIDEO_HEIGHT * 0.75))
                    
                    all_word_clips.append(chunk_clip)
            
            print(f"Generated {len(all_word_clips)} text chunk clips.")

            # 7. Composite (Layer) all the clips together
            print("Layering all clips...")
            all_layers = [background_video_clip, title_clip] + all_word_clips 
            
            # This is the critical line: set the final canvas size
            final_video = CompositeVideoClip(all_layers, size=(VIDEO_WIDTH, VIDEO_HEIGHT))

            # 8. Set the FINAL audio and write the file
            final_video = final_video.set_audio(final_audio)
            final_video.fps = 24
            
            print(f"Saving final video to {FINAL_VIDEO_FILENAME}...")
            final_video.write_videofile(FINAL_VIDEO_FILENAME, codec='libx264')
            
            print(f"Video generation complete: {FINAL_VIDEO_FILENAME}")

        except Exception as e:
            print(f"--- ERROR DURING VIDEO CREATION ---")
            print(f"Error: {e}")
            print("\nThis often happens if ImageMagick/FFmpeg is not installed")
        
        print(f"\n--- All done! ---")
        print(f"Your voice file is ready: {os.path.abspath(OUTPUT_FILENAME)}")
    
    else:
        print("Could not get timestamps. Stopping video generation.")

else:
    print("Could not get a story. Exiting.")