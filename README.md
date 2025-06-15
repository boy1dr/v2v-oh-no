# v2v-oh-no
World's least useful video-to-image-to-text-to-image-to-video converter, or v2i2t2i2v for short.

AI video-to-video conversion through the transformative process of image captioning and text-to-image generation

Usage: `python3 v2v.py -i input.mp4 -o output.mp4 -fps 3 --pip`

## Example Output
Check out what happens when you feed perfectly good footage through our AI fever dream machine:

[![v2v-oh-no Example](https://img.youtube.com/vi/MNMbtHhILHc/0.jpg)](https://youtube.com/shorts/MNMbtHhILHc?feature=share)

*Click to see the beautiful chaos in action*

## Program steps
1. Deconstruction (Video to Frames) 
2. AI Perception (Frames to Text)
3. AI Imagination (Text to Frames)
4. Reconstruction (Frames to Video)
5. Recombine (picture in picture and audio)

Optimised for low VRAM. For example RTX3080 10GB

## Installation
```bash
# Create a new folder and cd into it
mkdir v2v
cd v2v

# Setup and download this git repo
git clone https://github.com/boy1dr/v2v-oh-no.git
cd v2v-oh-no

# Create python virtual environment and install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install ffmpeg (required for audio support)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
```

## What This Does (And Why It Exists)

Ever wondered what would happen if you fed your vacation footage through the digital equivalent of a telephone game played by robots having fever dreams? Wonder no more! 

v2v-oh-no takes your perfectly good video and puts it through an unnecessarily convoluted AI pipeline that:
- Strips your video into individual frames (because apparently that's not destructive enough)
- Asks an AI to describe what it "sees" in each frame (spoiler: it's usually wrong)
- Feeds those descriptions to another AI that creates "artistic interpretations" (and we use that term very loosely)
- Stitches everything back together with the confidence of someone who definitely knows what they're doing

The result? A fever dream version of your original video that might vaguely resemble what you started with, if you squint really hard and have consumed the right amount of caffeine.

### Why Does This Exist?

Because someone had to answer the age-old question: "What if we made video editing as inefficient and unpredictable as possible?" 

This project exists in the proud tradition of "because we can" engineering, where the journey of unnecessary complexity is more important than the destination of actually useful results. It's like turning your bicycle into a Rube Goldberg machine that still gets you from point A to point B, but now involves seventeen AI models, a rubber duck, and at least three existential crises.

Perfect for:
- Impressing friends with your commitment to overthinking simple problems
- Creating "art" that makes people question their life choices
- Testing the limits of your GPU's patience
- Proving that just because AI can do something doesn't mean it should

**Warning:** Side effects may include uncontrollable laughter, deep philosophical questions about the nature of reality, and an inexplicable urge to show everyone your beautifully terrible creations.

*"It's not a bug, it's a feature!"* - The v2v-oh-no development philosophy

---

Joking aside, this was just a Sunday afternoon vibe project that I thought could be a nice starting point as a creative or artistic AI-fueled pipeline for LocalLLM enthusiasts.

Vibe coded under threat of a big stick by Claude Sonnet 4.0 (I didn't actually threaten it, but I hear that can help)
