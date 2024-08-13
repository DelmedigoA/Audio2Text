from whisper import Whisper
import gradio as gr
import requests

# Fetch the license text from GitHub
license_url = "https://raw.githubusercontent.com/openai/whisper/main/LICENSE"
response = requests.get(license_url)
license_text = response.text

# Initialize the Whisper model
whisper = Whisper()

# Path to your optional sample audio file
sample_audio = "test_data/test_audio.ogg"

# Create the Gradio interface
iface = gr.Interface(
    fn=whisper.predict,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio (OGG/WAV)"),  # Input for audio file
    outputs="text",  # Output will be the transcribed text
    title="(OpenAi) מתמלל שיחות הלשכה המרכזית לסטטיסטיקה",
    description="Upload or Record Audio",
    examples=[sample_audio],  # Optional sample audio file
    article = license_text
)

# Launch the app
iface.launch(debug=True, share=True)