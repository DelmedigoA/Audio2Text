import torch
from transformers import AutoModelForSpeechSeq2Seq
from transformers import AutoProcessor
import librosa
import numpy as np
from transformers import pipeline


class Whisper:
  def __init__(self, model_id =  "openai/whisper-large-v3", gpu = True):
    self.model_id = model_id
    self.device = "cuda:0" if torch.cuda.is_available() and gpu else "cpu"
    self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    self.gpu = gpu
    self.model = None
    self.processor = AutoProcessor.from_pretrained(model_id)
    self.pipe = None

  def load(self):
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id,
                                                  torch_dtype = self.torch_dtype,
                                                  low_cpu_mem_usage = True if self.gpu else False,
                                                  use_safetensors = True)
    self.model.to(self.device)
    self.pipe = pipeline(
        "automatic-speech-recognition",
        model = self.model,
        tokenizer = self.processor.tokenizer,
        feature_extractor = self.processor.feature_extractor,
        max_new_tokens = 128,
        chunk_length_s = 30,
        batch_size = 16,
        return_timestamps = True,
        torch_dtype = self.torch_dtype,
        device = self.device
    )
  
    return None 
  
  def audio_processor(self, audio_path = '/Audio2Text/test_data/test_audio.ogg'):
    
    audio, sr = librosa.load(audio_path, sr=16000)

    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    audio = librosa.util.normalize(audio)
    return audio

  def predict(self, audio):
    audio_as_vector = self.audio_processor(audio)
    data_dict = dict(path='audio_to_predict', array = audio_as_vector, sampling_rate = 16_000)
    result = self.pipe(data_dict)
    return result["text"].strip()
  
if __name__ == '__main__':
  whisper = Whisper()
  whisper.load()
  print(whisper.predict('/content/Audio2Text/test_data/test_audio.ogg'))