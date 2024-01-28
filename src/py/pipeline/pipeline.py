from openai import OpenAI
from pathlib import Path
import os

client = OpenAI(api_key="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")

def whisperTranscript(filename): 
  audio_file= open("audio/input/"+filename, "rb")
  transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
  )
  return transcript

def gptResponse(transcript_text): 
    response = client.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=[
        {"role": "user", "content": transcript_text+"Do it in 3 sentences or less. If it's not a fitness or exercise related prompt, do not answer and tell us to give a fitness related prompt."},
    ]
    )
    response_message=(response.choices[0].message.content)
    return response_message
  
def tts(gptResponse):
  speech_file_path = "audio/output/prompt-output.mp3"
  response = client.audio.speech.create(
    model="tts-1",
    voice="shimmer",
    input=gptResponse
  )
  
  response.write_to_file(speech_file_path)
  
def getResponseFromInput(filename): #This costs me money every time it runs, do not run it too much. :(
    transcriptedText=whisperTranscript(filename)
    response=gptResponse(transcriptedText.text)
    tts(response)
