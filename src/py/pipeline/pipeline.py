from openai import OpenAI
from pathlib import Path
import os
import time

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
        {"role": "user", "content": "Answer the following question very quickly using only the necessary exercises out of lunges, squats, pull-ups, planks: " + transcript_text},
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
    seconds=time.time()
    transcriptedText=whisperTranscript(filename)
    print(time.time()-seconds)
    seconds=time.time()
    response=gptResponse(transcriptedText.text)
    print(time.time()-seconds)
    seconds=time.time()
    tts(response)
    print(time.time()-seconds)
    return response
