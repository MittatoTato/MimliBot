import discord
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import zipfile
import os
from keep_alive import keep_alive
keep_alive()
url = 'https://drive.google.com/file/d/1umM2MDzlvZbo7_850yqY5YWVXRjt2FHS/view?usp=sharing'
id='1umM2MDzlvZbo7_850yqY5YWVXRjt2FHS'
output = './yes.zip'

import sys
import requests


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)




if not os.path.exists(output): download_file_from_google_drive(id, output)
if not os.path.exists("./content/gpt2-finetuned"):
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(".")
special_token = "<|endoftext|>"
model = GPT2LMHeadModel.from_pretrained("./content/gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./content/gpt2-finetuned")


def generate_response(input_text):
    input_ids = tokenizer.encode(input_text + special_token,
                                 return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=60,
        num_return_sequences=1,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


intents = discord.Intents.all()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'We are {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('.'):
        try:
            await message.reply(
                generate_response(message.content[1:]).split("\n")[1])
        except Exception as e:
            print(e)


TOKEN=os.environ.get('token')
client.run(TOKEN)
