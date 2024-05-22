import discord
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import zipfile
import os
import requests
from keep_alive import keep_alive
keep_alive()
url = 'https://drive.google.com/uc?id=1umM2MDzlvZbo7_850yqY5YWVXRjt2FHS'
output = './yes.zip'

def download_file_from_google_drive(shareable_link, destination):
    # Extract the file ID from the shareable link
    file_id = shareable_link.split('/d/')[1].split('/')[0]

    # Construct the direct download URL
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # Make a GET request to fetch the file
    response = requests.get(download_url, stream=True)

    # Save the file content to the destination file
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)

    print(f'File downloaded successfully to {destination}')
if not os.path.exists(output): download_file_from_google_drive(url, output)
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
