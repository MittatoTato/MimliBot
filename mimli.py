import discord
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import zipfile
import os
from keep_alive import keep_alive
keep_alive()
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
