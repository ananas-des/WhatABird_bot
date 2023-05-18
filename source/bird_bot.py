import logging
import os
from pathlib import Path
import dotenv
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ParseMode
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ParseMode
from io import BytesIO
from PIL import Image


import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms


import pickle
import numpy as np

from matplotlib import colors, pyplot as plt
import seaborn as sns

import requests
import io


dotenv_file = Path(".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)

API_TOKEN = os.environ["BOT_TOKEN"]

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN, parse_mode="Markdown")
dp = Dispatcher(bot)

_logger = logging.getLogger(__name__)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESCALE_SIZE = 232

model_resnet = models.resnet50(weights=None).to(DEVICE)
for i, child in enumerate(model_resnet.children()):
    if i not in [9]:
        for param in child.parameters():
            param.requires_grad = False

model_resnet.fc = nn.Sequential(nn.Linear(2048, 525))
model_resnet.load_state_dict(torch.load("birds/model_weights_50_best.pth"))
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))


@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    await message.reply(
        F"Alloha, {message.from_user.first_name}! I'm *WhatBirdBot* \U0001F424 I know 525 species of birds. Let's look at your images! "
        "Send me picture with a bird or a link. And I will try to find out What A Bird is in your picture. "
        "It's my job, you know.", parse_mode=ParseMode.MARKDOWN
    )


@dp.message_handler(commands=["hi", "hello"])
async def echo(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="Image \U0001F4F7"),
            types.KeyboardButton(text="Link \U0001F310")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await message.answer("What do you have? Image or link?", reply_markup=keyboard)
    buttons_answer(message)
    

@dp.message_handler(content_types=['text'])
async def buttons_answer(message):
    if message.text == "Image \U0001F4F7":
        await message.reply("Cool! I'm waiting for your image \U0001F51C")
    elif message.text == "Link \U0001F310":
        await message.reply("Link, of course! Make sure you copied the correct link \U0001F517")
    

@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def echo_image(message: types.Message):
    if message.photo:
        photo_path = BytesIO()
        photo = await message.photo[-1].download(photo_path)
    
        photo = Image.open(photo)

        # Reply with the photo and caption
        await message.answer(
            text="Got it! I recieved your image and started to find keys in my Birds photoalbom"
        )
        # Perform bird prediction
        # bird = predict(img_path=str(photo_path)"
        # сюда вкрутить предсказание

        # Reply with the bird prediction
        await message.reply(text=f"I think your bird is: {bird}")

        # # Remove the saved photo
        photo_path.unlink()
    else:
        await message.reply(text="I am not sure... Can you repeat, please?")
        

# может пригодиться
# class Links(StatesGroup):
#     get = State()
    
    
# @dp.message_handler(commands=["link"])
# async def collect_link_start(message: Message, state: FSMContext):
#     await message.reply("Send me links.\nTo finish, type or click /done")
#     await Links.get.set()
#     # You can use your own saving logic instead, this is just an example
#     await state.update_data(links=[])
    
    
# # You can set more advanced filter for links, `text_startswith` is just an example
# @dp.message_handler(text_startswith=["http" or "https"], state=Links.get)
# async def collect_links(message: Message, state: FSMContext):
#     # You can use your own saving logic instead, this is just an example
#     data = await state.get_data()
#     links = data["links"]
#     links.append(message.text)
#     await state.update_data(links=links)

#     await message.reply("Got it! Starting downloading and processing your bird image.")

# # This handler is a fallback in case user didn't provide valid link
# @dp.message_handler(state=Links.get)
# async def invalid_link(message: Message):
#     await message.reply("This doesn't look like a valid link!")


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)