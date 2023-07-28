import logging
import os
import dotenv
from pathlib import Path
import requests

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ParseMode
from aiogram.dispatcher.filters import Text
from aiogram.types import ParseMode

import io
from PIL import Image
import pickle

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import numpy as np
from matplotlib import colors, pyplot as plt
import seaborn as sns


# loading bot token from .env
dotenv_file = Path(".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)

# loading bot token into variable
API_TOKEN = os.environ["BOT_TOKEN"]

# for logging program
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# creating bot and dispatcher
bot = Bot(token=API_TOKEN, parse_mode="MarkdownV2")
dp = Dispatcher(bot)

# setting up classifiaction model
# loading pretrained ResNet50 model from torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESCALE_SIZE = 232
model_resnet = models.resnet50(weights=None).to(DEVICE)
for i, child in enumerate(model_resnet.children()):
    if i not in [9]:
        for param in child.parameters():
            param.requires_grad = False

# loading updated model weights and label encoder data
model_resnet.fc = nn.Sequential(nn.Linear(2048, 525))
model_resnet.load_state_dict(torch.load("model/model_weights_50_best.pth", map_location=torch.device(DEVICE)))
label_encoder = pickle.load(open("model/label_encoder.pkl", 'rb'))


# functions for commutication with telegram bot
@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    '''A function for greeting user'''
    
    
    await message.reply(
        F"Alloha, {message.from_user.first_name}! I'm *WhatBirdBot* \U0001F424 I know 525 species of birds. Let's look at your images! "
        "Send me picture with a bird or a link. And I will try to find out What A Bird is in your picture. "
        "It's my job, you know.", parse_mode=ParseMode.MARKDOWN
    )
    kb = [
        [
            types.KeyboardButton(text="Yes!")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await message.answer("Do you want to start?", reply_markup=keyboard)
    buttons_answer(message)
    

@dp.message_handler(content_types=['text'])
async def buttons_answer(message):
    '''A function for creating reply buttons with image data type options (image format or link).
    If user's input is link, the function downloads it in buffer, makes prediction sends the image with predicted
    species back to user'''
    
    
    if message.text == "Yes!":
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
    elif message.text == "Image \U0001F4F7":
        await message.reply("Cool\! I'm waiting for your image \U0001F51C")
    elif message.text == "Link \U0001F310":
        await message.reply("Link, of course\! Make sure you copied the correct link \U0001F517")
    elif message.text.startswith("http"):
        try:
            response = requests.get(message.text).content
            image = Image.open(io.BytesIO(response)).convert("RGB")
            await message.reply("Got it\! Starting downloading and processing your bird image \U0001F4E0")
            bird, image_return = predict_image(model_resnet, image)
            await message.reply(f"I think your bird is:\n\n{bird} \U0001F425")
            await message.answer_photo(image_return)
            
            kb = [
                [
                types.KeyboardButton(text="Yes!")
                ],
            ]
            keyboard = types.ReplyKeyboardMarkup(
                keyboard=kb,
                resize_keyboard=True,
                one_time_keyboard=True
            )
            await message.answer("Want to try again?", reply_markup=keyboard)
            buttons_answer(message)
        except:
            await message.reply(text=f"Your link is invalid \U0001F4DB Can you repeat, please?")
            
    else:
        await message.reply(text=f"Sorry, I don't understand \U0001F614")
        kb = [
            [
            types.KeyboardButton(text="Yes!")
            ],
        ]
        keyboard = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            one_time_keyboard=True
        )
        await message.answer("Want to try again?", reply_markup=keyboard)
        buttons_answer(message)
            
            
@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def echo_image(message: types.Message):
    '''A function for processing user's image, making prediction and sending the image with predicted
    species back to user'''
    
    
    if message.photo:
        photo_path = io.BytesIO()
        photo = await message.photo[-1].download(photo_path)
        photo = Image.open(photo).convert("RGB")

        # Reply with the photo and caption
        await message.answer(
            text="Got it\! I have recieved your image and started to find keys in my Birds photo album \U0001F4DA"
        )
        
        # Perform bird species prediction
        bird, image_return = predict_image(model_resnet, photo)

        # Reply with the bird prediction
        await message.reply(text=f"I think your bird is:\n\n{bird} \U0001F426")
        await message.answer_photo(image_return)
        kb = [
            [
            types.KeyboardButton(text="Yes!")
            ],
        ]
        keyboard = types.ReplyKeyboardMarkup(
            keyboard=kb,
            resize_keyboard=True,
            one_time_keyboard=True
        )
        await message.answer("Want to try again?", reply_markup=keyboard)
        buttons_answer(message)
    else:
        await message.reply(text="I am not sure... Can you repeat, please?")


def predict_image(model, image, device=DEVICE):
    '''The function predict_image() preprocesses image, predicts its class, creates updated image
    with the most probable bird species and text message with prediction and its accuracy. 
    In case of accuracy <60%, the text message contains the top three bird species by its prediction accuracy.
    
    Parameters:
        model: classification model
        image: user's bird image
        device: CUDA if it is available
    
    Retuns:
        predicted_text (str): text message for user
        buffer.getvalue(): updated image from buffer for user
    '''
    
    
    model = model.to(device)
    transform = transforms.Compose([
                transforms.Resize(size=(RESCALE_SIZE,RESCALE_SIZE)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_trans = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        inputs = image_trans.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        
    max_prob = probs.max() * 100
    y_pred_argmax = np.argmax(probs, -1)
    pred_class = label_encoder.classes_[y_pred_argmax]
    if max_prob > 60:
        y_pred_argmax = np.argmax(probs, -1)
        pred_class = label_encoder.classes_[y_pred_argmax]
        predicted_text = "{}: {:.0f}%".format(pred_class[0], max_prob)
        image_title = predicted_text
    else:
        idx = np.argsort(probs[-1])[-3:][::-1]
        y_preds = [probs[-1][i] for i in idx]
        predicted_text = []
        for i, y_pred in zip(idx, y_preds):
            preds_class = label_encoder.classes_[i]
            predicted_text.append("{}: {:.0f}%".format(preds_class, y_pred * 100))
        image_title = predicted_text[0]
        predicted_text = "\n".join(predicted_text)
    
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    ax.imshow(image)
    ax.set_title(image_title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='PNG')
    return predicted_text, buffer.getvalue()


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)