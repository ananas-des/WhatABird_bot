# WhatABird_bot 🐤

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)

Repository with Python code for Birds species image classification and for Telegram Bot. Also it contains [Jupyter Notebook](./source/birds_model.ipynb) with [ResNet-50](https://iq.opengenus.org/resnet50-architecture/) model training and [presentation](./What_A_Bird_Bot.pdf) with some descriptions. 
For model training, the data set of 525 bird species from [kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species?resource=download) was used. It contains three sunsets: *84635* training images, *2625* validation images, and *2625* test images (5 images per species for both subsets). 

Follow the guidelines to set up and launch your Telegram Bot.

## System

- <img src="https://github.com/simple-icons/simple-icons/raw/develop/icons/ubuntu.svg" style="height: 25px; width:25px;"/> **Ubuntu** v20.04.6 LTS
- <img src="https://github.com/simple-icons/simple-icons/raw/develop/icons/python.svg" style="height: 25px; width:25px;"/> **Python** v3.9.13
- <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/c44c0a776bfab5924f0ecef7e595be8d8afda2be/icons/pytorch.svg" style="height: 25px; width:25px;"/> **PyTorch** v2.0.1
- <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/c44c0a776bfab5924f0ecef7e595be8d8afda2be/icons/nvidia.svg" style="height: 25px; width:25px;"/> **CUDA** v11.8
- <img src="https://github.com/simple-icons/simple-icons/raw/develop/icons/gnubash.svg" style="height: 25px; width:25px;"/> **bash**

## Creating a Telegram Bot

[Here](https://www.freecodecamp.org/news/how-to-create-a-telegram-bot-using-python/) the instructions for creating automated chatbots.

1. Getting Bot token
- search for @botfather in Telegram;
- start a conversation with BotFather by clicking on the Start button;
- type `/newbot`, and follow the prompts to set up a new bot. The BotFather will give you a **token** that you will use to authenticate your bot and grant it access to the Telegram API.

**Note**: Make sure you *store the token securely*. Anyone with your token access can easily manipulate your bot.

2. Cloning repository and setting up an environment
- clone this repository using an **SSH** key

```
$ git clone git@github.com:AnasZol/WhatABird_bot.git
```

- navigate to local repository

```
$ cd WhatABird_bot/
```

- create and activate virtual environment with Python v3.9.13 using `conda`

```
$ conda create -n bird_bot python==3.9.13
$ conda activate bird_bot
```

- install Python packages from `requirements.txt` using `pip`

```
$ pip install -r requirements.txt
```

- install `PyTorch` package as [recommended by developer](https://pytorch.org/get-started/locally/) for your preferences (OS, compute platform, *etc.*). For example, for `Linux` OS, `pip` packager, and `CUDA 11.8` compute platform, use the following command 

```
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- navigate to `source` folder

```
$ cd source/
```

- open your favorite code editor and create a `.env` file to store your token generated by The BotFather for your bot authentification and access to the Telegram API. Paste the following into file

```
$ export BOT_TOKEN=your-bot-token-here
```

- read the environment variables from the `.env` file

```
$ source .env
```

## Running Telegram bot

- run the Telegram bot main program via executing [bird_bot.py](./source/bird_bot.py) script. As the result, `aiogram` dispatcher starts pooling a response from the Telegram

```
$ python bird_bot.py
```

## Telegram bot usage for Birds species image classification

Telegram bot accepts both birds images and their links. [Here](./bot_instructions.pdf) the instruction for bot usage. Generally, all you need to copy bird image or link, paste it into message box, and send to bot. The bot will *respond a text message* with the predicted bird species and prediction accuracy. In the case when accuracy is below 60%, the bot will send the top 3 best predictions. The bot will also *generate an image* based on the original one, adding a title with a bird species according to the best prediction.

Try it yourself!
