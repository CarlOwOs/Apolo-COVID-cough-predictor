import telegram
import random
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import Filters
from telegram.ext.dispatcher import run_async
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchvision
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image


# Starts the bot.
def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                     text="*Welcome I'm the cough doctor*",
                     parse_mode=telegram.ParseMode.MARKDOWN)
    bot.send_message(chat_id=update.message.chat_id,
                     text="Send a voice message with your cough and I will tell you if you are OK",
                     parse_mode=telegram.ParseMode.MARKDOWN)


def help(bot, update):
    intro = "*BotBicing* can execute the following commands:\n"
    start = ("/start \n Starts a new session with the bot."
             "*WARNING:* The current graph will be erased.\n")
    authors = "/authors \n Provides authors' names and e-mails.\n"
    graph = "/graph \n Generates a graph using the distance provided.\n"
    nodes = "/nodes \n Returns the number of nodes of  graph.\n"
    edges = "/edges \n Returns the number of edges of  graph.\n"
    components = ("/components \n Returns the number of connex "
                  "components of the graph.\n")
    plotgraph = ("/plotgraph \n Returns a .png image, representing "
                 "all the stations of the graph and their respective"
                 " connections.\n")
    route = ("/route \n Returns a .png image, with the shortest"
             " path between two coordinates. \n")
    distribute = ("/distribute \n Returns the cost of distributing"
                  " bikes given two parameters. \n")
    bot.send_message(chat_id=update.message.chat_id,
                     text=intro + start + authors + graph + nodes + edges +
                     components + plotgraph + route + distribute,
                     parse_mode=telegram.ParseMode.MARKDOWN)


# Authors of the project and their respective e-mails.
def authors(bot, update):
    version = "*Cough Bot 0.1*"
    first_author = "Victor Novelle Moriano: victor.novelle@est.fib.upc.edu"
    second_author = ("Carlos Hurtado Comin: carlos.hurtado"
                     ".comin@est.fib.upc.edu")
    licence = "_Univeristat Politecnica de Catalunya, 2019_"
    bot.send_message(chat_id=update.message.chat_id,
                     text=version + "\n" + first_author + "\n" +
                     second_author + "\n" + licence,
                     parse_mode=telegram.ParseMode.MARKDOWN)

@run_async
def is_cough_covid(bot, update):
    print('received audio')
    chat_id = update.message.chat.id

    file_name = str(chat_id) + '_' + str(update.message.from_user.id) + str(update.message.message_id) + '.ogg'

    update.message.voice.get_file().download(file_name)

    waveform, sample_rate = torchaudio.load('./'+file_name)
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    specgram_resize = torchvision.transforms.Resize((224,224))(specgram)
    plt.figure(frameon=False)
    plt.axis('off')
    specgram_resize += torch.ones(list(specgram_resize.shape))*1e-12
    plt.imshow(specgram_resize.log2()[0,:,:].numpy(), cmap='gray')
    plt.savefig('./'+file_name.strip('.ogg')+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    os.remove(file_name)

    model = torch.load('./models/cough_detection_model_from_scratch.pt')
    model.eval()
    print(model)

    image = Image.open('./'+file_name.strip('.ogg')+'.png')
    loader = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((112,112)),
        torchvision.transforms.ToTensor(),
        ])
    image = loader(image)
    image.unsqueeze_(0)
    out = model(image)

    os.remove(file_name.strip('.ogg')+'.png')

    print(out)
    if out > 0.5:
        bot.send_message(chat_id=update.message.chat_id,
                         text="That was not a cough")
    else:
        bot.send_message(chat_id=update.message.chat_id,
                         text="Checking for COVID-19...")





# Informs user of an invalid command.
def unknown(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                     text="Sorry, I didn't understand.")

TOKEN = '1493340494:AAETZOoevOLzIfsr4z-J1sGgXGkYNpnTEiU'

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('I am alive')
oh_handler = MessageHandler(Filters.voice, is_cough_covid)
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('start', start, pass_user_data=True))
dispatcher.add_handler(CommandHandler('help', help))
dispatcher.add_handler(CommandHandler('authors', authors))
dispatcher.add_handler(oh_handler)
dispatcher.add_handler(MessageHandler(Filters.command, unknown))
updater.start_polling()
updater.idle()
# si me envian un audio, guardarlo en user_data
# comando test: esperar a que se reciba un audio, decir si es tos y si es tos mirar si es covid

dispatcher.add_handler(MessageHandler(Filters.command, unknown))

updater.start_polling()
