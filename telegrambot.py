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
    #model = torch.load('./models/fine_tuned_transferv2.pt')
    model.eval()
    print(model)

    image = Image.open('./'+file_name.strip('.ogg')+'.png')#.convert('RGB')
    loader = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((112,112)),
        torchvision.transforms.ToTensor(),
        ])
    image = loader(image)
    image.unsqueeze_(0)
    out = model(image)

    print('Probability of cough:'+str(out))
    if out > 0.5:
        bot.send_message(chat_id=update.message.chat_id,
                         text="That was not a cough")
    else:
        bot.send_message(chat_id=update.message.chat_id,
                         text="Checking for COVID-19...")

        image = Image.open('./'+file_name.strip('.ogg')+'.png').convert('RGB')
        loader = torchvision.transforms.Compose([
            #torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((112,112)),
            torchvision.transforms.ToTensor(),
            ])
        image = loader(image)
        image.unsqueeze_(0)

        model = torch.load('./models/fine_tuned_transfer_augmented.pt')
        model.eval()
        out = model(image)
        print('Probability of no COVID:'+str(out))
        if out > 0.8:
            bot.send_message(chat_id=update.message.chat_id,
                             text="You are okay! Nice cough bro")

        else:
            bot.send_message(chat_id=update.message.chat_id,
                             text="You might have COVID-19, you should go to a doctor")
    os.remove(file_name.strip('.ogg')+'.png')




# Informs user of an invalid command.
def unknown(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                     text="Sorry, I didn't understand.")

TOKEN = 'hehe'

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

print('I am alive')
oh_handler = MessageHandler(Filters.voice, is_cough_covid)
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('start', start, pass_user_data=True))
dispatcher.add_handler(oh_handler)
dispatcher.add_handler(MessageHandler(Filters.command, unknown))
updater.start_polling()
updater.idle()
# si me envian un audio, guardarlo en user_data
# comando test: esperar a que se reciba un audio, decir si es tos y si es tos mirar si es covid

dispatcher.add_handler(MessageHandler(Filters.command, unknown))

updater.start_polling()
