from config import TOKEN
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage

bot = Bot(TOKEN, parse_mode=types.ParseMode.HTML)
bot.set_my_commands(['start', 'transfer'])

storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
