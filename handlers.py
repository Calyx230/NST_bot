from begin import bot, dp
from config import TOKEN
import logging

from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram import types
from NST import NST

logging.basicConfig(level=logging.INFO)


class StyleTransfer(StatesGroup):
    style_img = State()
    content_img = State()


@dp.message_handler(commands=['start'])
async def start_answer(message: types.Message):
    text = "Привет!\n При помощи этого бота вы одной картинки на другую. Стиль включает " \
           "мазки кисти, выбор цвета, света и тени и т. п. \n Чтобы начать перенос стиля используйте " \
           "комаду /transfer \n Чтобы отменить перенос стиля воспользуйтесь командой /cancel"
    await message.answer(text=text)


@dp.message_handler(commands=['transfer'])
async def begin_style_transfer(message: types.Message):
    await StyleTransfer.style_img.set()
    await message.answer("Отправьте картинку, стиль с которой нужно взять за образец")
    

@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info('Cancelling state %r', current_state)
    await state.finish()
    await message.reply('Перенос стиля отменён.', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(content_types=['photo'], state=StyleTransfer.style_img)
async def load_style(message: types.Message, state: FSMContext):
    await message.photo[-1].download('style.jpg')
    await StyleTransfer.content_img.set()
    await message.answer("Отправьте фото, к которому нужно примеить стиль")
    

@dp.message_handler(content_types=['photo'], state=StyleTransfer.content_img)
async def load_content_and_train(message: types.Message, state: FSMContext):
    await message.photo[-1].download('content.jpg')
    await message.answer("Подождите немного...")
    nst = NST([37], [3, 10, 20, 30, 40])
    nst.train('content.jpg', 'style.jpg')
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    media.attach_photo(types.InputFile('result.jpg'), 'Готово!')
    await message.reply_media_group(media=media)
    await state.finish()
