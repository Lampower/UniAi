import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram import types
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command

from analyzer import Analyzer
from config import BOT_TOKEN
from inlines.inlines import analyze_keyboard, back_to_stats_keyboard
from schemas.vk_schemas import UserModel
from vk_api import VkApi


async def main():

    logging.basicConfig(level=logging.INFO)

    if not BOT_TOKEN:
        raise RuntimeError("Не задан токен! Установи BOT_TOKEN или пропиши TOKEN в коде.")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start(message: types.Message):
        await message.answer("Привет! Я самый простой телеграм-бот 🤖")

    @dp.message(Command("help"))
    async def help(message: types.Message):
        await message.answer("Команды: \n/get_vk_info <user_id> - Получить информацию о пользователе ВКонтакте по его ID.")

    @dp.message(Command("get_vk_info"))
    async def get_vk_info(message: types.Message, state: FSMContext):
        vk_api = VkApi()
        analyzer = Analyzer()

        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) > 1:
            args = parts[1]
        else:
            args = None

        user_id = None
        try:
            user_id = int(args)
        except (TypeError, ValueError):
            await message.answer("Пожалуйста, укажи корректный числовой ID пользователя ВКонтакте.")
            return

        user_data = await vk_api.get_user_info(user_id)
        if not user_data:
            await message.answer("Не удалось получить информацию о пользователе ВКонтакте.")
            return
        await state.update_data({f"vk_user_data_{message.from_user.id}": user_data.model_dump()})

        analyzed_result = await analyzer.get_user_info_str(user_data)
        await state.update_data({f'vk_user_stats_{message.from_user.id}': analyzed_result})

        await message.answer(analyzed_result, reply_markup=analyze_keyboard)

    @dp.message()
    async def echo(message: types.Message):
        # Эхо-ответ — просто возвращаем текст
        if message.text:
            await message.answer(message.text)
        else:
            await message.answer("Я понимаю только текстовые сообщения 🙂")

    @dp.callback_query(lambda c: c.data == "analyze")
    async def analyze(callback_query: types.CallbackQuery, state: FSMContext):
        user_id = callback_query.from_user.id
        user_data = await get_context(state, user_id)
        if not user_data:
            await callback_query.message.answer("Нет данных для анализа. Сначала получи информацию о пользователе ВКонтакте.")
            return

        analyzer = Analyzer()
        analysis_result = await analyzer.analyze_vk_user(user_data)

        await callback_query.message.edit_text(analysis_result or "Не готово", reply_markup=back_to_stats_keyboard)

    @dp.callback_query(lambda c: c.data == "back_to_stats")
    async def back_to_stats(callback_query: types.CallbackQuery, state: FSMContext):
        stats = await get_stats(state, callback_query.from_user.id)
        await callback_query.message.edit_text(stats, reply_markup=analyze_keyboard)

    await dp.start_polling(bot)


async def get_context(state: FSMContext, user_id: int):
    context = await state.get_data()
    res = context.get(f"vk_user_data_{user_id}")
    return UserModel.model_validate(res) if res else None


async def get_stats(state: FSMContext, user_id: int) -> str | None:
    context = await state.get_data()
    res = context.get(f"vk_user_stats_{user_id}")
    return res if res else None


if __name__ == "__main__":
    asyncio.run(main())
