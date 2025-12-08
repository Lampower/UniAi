from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

analyze_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Провести анализ", callback_data="analyze")]
    ]
)

back_to_stats_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Вернуться к статистике", callback_data="back_to_stats")]
    ]
)