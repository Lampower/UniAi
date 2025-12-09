from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

analyze_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Провести анализ", callback_data="analyze")],
        [InlineKeyboardButton(text="Генерация расхождения", callback_data="generate_diff_image")]
    ]
)

back_to_stats_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Вернуться к статистике", callback_data="back_to_stats")]
    ]
)