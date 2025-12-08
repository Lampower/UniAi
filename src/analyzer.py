from dataclasses import dataclass
from datetime import datetime
import logging

# from read_file import ReadReadyModels
from schemas.vk_schemas import UserModel
from utils import Utils


@dataclass
class Analyzer:
    _excluded_fields = {'original_user_id', 'bot_score', 'is_bot'}

    async def get_user_info_str(self, user_data: UserModel) -> str:
        # Пример анализа данных
        analysis_result = """Анализ профиля пользователя:
>Имя пользователя: {name}
>Возраст: {age}
>Город: {city}
>Статус: {status}
>Пол: {gender}
"""
        age = None
        try:
            bdate = user_data.bdate
            if bdate:
                bdate = Utils.parse_bdate(bdate)
                today = datetime.today()
                age = today.year - bdate.year - ((today.month, today.day) < (bdate.month, bdate.day))
        except Exception as e:
            logging.error(e)
            pass

        return analysis_result.format_map({
            "name": user_data.first_name or "Не указано",
            "age": f"{age} лет" if age else "Не указано",
            "city": user_data.city.title if user_data.city else "Не указано",
            "status": user_data.status or "Не указано",
            "gender": Utils.get_sex(user_data.sex)
        })

    async def analyze_user_data(self, user_data: UserModel):
        # reader = ReadReadyModels()
        # model_data = reader.read()

        # profile_sensivity = await self.get_profile_sensivity(user_data)

        analysis_result = ""

        return analysis_result

    async def get_profile_sensivity(self, user_data: UserModel):
        dic = user_data.model_dump()
        user_fields = dic.keys()
        avg = 0
        k = 0
        for user_field in user_fields:
            if user_field in self._excluded_fields:
                continue
            k += 1
            field_data = dic[user_field]
            if field_data is not None:
                if isinstance(field_data, list):
                    for item in field_data:
                        if item:
                            avg += 1
                            break
                else:
                    avg += 1

        return avg / k if k > 0 else 0
