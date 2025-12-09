import logging
import pandas as pd

from dataclasses import dataclass
from datetime import datetime

from read_file import ReadReadyModels
from schemas.vk_schemas import UserModel
from vk_api import VkApi
from utils import Utils


@dataclass
class Analyzer:
    _excluded_fields = {'original_user_id', 'bot_score', 'is_bot'}

    async def get_user_info_str(self, user_data: UserModel) -> str:
        # Пример анализа данных
        analysis_result = """Статистика профиля пользователя:
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
        reader = ReadReadyModels()
        model_data = reader.read()

        sex_name = Utils.get_sex(user_data.sex)

        row = model_data[model_data["sex"] == user_data.sex]

        if user_data.sex == 0:
            row = model_data[model_data["sex"] == 2]

        profile_density = await self.get_profile_sensivity(user_data)

        analysis_result = f"""Данные анализа профиля пользователя с средним пользователем с полом: {sex_name}:
>Заполненость профиля: {profile_density:.2f}, {(profile_density - row["profile_density"]) / row["profile_density"] * 100:.1f}% разница со средним пользователем
"""

        user_age = await self.get_age(user_data)
        if user_age is not None:
            analysis_result += f">Возраст: {user_age} лет, {(user_age - row['age']) / row['age'] * 100:.1f}%\n"

        has_photo_density = row["has_photo"]
        if user_data.has_photo:
            analysis_result += f">Наличие фотографий: Да, больше чем у {100 - has_photo_density * 100:.1f}% пользователей!!!\n"
        else:
            analysis_result += f">Наличие фотографий: Нет, вы входите в {100 - has_photo_density * 100:.1f}% пользователей, у которых нет фотографий\n"

        if user_data.followers_count:
            followers_count = user_data.followers_count
            percentage = (followers_count - row['followers_count']) / row['followers_count'] * 100
            analysis_result += f">Количество подписчиков: {followers_count}, {percentage:.1f}% разница со средним пользователем\n"

        if user_data.relation:
            analysis_result += f">Имеет данные об отношении что больше чем у \n"

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

    async def draw_schema(self):
        pass

    async def get_age(self, user_data: UserModel) -> int | None:
        try:
            bdate = user_data.bdate
            if bdate:
                bdate = Utils.parse_bdate(bdate)
                today = datetime.today()
                return today.year - bdate.year - ((today.month, today.day) < (bdate.month, bdate.day))
        except Exception as e:
            logging.error(e)
        return None
    

    async def analyze_vk_user(
        self,
        user: UserModel,
    ) -> str:
        
        reader = ReadReadyModels()
        df = reader.read()

        if "sex" in df.columns:
            avg_row = df[df["sex"] == user.sex]
        else:
            avg_row = df.loc[user.sex]

        sex_name = Utils.get_sex(user.sex)

        lines = []

        profile_density = await self.get_profile_sensivity(user)
        friends_count = await VkApi().get_friends_count(user.id)
        user.profile_density = profile_density
        user.age = await self.get_age(user)
        user.friends_count = friends_count

        def fmt_num(x, digits=1):
            if pd.isna(x):
                return "нет данных"
            return f"{x:.{digits}f}"

        def compare_numeric(col_user, col_df, label, unit=""):

            u = float(getattr(user, col_user, float('nan')))
            a = float(avg_row[col_df])

            if pd.isna(u) or pd.isna(a):
                return

            diff = u - a
            if abs(a) > 1e-9:
                diff_pct = diff / a * 100
                sign = "больше" if diff > 0 else "меньше"
                lines.append(
                    f"{label}: у вас {fmt_num(u,0)}{unit}, "
                    f"в среднем у {sex_name} — {fmt_num(a,0)}{unit} "
                    f"({abs(diff_pct):.1f}% {sign} среднего)."
                )
            else:
                lines.append(
                    f"{label}: у вас {fmt_num(u,0)}{unit}, "
                    f"для среднего значения по полу данных почти нет."
                )

        def compare_prob(col_user, col_df, label):
            u = float(getattr(user, col_user, float('nan')))
            a = float(avg_row[col_df])

            if pd.isna(u) or pd.isna(a):
                return

            u_p = u * 100
            a_p = a * 100
            diff = u_p - a_p
            sign = "выше" if diff > 0 else "ниже"
            lines.append(
                f"{label}: у вас {u_p:.1f}% против среднего {a_p:.1f}% "
                f"({abs(diff):.1f} п.п. {sign} среднего)."
            )

        def compare_bool_prob(col_user, col_df, label, positive_text="есть", negative_text="нет"):
            """
            user[col] — 0/1, avg_row[col] — доля от 0 до 1.
            """
            if col_user not in user or col_df not in avg_row:
                return

            u = int(getattr(user, col_user, 0))
            a = float(avg_row[col_df])
            if pd.isna(a):
                return

            avg_p = a * 100
            has_txt = positive_text if u == 1 else negative_text
            lines.append(
                f"{label}: у {sex_name} в {avg_p:.1f}% случаев {positive_text}, "
                f"у вас — {has_txt}."
            )

        compare_numeric("followers_count", "followers_count", "Подписчики", " чел.")
        compare_numeric("friends_count", "count", "Друзья", " чел.")
        compare_numeric("age", "age", "Возраст", " лет")
        compare_prob("profile_density", "profile_density", "Заполненность профиля")
        compare_bool_prob("relation", "relation", "Доля пользователей в отношениях/браке", "указаны какие либо отношения", "отношения не указаны")

        compare_bool_prob("has_photo","has_photo", "Наличие аватарки")

        # Итоговая интерпретация (очень грубая, по количеству «выше среднего» признаков)
        higher = sum("больше среднего" in s or "выше среднего" in s for s in lines)
        lower = sum("меньше среднего" in s or "ниже среднего" in s for s in lines)

        conclusion = "\n\nВывод: "
        if higher > lower and higher >= 4:
            conclusion += (
                "по большинству метрик вы выглядите активнее и заметнее, "
                "чем средний пользователь вашего пола."
            )
        elif lower > higher and lower >= 4:
            conclusion += (
                "по ряду метрик активность и заполненность ниже средней; "
                "если цель — делать профиль более привлекательным, имеет смысл "
                "добавить информацию и проявлять больше активности."
            )
        else:
            conclusion += (
                "в целом вы близки к среднему пользователю вашего пола. "
                "Отдельные показатели чуть выше или ниже, но без сильных перекосов."
            )

        header = f"Сравнение профиля с средним профилем {sex_name}:\n"
        return header + "\n".join(lines) + conclusion