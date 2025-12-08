from datetime import datetime


class Utils:
    @classmethod
    def parse_bdate(cls, bdate: str):
        if not bdate:
            return None
        try:
            return datetime.strptime(bdate, "%d.%m.%Y")
        except ValueError:
            try:
                return datetime.strptime(bdate, "%d.%m")  # будет 1900 год
            except ValueError:
                return None

    @classmethod
    def get_sex(cls, sex: int):
        if sex == 0:
            return "Неопределен"
        elif sex == 1:
            return "Женщина"
        elif sex == 2:
            return "Мужчина"
        else:
            return "Не опознан"
