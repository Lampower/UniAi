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
