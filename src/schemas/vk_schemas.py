from typing import Generic, TypeVar
from pydantic import BaseModel


T = TypeVar('T')


class City(BaseModel):
    id: int | None = None
    title: str | None = None


class LastSeen(BaseModel):
    platform: int | None = None
    time: int | None = None


class Military(BaseModel):
    unit: str | None = None
    unit_id: int | None = None
    from_: int | None = None  # поле "from" нельзя использовать напрямую
    until: int | None = None

    model_config = {
        "populate_by_name": True,
        "fields": {"from_": "from"}
    }


class Personal(BaseModel):
    alcohol: int | None = None
    inspired_by: str | None = None
    life_main: int | None = None
    people_main: int | None = None
    smoking: int | None = None


class UserModel(BaseModel):
    id: int | None = None
    bdate: str | None = None
    city: City | None = None
    photo_200: str | None = None
    photo_max: str | None = None
    photo_id: str | None = None
    has_photo: int | None = None
    can_post: int | None = None
    can_see_all_posts: int | None = None
    interests: str | None = None
    books: str | None = None
    tv: str | None = None
    quotes: str | None = None
    about: str | None = None
    games: str | None = None
    movies: str | None = None
    activities: str | None = None
    music: str | None = None
    can_write_private_message: int | None = None
    can_send_friend_request: int | None = None
    site: str | None = None
    status: str | None = None
    last_seen: LastSeen | None = None
    followers_count: int | None = None
    blacklisted: int | None = None
    blacklisted_by_me: int | None = None
    is_hidden_from_feed: int | None = None
    career: list | None = None
    military: list[Military] | None = None
    home_town: str | None = None
    relation: int | None = None
    personal: Personal | None = None
    universities: list | None = None
    schools: list | None = None
    relatives: list | None = None
    is_no_index: bool | None = None
    sex: int | None = None
    photo_50: str | None = None
    photo_100: str | None = None
    online: int | None = None
    verified: int | None = None
    trending: int | None = None
    friend_status: int | None = None
    first_name: str | None = None
    last_name: str | None = None
    can_access_closed: bool | None = None
    is_closed: bool | None = None
    
    # отсебятина
    profile_density: float | None = None
    age: int | None = None
    friends_count: int | None = None


class VkApiSchema(BaseModel, Generic[T]):
    response: list[T]
