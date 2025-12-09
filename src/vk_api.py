from dataclasses import dataclass

from httpx import AsyncClient

from config import VK_AUTH_TOKEN, Config
from schemas.vk_schemas import UserModel


@dataclass
class VkApi:

    async def get_user_info(self, user_id: int):
        params = {
            'user_ids': user_id,
            'fields': ','.join(Config.vk_fields)
        }
        try:
            response = await self._make_request('users.get', params)
            return UserModel.model_validate(response["response"][0])
        except Exception:
            return None

    async def _make_request(self, endpoint: str, params: dict):
        # headers = {
        #     "Authorization": VK_AUTH_TOKEN,
        # }
        params.update({
            'access_token': VK_AUTH_TOKEN,
            'v': '5.131'
        })
        response = None
        async with AsyncClient() as client:
            response = await client.get(f"https://api.vk.com/method/{endpoint}", params=params)
            response.raise_for_status()
        return response.json()

    async def get_friends_count(self, user_id: int):
        params = {
            'user_id': user_id,
        }
        try:
            response = await self._make_request('friends.get', params)
            friends = response["response"]["items"]
            return len(friends)
        except Exception:
            return 0