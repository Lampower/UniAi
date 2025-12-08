import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
VK_AUTH_TOKEN = os.getenv("VK_AUTH_TOKEN")


class Config:
    vk_fields = [
        "about", "activities", "bdate", "blacklisted", "blacklisted_by_me",
        "books", "can_be_invited_group", "can_post", "can_see_all_posts",
        "can_see_audio", "can_send_friend_request", "can_write_private_message",
        "career", "city", "connections", "contacts", "counters", "country",
        "crop_photo", "education", "exports", "followers_count", "friend_status",
        "games", "has_mobile", "has_photo", "home_town", "id", "interests",
        "is_favorite", "is_friend", "is_hidden_from_feed", "is_no_index",
        "is_service", "last_seen", "lists", "military", "movies", "music",
        "occupation", "online", "personal", "photo_50", "photo_100",
        "photo_200", "photo_200_orig", "photo_400_orig", "photo_id",
        "photo_max", "photo_max_orig", "quotes", "relatives", "relation",
        "schools", "sex", "site", "status", "timezone", "trending", "tv",
        "universities", "verified", "wall_comments"
    ]
