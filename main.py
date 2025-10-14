import pandas as pd
import bson
import os
from bson import ObjectId
import json
from datetime import datetime, timedelta
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import gc
import struct
from scipy import stats
import warnings
import traceback
import matplotlib.pyplot as plt
from dateutil import tz

warnings.filterwarnings('ignore')

class MultiBSONProcessor:
    def __init__(self, gui_output=None):
        self.file_paths = {}
        self.df_dict: dict[str, pd.DataFrame] = {}  # Словарь для хранения отдельных DataFrame
        self.df_merged = pd.DataFrame()  # Объединенный DataFrame
        self.df = pd.DataFrame()  # Основной DataFrame для обратной совместимости
        self.gui_output = gui_output
        self.is_loading = False
        self.is_analyzing = False
        self.analysis_stopped = False
        self.loaded_records = 0

    def _print(self, text):
        """Универсальный вывод - в GUI или консоль"""
        if self.gui_output:
            self.gui_output.insert(tk.END, text + "\n")
            self.gui_output.see(tk.END)
            self.gui_output.update_idletasks()
        else:
            print(text)

    def stop_analysis(self):
        """Остановка анализа"""
        self.is_analyzing = False
        self.analysis_stopped = True

    def read_bson_stream(self, file_path):
        """Потоковое чтение BSON файла"""
        try:
            with open(file_path, 'rb') as f:
                while True:
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break
                    
                    length = struct.unpack('<i', length_bytes)[0]
                    document_bytes = length_bytes + f.read(length - 4)
                    
                    if len(document_bytes) < length:
                        break
                    
                    yield bson.BSON(document_bytes).decode()
                        
        except Exception as e:
            self._print(f"Ошибка при потоковом чтении BSON файла {file_path}: {e}")
    
    def load_single_bson(self, file_path, file_name, chunk_size=1000, max_records=None):
        """Загрузка одного BSON файла"""
        self._print(f"Загрузка файла {file_name}: {file_path}")
        
        file_size = os.path.getsize(file_path)
        self._print(f"Размер файла {file_name}: {file_size / (1024**3):.2f} GB")
        
        self.is_loading = True
        self.loaded_records = 0
        all_data = []
        chunk = []
        
        try:
            for doc in self.read_bson_stream(file_path):
                if not self.is_loading:
                    self._print(f"Загрузка файла {file_name} прервана")
                    return None
                
                chunk.append(doc)
                self.loaded_records += 1
                
                if len(chunk) >= chunk_size:
                    processed_chunk = self._process_chunk(chunk)
                    all_data.extend(processed_chunk)
                    chunk = []
                    
                    # Очистка памяти
                    gc.collect()
                
                if max_records and self.loaded_records >= max_records:
                    break
            
            if chunk:
                processed_chunk = self._process_chunk(chunk)
                all_data.extend(processed_chunk)
            
            self._print(f"✓ Успешно загружено {len(all_data)} записей из {file_name}")
            return all_data
            
        except Exception as e:
            self._print(f"✗ Ошибка при загрузке {file_name}: {e}")
            return None
    
    def _process_chunk(self, chunk):
        """Обработка чанка данных с преобразованием типов"""
        processed_chunk = []
        for doc in chunk:
            processed_doc = {}
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    processed_doc[key] = str(value)
                elif isinstance(value, (dict, list)):
                    try:
                        processed_doc[key] = json.dumps(value, ensure_ascii=False, default=str)
                    except:
                        processed_doc[key] = str(value)
                elif isinstance(value, datetime):
                    processed_doc[key] = value.isoformat()
                else:
                    processed_doc[key] = value
            processed_chunk.append(processed_doc)
        return processed_chunk
    
    def stop_loading(self):
        """Остановка загрузки"""
        self.is_loading = False
    
    def load_multiple_files(self, file_paths_dict, chunk_size=1000, max_records_per_file=None):
        """Загрузка нескольких BSON файлов"""
        self.file_paths = file_paths_dict
        self.df_dict = {}
        
        for file_name, file_path in file_paths_dict.items():
            if not os.path.exists(file_path):
                self._print(f"✗ Файл не найден: {file_path}")
                continue
                
            data = self.load_single_bson(file_path, file_name, chunk_size, max_records_per_file)
            if data:
                df = self.create_dataframe_from_data(data, file_name)
                if df is not None:
                    self.df_dict[file_name] = df
                    self._print(f"✓ DataFrame создан для {file_name}: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        return len(self.df_dict) > 0
    
    def create_dataframe_from_data(self, data, file_name):
        """Создание DataFrame из данных"""
        if not data:
            return None
        
        self._print(f"Создание DataFrame для {file_name}...")
        df = pd.DataFrame(data)
        
        # Оптимизация типов данных
        df = self._optimize_memory_usage(df)
        
        return df
    
    def _optimize_memory_usage(self, df: pd.DataFrame):
        """Оптимизация использования памяти DataFrame"""
        if df is None:
            return df
        
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().any():
                col_min = df[col].min()
                col_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(df[col]):
                    if col_min >= 0:
                        if col_max < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif col_max < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif col_max < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                    else:
                        if col_min > -128 and col_max < 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min > -32768 and col_max < 32767:
                            df[col] = df[col].astype(np.int16)
        
        memory_after = df.memory_usage(deep=True).sum() / 1024**2
        self._print(f"Оптимизация памяти: {memory_before:.2f} MB -> {memory_after:.2f} MB")
        
        return df
    
    def merge_data(self):
        """Объединение данных из разных файлов по полю fake_id"""
        if not self.df_dict:
            self._print("Нет данных для объединения")
            return False
        
        if 'friends' not in self.df_dict:
            self._print("Отсутствует основной файл friends для объединения")
            return False
        
        self._print("\nНачинаем объединение данных...")
        
        try:
            # Начинаем с основного файла friends
            self.df_merged = self.df_dict['friends'].copy()
            self._print(f"Основной DataFrame (friends): {self.df_merged.shape}")
            self._print(f"Колонки в friends: {list(self.df_dict['friends'].columns)}")
            
            # ПРОВЕРКА НАЛИЧИЯ fake_id В ОСНОВНОМ ФАЙЛЕ
            if 'fake_id' not in self.df_merged.columns:
                self._print("⚠ ВАЖНО: В файле friends отсутствует колонка fake_id!")
                self._print("  Объединение может не работать корректно")
            
            # Объединяем с users по fake_id -> id
            if 'users' in self.df_dict:
                self._print("Объединение с users...")
                users_df = self.df_dict['users'].copy()
                # Переименуем id в users_id для ясности
                if 'id' in users_df.columns:
                    users_df = users_df.rename(columns={'id': 'users_id'})
                    self.df_merged = pd.merge(
                        self.df_merged, 
                        users_df, 
                        left_on='fake_id', 
                        right_on='users_id', 
                        how='left',
                        suffixes=('', '_users')
                    )
                    self._print(f"После объединения с users: {self.df_merged.shape}")
                else:
                    self._print("⚠ В файле users отсутствует колонка 'id'")
            
            # Объединяем с walls по fake_id -> user_id
            if 'walls' in self.df_dict:
                self._print("Объединение с walls...")
                walls_df = self.df_dict['walls'].copy()
                if 'user_id' in walls_df.columns:
                    self.df_merged = pd.merge(
                        self.df_merged, 
                        walls_df, 
                        left_on='fake_id', 
                        right_on='user_id', 
                        how='left',
                        suffixes=('', '_walls')
                    )
                    self._print(f"После объединения с walls: {self.df_merged.shape}")
                else:
                    self._print("⚠ В файле walls отсутствует колонка 'user_id'")
            
            # Объединяем с username по fake_id -> id
            if 'username' in self.df_dict:
                self._print("Объединение с username...")
                username_df = self.df_dict['username'].copy()
                # Переименуем id в username_id для ясности
                if 'id' in username_df.columns:
                    username_df = username_df.rename(columns={'id': 'username_id'})
                    self.df_merged = pd.merge(
                        self.df_merged, 
                        username_df, 
                        left_on='fake_id', 
                        right_on='username_id', 
                        how='left',
                        suffixes=('', '_username')
                    )
                    self._print(f"После объединения с username: {self.df_merged.shape}")
                else:
                    self._print("⚠ В файле username отсутствует колонка 'id'")
            
            # Устанавливаем объединенный DataFrame как основной для анализа
            self.df = self.df_merged
            self._print(f"✓ Объединение завершено. Итоговый размер: {self.df.shape}")
            self._print(f"Колонки в объединенном DataFrame: {len(self.df_merged.columns)}")
            
            return True
            
        except Exception as e:
            self._print(f"✗ Ошибка при объединении данных: {e}")
            self._print(f"Трассировка: {traceback.format_exc()}")
            return False
    
    def analyze_merged_data(self):
        """Анализ объединенных данных"""
        if self.df_merged is None:
            self._print("Сначала выполните объединение данных")
            return
        
        self._print("\n" + "="*80)
        self._print("АНАЛИЗ ОБЪЕДИНЕННЫХ ДАННЫХ")
        self._print("="*80)
        
        self._print(f"Размер объединенных данных: {self.df_merged.shape[0]} строк, {self.df_merged.shape[1]} столбцов")
        
        self._print("\nИСТОЧНИКИ ДАННЫХ:")
        for file_name, df in self.df_dict.items():
            self._print(f"  {file_name}: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        self._print("\nСТОЛБЦЫ ОБЪЕДИНЕННОЙ ТАБЛИЦЫ:")
        for i, col in enumerate(self.df_merged.columns, 1):
            dtype = self.df_merged[col].dtype
            self._print(f"  {i:2d}. {col}: {dtype.name}")
        
        # Показываем первые строки объединенной таблицы
        self._print("\nПЕРВЫЕ 5 СТРОК ОБЪЕДИНЕННОЙ ТАБЛИЦЫ:")
        preview_columns = self.df_merged.columns[:8] if len(self.df_merged.columns) > 8 else self.df_merged.columns
        self._print(self.df_merged[preview_columns].head().to_string())
    
    def detect_bots_and_empty_profiles_advanced(self):
        """Обнаружение ботов, расчёт bot_score/is_bot и краткая сводка"""
        try:
            if self.df_merged is None or self.df_merged.empty:
                self._print("Сначала выполните объединение данных")
                return

            df = self.df_merged.copy()
            total = len(df)

            self._print("\n" + "="*80)
            self._print("ОБНАРУЖЕНИЕ БОТОВ И ПУСТЫХ ПРОФИЛЕЙ")
            self._print("="*80)
            self._print(f"Размер объединенных данных: {df.shape}")

            # ---------- 1) ДРУЗЬЯ ----------
            friends_col = None
            for c in ('friends_count', 'friends.total', 'counters.friends', 'count'):
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    friends_col = c
                    break
            if friends_col is None:
                df['friends_count_norm'] = np.nan
            else:
                df['friends_count_norm'] = pd.to_numeric(df[friends_col], errors='coerce')

            crit_few_friends = df['friends_count_norm'] < 3

            # ---------- 2) ПОСТЫ ----------
            # пробуем взять явную числовую колонку
            posts_col = None
            for c in ('posts_count', 'wall_posts', 'posts', 'counters.posts'):
                if c in df.columns:
                    if pd.api.types.is_array_like(df[c]):
                        posts_col = c
                        break
            # если не нашли числовую — посчитаем длину "списка" в возможных колонках
       
            if posts_col is not None:
                if pd.api.types.is_numeric_dtype(df[posts_col]):
                    # если сразу число
                    df['posts_count_norm'] = pd.to_numeric(df[posts_col], errors='coerce')
                else:
                    # wall_posts хранится как список или JSON — считаем длину
                    df['posts_count_norm'] = df[posts_col].apply(self._safe_len_listlike)
            else:
                df['posts_count_norm'] = np.nan

            crit_few_posts = df['posts_count_norm'] < 2

            # ---------- 3) ЗАПОЛНЕННОСТЬ ПРОФИЛЯ ----------
            # Берём только содержательные поля
            content_cols = [c for c in [
                'about', 'status', 'site', 'interests', 'books', 'music', 'movies', 'tv',
                'quotes', 'bdate', 'city', 'home_town', 'occupation', 'education_status',
                'university', 'university_name', 'faculty', 'faculty_name', 'relation'
            ] if c in df.columns]

            def _is_filled(x):
                if pd.isna(x): return 0
                if isinstance(x, str): return int(len(x.strip()) > 0 and x.strip().lower() not in ('none','null','nan'))
                if isinstance(x, (list, dict)): return int(len(x) > 0)
                return 1

            if content_cols:
                filled_counts = df[content_cols].applymap(_is_filled).sum(axis=1)
                filled_ratio = filled_counts / len(content_cols)
            else:
                filled_ratio = pd.Series(np.nan, index=df.index)

            crit_empty_profile = filled_ratio < 0.30

            # ---------- 4) АКТИВНОСТЬ ПО ДАТАМ ----------
            # Ищем первую колонку, похожую на дату/last_seen
            date_columns = [c for c in df.columns if any(k in c.lower() for k in ('last_seen','date','time','created'))]
            inactive_mask = pd.Series(False, index=df.index)
            if date_columns:
                # берём первую подходящую
                dcol = date_columns[0]
                dates = df[dcol].apply(self._to_datetime)
                six_months_ago = pd.Timestamp.utcnow() - pd.Timedelta(days=180)
                # даты могут быть tz-aware — приводим к naive UTC
                dates = pd.to_datetime(dates, utc=True, errors='coerce').dt.tz_convert(None)
                six_months_ago = six_months_ago.tz_localize(None)
                dates = pd.to_datetime(dates, errors='coerce')
                if hasattr(dates.dt, "tz_localize"):
                    dates = dates.dt.tz_localize(None)

                inactive_mask = dates.notna() & (dates < six_months_ago)
            crit_inactive = inactive_mask

            # ---------- 5) НЕТ ФОТО (опционально) ----------
            has_photo_col = 'has_photo' if 'has_photo' in df.columns else None
            if has_photo_col:
                crit_no_photo = (pd.to_numeric(df[has_photo_col], errors='coerce') == 0)
            else:
                crit_no_photo = pd.Series(False, index=df.index)

            # ---------- СВОДИМ В bot_score ----------
            criteria = {
                'мало друзей (<3)': crit_few_friends.fillna(False),
                'мало постов (<2)': crit_few_posts.fillna(False),
                'пустой профиль (<30%)': crit_empty_profile.fillna(False),
                'неактивен (>6 мес)': crit_inactive.fillna(False),
                'нет фото': crit_no_photo.fillna(False)
            }

            crit_df = pd.DataFrame(criteria)
            df['bot_score'] = crit_df.sum(axis=1)
            df['is_bot'] = df['bot_score'] >= 2

            # сохраним диагностические колонки (полезно в отчёте)
            df['filled_ratio'] = filled_ratio
            self.df_merged = df
            self.df = df
            self.bot_mask = df['is_bot']
            self.duplicate_mask = df.duplicated(subset=['fake_id'], keep=False) if 'fake_id' in df.columns else pd.Series(False, index=df.index)

            # ---------- ВЫВОД СТАТИСТИКИ ----------
            bots_cnt = int(df['is_bot'].sum())
            self._print(f"Потенциальных ботов (bot_score ≥ 2): {bots_cnt} из {total} ({bots_cnt/total*100:.1f}%)")
            self._print("\nРАСПРЕДЕЛЕНИЕ ПО КРИТЕРИЯМ:")
            for i, (name, mask) in enumerate(criteria.items(), 1):
                self._print(f"  {i}. {name}: {int(mask.sum())}")

            # Покажем топ-3 «ботов» с причинами
            if bots_cnt > 0:
                self._print("\nПРИМЕРЫ БОТОВ:")
                for idx, row in df[df['is_bot']].nlargest(3, 'bot_score').iterrows():
                    reasons = [name for name, mask in criteria.items() if bool(mask.loc[idx])]
                    self._print(f"  idx={idx}, bot_score={int(row['bot_score'])}: " + ", ".join(reasons))

            self._print("\n✓ Анализ завершён. Доступны колонки: bot_score, is_bot, filled_ratio, friends_count_norm, posts_count_norm")

        except Exception as e:
            self._print(f"✗ КРИТИЧЕСКАЯ ОШИБКА при анализе ботов: {e}")
            self._print(f"Трассировка: {traceback.format_exc()}")
        finally:
            self.is_analyzing = False
        
        self.save_bot_report()
    
    def save_bot_report(self, out_dir="bot_report"):
        """Сохранить PNG-графики по ботам и активности"""
        if self.df_merged is None or self.df_merged.empty:
            self._print("Нет данных для отчёта — сначала запусти анализ")
            return False

        os.makedirs(out_dir, exist_ok=True)
        df = self.df_merged

        # 1) Гистограмма друзей
        plt.figure()
        df['friends_count_norm'].dropna().clip(upper=300).plot(kind='hist', bins=30)
        plt.title("Распределение количества друзей (обрезка 300+)")
        plt.xlabel("Друзья")
        plt.ylabel("Число профилей")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "01_friends_hist.png"), dpi=160)
        plt.close()

        # 2) Гистограмма постов
        plt.figure()
        df['posts_count_norm'].dropna().clip(upper=200).plot(kind='hist', bins=30)
        plt.title("Распределение количества постов (обрезка 200+)")
        plt.xlabel("Посты")
        plt.ylabel("Число профилей")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "02_posts_hist.png"), dpi=160)
        plt.close()

        # 3) Доля ботов
        plt.figure()
        df['is_bot'].value_counts().rename(index={True:'Боты', False:'Живые'}).plot(kind='bar')
        plt.title("Доля ботов и живых профилей")
        plt.ylabel("Число профилей")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "03_bots_share.png"), dpi=160)
        plt.close()

        # 4) Boxplot друзей/постов по ботам
        plt.figure()
        df[['friends_count_norm','is_bot']].dropna().boxplot(by='is_bot')
        plt.suptitle("")
        plt.title("Друзья по классам (бот/не бот)")
        plt.xlabel("is_bot")
        plt.ylabel("Друзья")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "04_box_friends_by_bot.png"), dpi=160)
        plt.close()

        plt.figure()
        df[['posts_count_norm','is_bot']].dropna().boxplot(by='is_bot')
        plt.suptitle("")
        plt.title("Посты по классам (бот/не бот)")
        plt.xlabel("is_bot")
        plt.ylabel("Посты")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "05_box_posts_by_bot.png"), dpi=160)
        plt.close()

        # 5) Heatmap быстрых корреляций (если есть что коррелировать)
        num_cols = [c for c in ['friends_count_norm','posts_count_norm','followers_count','has_photo','bot_score','filled_ratio'] if c in df.columns]
        if num_cols:
            corr = pd.to_numeric(df[num_cols].stack(), errors='coerce').unstack().corr()
            plt.figure()
            im = plt.imshow(corr, interpolation='nearest')
            plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right')
            plt.yticks(range(len(num_cols)), num_cols)
            plt.title("Корреляции (быстрая матрица)")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "06_corr_heatmap.png"), dpi=160)
            plt.close()

        self._print(f"✓ PNG-отчёт сохранён в папку: {os.path.abspath(out_dir)}")
        return True

    def get_data_structure(self):
        """Получить структуру данных для отображения"""
        if self.df_merged is None:
            return "Данные не объединены"
        
        result = []
        result.append("СТРУКТУРА ОБЪЕДИНЕННЫХ ДАННЫХ:")
        result.append("="*60)
        result.append(f"Размер: {self.df_merged.shape[0]} строк, {self.df_merged.shape[1]} столбцов")
        result.append("\nКОЛОНКИ И ТИПЫ ДАННЫХ:")
        
        for i, col in enumerate(self.df_merged.columns, 1):
            dtype = self.df_merged[col].dtype
            non_null = self.df_merged[col].notna().sum()
            result.append(f"{i:3d}. {col} ({dtype.type}) - заполнено: {non_null}")
        
        return "\n".join(result)
    
    def remove_detected_issues(self, remove_bots=True, remove_empty=True, remove_duplicates=True):
        """Удаление обнаруженных проблем: ботов, пустых аккаунтов и дубликатов"""
        if self.df_merged is None:
            self._print("Сначала выполните объединение и анализ данных")
            return False
        
        original_count = len(self.df_merged)
        df_clean = self.df_merged.copy()
        
        self._print("\n" + "="*80)
        self._print("УДАЛЕНИЕ ПРОБЛЕМНЫХ ЗАПИСЕЙ")
        self._print("="*80)
        
        removed_counts = {}
        
        # Удаление ботов
        if remove_bots and hasattr(self, 'bot_mask'):
            bots_count = self.bot_mask.sum()
            df_clean = df_clean[~self.bot_mask]
            removed_counts['ботов'] = bots_count
            self._print(f"Удалено ботов: {bots_count}")
        
        # Удаление дубликатов (оставляем первую запись для каждого fake_id)
        if remove_duplicates and hasattr(self, 'duplicate_mask'):
            duplicates_count = self.duplicate_mask.sum()
            df_clean = df_clean.drop_duplicates(subset=['fake_id'], keep='first')
            removed_counts['дубликатов'] = duplicates_count - (len(self.df_merged) - len(df_clean))
            self._print(f"Удалено дубликатов: {removed_counts['дубликатов']}")
        
        # Удаление пустых профилей (на основе критерия заполненности)
        if remove_empty:
            filled_ratio = df_clean.notna().sum(axis=1) / len(df_clean.columns)
            empty_profiles = filled_ratio < 0.2  # Заполнено менее 20% полей
            empty_count = empty_profiles.sum()
            df_clean = df_clean[~empty_profiles]
            removed_counts['пустых профилей'] = empty_count
            self._print(f"Удалено пустых профилей: {empty_count}")
        
        # Обновляем основной DataFrame
        self.df_merged = df_clean
        self.df = df_clean
        
        final_count = len(df_clean)
        total_removed = original_count - final_count
        
        self._print(f"\nИТОГИ ОЧИСТКИ:")
        self._print(f"Было записей: {original_count}")
        self._print(f"Стало записей: {final_count}")
        self._print(f"Удалено всего: {total_removed}")
        self._print(f"Сокращение: {total_removed/original_count*100:.1f}%")
        
        return True
    
    def _safe_len_listlike(self, val):
        """Вернуть длину списка/массива/JSON-строки, иначе NaN"""
        import math
        if val is None:
            return np.nan
        # если это уже список/кортеж/множество
        if isinstance(val, (list, tuple, set)):
            return len(val)
        # если строка с JSON/CSV
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return np.nan
            # JSON-массив
            if s.startswith('[') and s.endswith(']'):
                try:
                    arr = json.loads(s)
                    return len(arr) if isinstance(arr, list) else np.nan
                except:
                    return np.nan
            # иногда там строка с запятыми
            if ',' in s:
                return len([x for x in s.split(',') if x.strip()])
        return np.nan

    def _to_datetime(self, val):
        """Преобразовать last_seen/*date*/time в pd.Timestamp (UTC-naive) или NaT"""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return pd.NaT
        # уже дата
        if isinstance(val, (pd.Timestamp, datetime)):
            return pd.to_datetime(val, errors='coerce')
        # UNIX int/float
        if isinstance(val, (int, np.integer, float)) and not np.isnan(val):
            try:
                return pd.to_datetime(int(val), unit='s', errors='coerce')
            except:
                pass
        # словарь/JSON со структурой VK last_seen: {"time": 1700000000, ...}
        if isinstance(val, dict):
            for k in ('time', 'date', 'ts'):
                if k in val:
                    return self._to_datetime(val[k])
            return pd.NaT
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return pd.NaT
            # JSON
            if s.startswith('{') and s.endswith('}'):
                try:
                    d = json.loads(s)
                    return self._to_datetime(d)
                except:
                    pass
            # ISO/обычная дата
            return pd.to_datetime(s, errors='coerce')
        return pd.NaT

    def get_merged_column_names(self):
        """Получить список всех столбцов объединенной таблицы"""
        if self.df_merged is None:
            return []
        return list(self.df_merged.columns)
    
    def save_merged_to_csv(self, output_path, sample_size=None):
        """Сохранение объединенных данных в CSV"""
        if self.df_merged is None:
            self._print("Нет объединенных данных для сохранения")
            return False
        
        if sample_size and sample_size < len(self.df_merged):
            data_to_save = self.df_merged.head(sample_size)
        else:
            data_to_save = self.df_merged
        
        try:
            data_to_save.to_csv(output_path, index=False, encoding='utf-8')
            self._print(f"✓ Объединенные данные сохранены в {output_path}")
            self._print(f"Сохранено записей: {len(data_to_save)}")
            self._print(f"Сохранено столбцов: {len(data_to_save.columns)}")
            return True
        except Exception as e:
            self._print(f"✗ Ошибка при сохранении: {e}")
            return False

    # Методы для обратной совместимости с оригинальным интерфейсом
    def analyze_data(self):
        """Анализ данных (для обратной совместимости)"""
        if self.df is None:
            self._print("Сначала загрузите данные")
            return
        self.analyze_merged_data()

    def get_column_names(self):
        """Получить список столбцов (для обратной совместимости)"""
        return self.get_merged_column_names()

class MultiBSONViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi BSON Data Viewer - Расширенный анализ")
        self.root.geometry("1400x900")
        
        self.processor: MultiBSONProcessor = None
        self.load_thread = None
        self.setup_ui()
    
    def setup_ui(self):
        """Создание пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка весов для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Заголовок
        title_label = ttk.Label(main_frame, 
                               text="ПРОСМОТРЩИК МНОГИХ BSON ФАЙЛОВ - РАСШИРЕННЫЙ АНАЛИЗ", 
                               font=('Arial', 12, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # Панель управления загрузкой нескольких файлов
        multi_load_frame = ttk.LabelFrame(main_frame, text="Загрузка нескольких BSON файлов", padding="10")
        multi_load_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        multi_load_frame.columnconfigure(1, weight=1)
        
        # Выбор папки с файлами
        ttk.Label(multi_load_frame, text="Папка с файлами:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.folder_path_var = tk.StringVar()
        folder_entry = ttk.Entry(multi_load_frame, textvariable=self.folder_path_var, width=60)
        folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(multi_load_frame, text="Обзор...", command=self.browse_folder).grid(row=0, column=2)
        
        # Информация о требуемых файлах
        files_info = ttk.Label(multi_load_frame, 
                              text="Требуемые файлы: friends.bson, users.bson, walls.bson, username.bson", 
                              font=('Arial', 8))
        files_info.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky=tk.W)
        
        # Параметры загрузки
        params_frame = ttk.Frame(multi_load_frame)
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(params_frame, text="Размер чанка:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.chunk_size_var = tk.StringVar(value="1000")
        ttk.Entry(params_frame, textvariable=self.chunk_size_var, width=10).grid(row=0, column=1, padx=(0, 15))
        
        # ttk.Label(params_frame, text="Макс. записей на файл:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.max_records_var = tk.StringVar(value=None)
        # ttk.Entry(params_frame, textvariable=self.max_records_var, width=10).grid(row=0, column=3)
        
        # Кнопки управления
        button_frame = ttk.Frame(multi_load_frame)
        button_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame, text="Загрузить все файлы", 
                  command=self.start_multi_loading).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Объединить данные", 
                  command=self.merge_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Остановить загрузку", 
                  command=self.stop_loading).pack(side=tk.LEFT, padx=(0, 10))
        
        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(multi_load_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.progress_label = ttk.Label(multi_load_frame, text="Готов к работе")
        self.progress_label.grid(row=5, column=0, columnspan=3, pady=(5, 0))
        
        # Область вывода
        output_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        output_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Текстовое поле с прокруткой
        self.output_text = scrolledtext.ScrolledText(output_frame, width=120, height=25, wrap=tk.WORD)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Панель анализа объединенных данных
        analysis_frame = ttk.LabelFrame(main_frame, text="Анализ объединенных данных", padding="10")
        analysis_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Кнопки анализа
        ttk.Button(analysis_frame, text="Анализ объединенных данных", 
                  command=self.analyze_merged_data).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(analysis_frame, text="Расширенный анализ ботов", 
                  command=self.detect_bots_advanced).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(analysis_frame, text="Структура данных", 
                  command=self.show_data_structure).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(analysis_frame, text="Показать все столбцы", 
                  command=self.show_all_columns).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(analysis_frame, text="Остановить анализ", 
                  command=self.stop_analysis).pack(side=tk.LEFT, padx=(0, 10))
        
        # Панель очистки данных
        cleanup_frame = ttk.LabelFrame(main_frame, text="Очистка данных", padding="10")
        cleanup_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(cleanup_frame, text="Удалить ботов", 
                  command=self.remove_bots).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(cleanup_frame, text="Удалить дубликаты", 
                  command=self.remove_duplicates).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(cleanup_frame, text="Удалить пустые профили", 
                  command=self.remove_empty_profiles).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(cleanup_frame, text="Полная очистка", 
                  command=self.full_cleanup).pack(side=tk.LEFT, padx=(0, 10))
        
        # Панель экспорта
        export_frame = ttk.LabelFrame(main_frame, text="Экспорт данных", padding="10")
        export_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(export_frame, text="Сохранить объединенные данные", 
                  command=self.save_merged_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="Очистить вывод", 
                  command=self.clear_output).pack(side=tk.LEFT, padx=(0, 10))
    
    def _print(self, text):
        """Вывод текста в текстовое поле"""
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.output_text.update_idletasks()
    
    def clear_output(self):
        """Очистка текстового поля"""
        self.output_text.delete(1.0, tk.END)
    
    def browse_folder(self):
        """Выбор папки с BSON файлами"""
        folder_path = filedialog.askdirectory(title="Выберите папку с BSON файлами")
        if folder_path:
            self.folder_path_var.set(folder_path)
            self._print(f"Выбрана папка: {folder_path}")
            
            # Проверяем наличие файлов
            required_files = ['friends.bson', 'users.bson', 'walls.bson', 'username.bson']
            found_files = []
            missing_files = []
            
            for file_name in required_files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    found_files.append(file_name)
                else:
                    missing_files.append(file_name)
            
            self._print(f"Найдены файлы: {', '.join(found_files)}")
            if missing_files:
                self._print(f"Отсутствуют файлы: {', '.join(missing_files)}")
    
    def get_file_paths(self):
        """Получение путей к файлам из выбранной папки"""
        folder_path = self.folder_path_var.get()
        if not folder_path or not os.path.exists(folder_path):
            return None
        
        file_paths = {}
        required_files = {
            'friends': 'friends.bson',
            'users': 'users.bson', 
            'walls': 'walls.bson',
            'username': 'username.bson'
        }
        
        for file_key, file_name in required_files.items():
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                file_paths[file_key] = file_path
            else:
                self._print(f"⚠ Внимание: файл {file_name} не найден")
        
        return file_paths
    
    def start_multi_loading(self):
        """Запуск загрузки нескольких файлов"""
        file_paths = self.get_file_paths()
        if not file_paths:
            messagebox.showwarning("Предупреждение", "Сначала выберите папку с BSON файлами")
            return
        
        try:
            chunk_size = int(self.chunk_size_var.get())
            max_records = int(self.max_records_var.get()) if self.max_records_var.get() else None
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные значения параметров")
            return
        
        self.clear_output()
        self._print("Запуск загрузки нескольких BSON файлов...")
        self.progress_var.set(0)
        
        def load_thread():
            self.processor = MultiBSONProcessor(self.output_text)
            success = self.processor.load_multiple_files(
                file_paths, 
                chunk_size=chunk_size, 
                max_records_per_file=max_records
            )
            
            if success:
                self.root.after(0, lambda: self._print("✓ Все файлы успешно загружены!"))
                # Показываем статистику по загруженным файлам
                for file_name, df in self.processor.df_dict.items():
                    self.root.after(0, lambda n=file_name, d=df: 
                                  self._print(f"  {n}: {d.shape[0]} строк, {d.shape[1]} столбцов"))
            else:
                self.root.after(0, lambda: self._print("✗ Ошибка при загрузке файлов"))
        
        self.load_thread = threading.Thread(target=load_thread, daemon=True)
        self.load_thread.start()
    
    def stop_loading(self):
        """Остановка загрузки"""
        if self.processor:
            self.processor.stop_loading()
            self._print("Загрузка остановлена пользователем")
            self.progress_label.config(text="Загрузка остановлена")
    
    def stop_analysis(self):
        """Остановка анализа"""
        if self.processor:
            self.processor.stop_analysis()
            self._print("Анализ остановлен пользователем")
    
    def merge_data(self):
        """Объединение загруженных данных"""
        if not self.processor or not self.processor.df_dict:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файлы")
            return
        
        self.clear_output()
        self._print("Выполняется объединение данных...")
        
        def merge_thread():
            success = self.processor.merge_data()
            if success:
                self.root.after(0, lambda: self._print("✓ Данные успешно объединены!"))
            else:
                self.root.after(0, lambda: self._print("✗ Ошибка при объединении данных"))
        
        threading.Thread(target=merge_thread, daemon=True).start()
    
    def analyze_merged_data(self):
        """Анализ объединенных данных"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        self.clear_output()
        self._print("Выполняется анализ объединенных данных...")
        
        threading.Thread(target=self.processor.analyze_merged_data, daemon=True).start()
    
    def detect_bots_advanced(self):
        """Расширенное обнаружение ботов"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        # Проверяем, не выполняется ли уже анализ
        if hasattr(self.processor, 'is_analyzing') and self.processor.is_analyzing:
            messagebox.showwarning("Предупреждение", "Анализ уже выполняется")
            return
        
        self.clear_output()
        self._print("Выполняется расширенное обнаружение ботов...")
        
        def analysis_thread():
            try:
                self.processor.detect_bots_and_empty_profiles_advanced()
            except Exception as e:
                self.root.after(0, lambda: self._print(f"Ошибка в потоке анализа: {e}"))
        
        threading.Thread(target=analysis_thread, daemon=True).start()
    
    def show_data_structure(self):
        """Показать структуру объединенных данных"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        self.clear_output()
        structure_info = self.processor.get_data_structure()
        self._print(structure_info)
        
        # Показываем примеры данных для первых 5 колонок
        self._print("\nПРИМЕР ДАННЫХ (первые 3 строки, первые 5 колонок):")
        preview_cols = self.processor.df_merged.columns[:5]
        if len(preview_cols) > 0:
            self._print(self.processor.df_merged[preview_cols].head(3).to_string())
    
    def show_all_columns(self):
        """Показать все столбцы объединенной таблицы"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        self.clear_output()
        columns = self.processor.get_merged_column_names()
        self._print("ВСЕ СТОЛБЦЫ ОБЪЕДИНЕННОЙ ТАБЛИЦЫ:")
        self._print("="*60)
        for i, col in enumerate(columns, 1):
            dtype = self.processor.df_merged[col].dtype
            self._print(f"{i:3d}. {col} ({dtype})")
    
    def remove_bots(self):
        """Удаление ботов"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        if not hasattr(self.processor, 'bot_mask'):
            messagebox.showwarning("Предупреждение", "Сначала выполните обнаружение ботов")
            return
        
        result = messagebox.askyesno("Подтверждение", 
                                   "Удалить обнаруженных ботов? Это действие нельзя отменить.")
        if result:
            self.clear_output()
            self._print("Удаление ботов...")
            
            def remove_thread():
                success = self.processor.remove_detected_issues(
                    remove_bots=True, 
                    remove_empty=False, 
                    remove_duplicates=False
                )
                if success:
                    self.root.after(0, lambda: self._print("✓ Боты успешно удалены!"))
            
            threading.Thread(target=remove_thread, daemon=True).start()
    
    def remove_duplicates(self):
        """Удаление дубликатов"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        result = messagebox.askyesno("Подтверждение", 
                                   "Удалить дубликаты аккаунтов? Это действие нельзя отменить.")
        if result:
            self.clear_output()
            self._print("Удаление дубликатов...")
            
            def remove_thread():
                success = self.processor.remove_detected_issues(
                    remove_bots=False, 
                    remove_empty=False, 
                    remove_duplicates=True
                )
                if success:
                    self.root.after(0, lambda: self._print("✓ Дубликаты успешно удалены!"))
            
            threading.Thread(target=remove_thread, daemon=True).start()
    
    def remove_empty_profiles(self):
        """Удаление пустых профилей"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        result = messagebox.askyesno("Подтверждение", 
                                   "Удалить пустые профили? Это действие нельзя отменить.")
        if result:
            self.clear_output()
            self._print("Удаление пустых профилей...")
            
            def remove_thread():
                success = self.processor.remove_detected_issues(
                    remove_bots=False, 
                    remove_empty=True, 
                    remove_duplicates=False
                )
                if success:
                    self.root.after(0, lambda: self._print("✓ Пустые профили успешно удалены!"))
            
            threading.Thread(target=remove_thread, daemon=True).start()
    
    def full_cleanup(self):
        """Полная очистка данных"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        result = messagebox.askyesno("Подтверждение", 
                                   "Выполнить полную очистку (боты, дубликаты, пустые профили)? Это действие нельзя отменить.")
        if result:
            self.clear_output()
            self._print("Выполняется полная очистка данных...")
            
            def cleanup_thread():
                success = self.processor.remove_detected_issues(
                    remove_bots=True, 
                    remove_empty=True, 
                    remove_duplicates=True
                )
                if success:
                    self.root.after(0, lambda: self._print("✓ Полная очистка завершена!"))
            
            threading.Thread(target=cleanup_thread, daemon=True).start()
    
    def save_merged_data(self):
        """Сохранение объединенных данных"""
        if not self.processor or self.processor.df_merged is None:
            messagebox.showwarning("Предупреждение", "Сначала объедините данные")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить объединенные данные как CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self._print(f"Сохранение объединенных данных в: {file_path}")
            
            def save_thread():
                try:
                    self.processor.save_merged_to_csv(file_path)
                except Exception as e:
                    self.root.after(0, lambda: self._print(f"✗ Ошибка сохранения: {e}"))
            
            threading.Thread(target=save_thread, daemon=True).start()

def main():
    root = tk.Tk()
    app = MultiBSONViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()