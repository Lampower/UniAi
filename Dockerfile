
FROM python:3.11.9

WORKDIR /app

ENV YOUR_ENV=${YOUR_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry's configuration:
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' 

RUN pip install poetry

COPY ./pyproject.toml /app/

# RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /app

# CMD [ "poetry", "run", "main" ]
CMD [ "poetry", "run", "python", "src/bot.py" ]