DOCKER_COMPOSE_FILE=docker-compose.yml
APP_FOLDER=app/
APP_NAME=pdf-processing-app

run:
	docker compose -f $(DOCKER_COMPOSE_FILE) up --build -d

stop:
	docker-compose -f $(DOCKER_COMPOSE_FILE) stop

logs:
	docker logs -f $(APP_NAME)

linters:
	python -m black --line-length=79 --exclude=tests/ $(APP_FOLDER)
	python -m black --line-length=79 tests/
	python -m flake8 --exclude=tests/ $(APP_FOLDER)
	python -m bandit -r $(APP_FOLDER) --skip "B101" --recursive
	python -m mypy --ignore-missing-imports --disallow-untyped-defs $(APP_FOLDER)