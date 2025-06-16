DOCKER_COMPOSE_FILE=docker-compose.yml
APP_FOLDER=app/
APP_NAME=pdf-processing-app
LOCALSTACK_NAME=localstack-main

run:
	docker compose -f $(DOCKER_COMPOSE_FILE) up --build -d

stop:
	docker-compose -f $(DOCKER_COMPOSE_FILE) stop

logs:
	docker logs -f $(APP_NAME)

lint:
	poetry run ruff format .
	poetry run ruff check . --fix

clean-db:
	docker exec -it pdf-processing-db psql -U postgres -d postgres -c "DELETE FROM knowledge_base;"

init-s3:
	docker exec -it pdf-processing-app python init-s3.py