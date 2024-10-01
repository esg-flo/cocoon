.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available tagets:"
	@echo "  set-up	Install all python libraries and pre-commit"
	@echo "  shell		Start the pipenv shell"
	@echo "  unit-test	Run unit test"

.PHONY: set-up
set-up:
	@echo "Running 'pipenv install --dev'..."
	@pipenv install --dev
	@echo "Running 'pre-commit install'..."
	@pre-commit install

.PHONY: shell
shell:
	@echo "Running 'pipenv shell'..."
	@pipenv shell

.PHONY: unit-test
unit-test:
	@bash -c '\
	read -p "1. Verbose (y/blank [default]): " verbose; \
	if [ "$$verbose" = "y" ]; then \
		VERBOSE_FLAG="-vv"; \
	else \
		VERBOSE_FLAG=""; \
	fi; \
	read -p "2. Show output/logs (y/blank [default]): " logs; \
	if [ "$$logs" = "y" ]; then \
		LOGS_FLAG="-s"; \
	else \
		LOGS_FLAG=""; \
	fi; \
	read -p "3. Run a Specific Test (y/blank [default]): " specific_test; \
	if [ "$$specific_test" = "y" ]; then \
		read -p "- Enter the Test Name: " TEST_NAME; \
		TEST_FLAG="-k $$TEST_NAME"; \
	else \
		TEST_FLAG=""; \
	fi; \
	COMMAND="pytest"; \
	if [ -n "$$LOGS_FLAG$$VERBOSE_FLAG$$TEST_FLAG" ]; then \
		COMMAND="$$COMMAND $$LOGS_FLAG $$VERBOSE_FLAG $$TEST_FLAG"; \
	fi; \
	COMMAND=$$(echo "$$COMMAND" | xargs); \
	echo ""; \
	echo "Running command: \"$$COMMAND\""; \
	eval $$COMMAND \
	'
