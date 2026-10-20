# turbine-serve — thin wrappers over uv and scripts/. `make help` lists targets.
GATES := $(patsubst scripts/check-%.sh,%,$(wildcard scripts/check-*.sh))
.PHONY: help install check fmt sync spark-build spark-test spark-serve handoff test $(GATES)

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo "  single gates: $(GATES)   (make <gate> runs scripts/check-<gate>.sh)"

install: ## uv sync (dev group) + wire the pre-push hook
	uv sync
	git config core.hooksPath .githooks

check: ## every gate, failures only (same as pre-push and CI); logs in .logs/
	bash scripts/run-checks.sh

$(GATES):
	bash scripts/check-$@.sh
test: tests

fmt: ## ruff format + autofix imports
	uv run ruff format . && uv run ruff check --fix .

sync: ## rsync the tree to dgx-spark:~/turbine-serve
	bash scripts/sync.sh
spark-build: ## build the dev image on the Spark
	bash scripts/spark.sh --build
spark-test: sync ## run the full suite on the Spark (gpu tests included)
	bash scripts/spark.sh pytest
spark-serve: sync ## serve Qwen3-4B on the Spark (host 127.0.0.1:8000)
	bash scripts/spark.sh --serve Qwen/Qwen3-4B --kv-gb 30
handoff: ## write HANDOFF.md skeleton
	bash scripts/handoff.sh
