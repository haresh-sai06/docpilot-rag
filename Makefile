PY ?= python
export PYTHONPATH := src

.PHONY: corpus testset eval-retrieval eval-generation serve ui test lint docker

corpus:           ## download FastAPI docs (pinned 0.115.0) and chunk them
	$(PY) -m docpilot.corpus
testset:          ## synthetic 100-question set (needs an LLM)
	$(PY) eval/build_testset.py --n 100
eval-retrieval:   ## BM25 / dense / hybrid x rerank ablation on all 100 questions
	$(PY) eval/run_retrieval_eval.py
eval-generation:  ## faithfulness / relevance / latency / tokens, rerank on vs off, cache cold vs warm
	PY=$(PY) bash scripts/run_generation_eval.sh
serve:
	uvicorn docpilot.api:app --port 8000
ui:
	cd frontend && npm install && npm run dev
test:
	$(PY) -m pytest -q
lint:
	ruff check src tests eval scripts
docker:
	docker build -t docpilot .
