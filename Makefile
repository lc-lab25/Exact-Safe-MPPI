PY      ?= python
WORKERS ?= 1
export PYTHONPATH := src:experiments:$(PYTHONPATH)

.PHONY: install test quick reproduce figures videos gifs clean

install:          ## create the environment from pinned requirements
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:             ## run the unit tests
	$(PY) -m pytest tests -q

quick:            ## every experiment, 3 seeds and a reduced grid (~10 min)
	$(PY) experiments/closed_form_checks.py --quick
	$(PY) experiments/lcp_vs_scalar.py      --quick
	$(PY) experiments/gap_threshold.py      --quick --workers $(WORKERS)
	$(PY) experiments/fig2_median_runs.py   --quick
	$(PY) experiments/nine_constraint.py    --quick --workers $(WORKERS)
	$(PY) experiments/kmax_ablation.py      --quick --workers $(WORKERS)
	$(PY) scripts/verify_results.py --quick

reproduce:        ## full paper scale, then verification
	WORKERS=$(WORKERS) bash scripts/reproduce_all.sh

figures:          ## regenerate figures from the CSVs already in results/; reruns nothing
	$(PY) experiments/gap_threshold.py    --plot-only
	$(PY) experiments/fig2_median_runs.py --plot-only

videos: gifs      ## render every simulation video, then rebuild the inline GIF previews

gifs:             ## build media/gifs/*.gif from media/videos/*.mp4 (needs ffmpeg)
	$(PY) scripts/render_video.py --all
	bash scripts/make_gifs.sh

clean:
	rm -rf results/gap_threshold results/nine_constraint results/kmax_ablation \
	       results/fig2_median_runs results/closed_form_checks results/lcp_vs_scalar figures
