# CineMuse — AI Film Concierge

CineMuse is a Streamlit movie recommendation agent. It helps a viewer describe a mood or a set of requirements in natural language, searches a curated movie corpus, checks hard constraints in Python, and asks an LLM to rerank the eligible candidates and explain its choices. The application is designed as a course prototype, not as a general-purpose movie database or a guarantee of content suitability.

## Run locally (Windows PowerShell)

Open PowerShell in the extracted project folder. Confirm that `app.py`, `agent.py`, and `movies.csv` are visible:

```powershell
Set-Location "C:\path\to\movie_app_cinemuse_llm"
dir
```

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file and add your API key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set `OPENAI_API_KEY` to an API key that has access to the configured model. Keep `OPENAI_MODEL=gpt-5.4-mini`. Save the file, then run:

```powershell
python -m streamlit run app.py
```

The app opens at a local address printed in the terminal. Stop it with `Ctrl+C`. Do not put a real key in source code, screenshots, or a GitHub repository.

## Publish the code and deploy the website

GitHub stores the project files and their history; it does not run this Streamlit application as an interactive website. For a hosted demo, push this project to GitHub, then connect that repository to Streamlit Community Cloud.

1. Create a GitHub repository and push the contents of this project folder. Before publishing, check `git status` and confirm that `.env` and `.streamlit/secrets.toml` are not listed.
2. In Streamlit Community Cloud, choose **Create app**, select the GitHub repository and branch, and set the app file to `app.py`.
3. In the deployment's **Advanced settings / Secrets** field, add the following TOML values with the group's API key:

   ```toml
   OPENAI_API_KEY = "your-real-key"
   OPENAI_MODEL = "gpt-5.4-mini"
   ```

   Do not commit a real `secrets.toml` file. The deployed code reads Streamlit-managed secrets; local development reads `.env`.
4. Choose whether the deployed app should be public or private. Public apps can be opened by anyone with the link. A private app requires you to invite viewers. Deploy the app and check its build logs if startup fails. API use consumes the key owner's API quota, so agree on visibility and usage limits with the key owner before sharing.

## Files

- `app.py` — English CineMuse chat interface, theme prompts, preference sidebar, and movie cards.
- `agent.py` — LangGraph workflow, LLM tool call, result validation, and bounded correction attempts.
- `retrieval.py` — constraint validation, CSV loading, filtering, and lexical ranking.
- `llm_client.py` — OpenAI Responses API transport and local/deployed secret lookup.
- `movies.csv` — the 584-record dataset used by the app. Keep it beside the Python files.
- `evaluate.py`, `test_*.py` — evaluation runner and automated regression tests.
- `TEST_REPORT.md`, `live_results.json`, `LIVE_STATUS.json` — supplied test report and recorded smoke-test results.
- `.env.example` — blank local configuration template.
- `.streamlit/secrets.toml.example` — deployment secrets format example; it contains no working key.

## Dataset and provenance

The provided CSV contains 584 movie records: 496 historical plot summaries attributed to the CMU Movie Summary Corpus and 88 plot summaries from current Wikipedia articles. The source package identifies the historical text as CC BY-SA 3.0 US and the Wikipedia text as CC BY-SA 4.0; row-level provenance and license fields are in the CSV. The corpus has 530 poster or release-cover URLs. Image rights are separate from plot-text licenses: the source package marks most images as fair use, so the text license does not grant permission to reuse posters. Review the row-level fields and your instructor's requirements before making a public release.

At runtime, `retrieval.py` reads and validates the CSV, filters records by supported hard constraints, excludes previously recommended IDs, and ranks a shortlist with lexical TF-IDF. The LLM does not invent the underlying corpus records: it selects from that shortlist, and code validates record IDs, constraints, duplicates, and evidence-snippet IDs before results are returned.

## Workflow at a glance

```mermaid
flowchart LR
    A[User message + previous constraints] --> B[LangGraph planner]
    B --> C[search_movies tool]
    C --> D[CSV validation and hard filters]
    D --> E[TF-IDF shortlist]
    E --> F[LLM reranking and English explanations]
    F --> G[Python validation of IDs, filters, evidence, duplicates]
    G --> H[Movie cards + updated state]
```

The workflow retains state between turns, including supported preferences and seen movie IDs. “Another one” requests one new result while preserving prior constraints. The graph limits correction attempts and does not silently relax hard filters. It uses a lexical search shortlist rather than vector retrieval. Mood matching and the semantic strength of explanations still require human evaluation.

## Assignment readiness: current evidence and remaining work

| Criterion | Current evidence | Status / next step |
|---|---|---|
| Problem definition and scope | A focused film-discovery problem is described above. | Add a formal problem statement, objectives, target users, and success measures to the report. |
| Corpus curation and ingestion | 584 records; row-level source/license fields; CSV schema checks and deduplication. | Explain collection, selection, cleaning, and license review in the report. Verify that the teaching team accepts these plot summaries as “curated artifacts.” |
| Agent architecture and state | LangGraph plan/search/compose/verify flow; conversation constraints and seen-ID state; bounded retries. | Include a diagram and justify why deterministic Python filtering surrounds LLM decisions. |
| Orchestration and implementation | Tool calling, typed schemas, validation, and automated tests are included. | Run the tests on the final English version and describe failures/fixes in the report. |
| Agent versus baseline evaluation | The supplied report records 37 automated tests and a three-turn real-API smoke test with a single-prompt baseline. | This is not a strong effectiveness study. Run a larger, balanced set of English prompts; compare constraint satisfaction, recommendation relevance, evidence support, latency, and cost, with human scoring for subjective measures. Disclose that the baseline has no retrieval corpus or recommendation-history state. |
| Media analytics | Poster links are displayed as metadata; the current recommender does not analyze poster pixels or other media signals. | The assignment says teams may select techniques, but this is a possible gap if the instructor expects media analytics. Either add a small, defensible visual-analysis component or explain the text-first scope and ask the instructor whether that is sufficient. |
| Written report and presentation | No final report or slides are included in this code package. | The assignment separately requires a cover page, at least 2,000 words, complete references, and a 20-minute presentation. Prepare these deliverables; the code package alone is not a complete submission. |

The supplied `TEST_REPORT.md` describes 37 passing automated tests and a three-turn real-API smoke test recorded on 2026-10-29. It explicitly states that this is not a full effectiveness evaluation or browser integration test. Those results predate the English UI/prompt edit in this package and must not be presented as a live re-test of this exact revision. The English version has only had a Python syntax check so far. `evaluate.py` can make real API calls and incur charges; its manual evaluation fields are intentionally blank.

## Known limitations

- The corpus contains 584 titles and is not comprehensive; a zero-result response does not prove that no matching film exists.
- “Avoid gore” cannot be reliably checked because the corpus has no dependable gore labels. The system should disclose that limitation rather than infer safety.
- Non-sequel review coverage is limited; unknown entries are conservatively excluded when that condition is requested.
- Plot summaries and metadata may contain source errors. Poster links can be unavailable, and image rights require separate review.
- A cited plot excerpt is checked against the source record, but this does not prove that the generated explanation is semantically supported by that excerpt.

## References

- Carnegie Mellon University, *Movie Summary Corpus*: https://www.cs.cmu.edu/~ark/personas/
- Creative Commons, *Attribution-ShareAlike 3.0 United States*: https://creativecommons.org/licenses/by-sa/3.0/us/
- Creative Commons, *Attribution-ShareAlike 4.0 International*: https://creativecommons.org/licenses/by-sa/4.0/
- OpenAI, *Function calling*: https://developers.openai.com/api/docs/guides/function-calling
- LangChain, *LangGraph documentation*: https://docs.langchain.com/oss/python/langgraph/overview
- Streamlit, *Community Cloud deployment*: https://docs.streamlit.io/deploy/streamlit-community-cloud

