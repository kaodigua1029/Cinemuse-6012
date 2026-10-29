import base64
import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from agent import ConfigurationError, DataError, ProviderError, run_agent, run_baseline

HERE = Path(__file__).parent

st.set_page_config(page_title="CineMuse · AI Film Concierge", page_icon="🎞️", layout="wide")
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
:root {--ink:#292721;--muted:#777268;--line:#e6e0d5;--paper:#f8f6f0;--accent:#8b6048;}
.stApp {background:var(--paper);color:var(--ink);}
.block-container {max-width:1240px;padding-top:2rem;padding-bottom:4rem;}
[data-testid="stSidebar"] {background:#f0ede5;border-right:1px solid var(--line);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:var(--ink);}
.hero {padding:18px 0 24px;border-bottom:1px solid var(--line);margin-bottom:24px;}
.eyebrow {font-size:.72rem;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-weight:700;}
.hero h1 {font-family:Georgia,'Times New Roman',serif;font-size:3rem;line-height:1.05;letter-spacing:-.04em;margin:8px 0;color:var(--ink);}
.hero p {color:var(--muted);font-size:1rem;margin:0;}
.section-kicker {font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:18px 0 8px;}
.m-title {font-family:Georgia,'Times New Roman',serif;font-size:1.25rem;font-weight:700;margin-top:8px;}
.m-year {font-family:inherit;color:var(--muted);font-size:.95rem;font-weight:400;}
.m-meta {color:var(--muted);margin:4px 0 8px;font-size:.88rem;}
.tag {display:inline-block;padding:3px 10px;margin:0 5px 6px 0;border-radius:999px;font-size:.75rem;background:#eee8de;color:#665545;border:1px solid #e3dacd;}
.m-reason {margin-top:8px;line-height:1.55;}
.m-evidence {margin-top:10px;font-size:.78rem;color:#716d64;border-left:2px solid #b99a7b;padding-left:10px;line-height:1.45;}
div[data-testid="stChatMessage"] {border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.44);padding:1rem;}
.stButton button {border-radius:999px;border-color:#d9d0c3;color:#685646;background:#fffdf8;}
.stChatInput textarea {background:#fffdf8;}
</style>
""", unsafe_allow_html=True)

LABELS = {
    "genres":"Genres", "exclude_genres":"Excluded genres", "min_year":"From year", "max_year":"Through year",
    "min_runtime":"Minimum runtime (min)", "max_runtime":"Maximum runtime (min)", "limit":"Movies this turn",
    "query_text":"Theme / topic", "sort":"Sort order", "exclude_sequels":"Exclude sequels", "avoid_gore":"Avoid gore",
    "unsupported_constraints":"Requirements we cannot verify", "seen_movie_ids":"Previously recommended records",
}

@st.cache_data
def placeholder_uri():
    svg = (HERE / "poster_placeholder.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()


def poster(movie):
    url = movie.get("poster_url") or ""
    placeholder = placeholder_uri()
    page = f'''<style>body{{margin:0;background:transparent}}img{{width:100%;height:260px;object-fit:contain;border-radius:8px;background:#eeeae1}}</style>
<img src="{html.escape(url or placeholder, quote=True)}" alt="Movie poster" loading="lazy" referrerpolicy="no-referrer" onerror="this.onerror=null;this.src='{placeholder}'">'''
    try:
        st.iframe(page, height=266)
    except Exception:
        components.html(page, height=266)


def movie_card(movie):
    esc = html.escape
    with st.container(border=True):
        poster(movie)
        tags = "".join(f'<span class="tag">{esc(g.strip())}</span>' for g in (movie.get("genres") or "").split("/") if g.strip())
        st.markdown(
            f'<div class="m-title">{esc(str(movie["title"]))} <span class="m-year">({movie["year"]})</span></div>'
            f'<div class="m-meta">{movie["runtime"]} min</div><div>{tags}</div>'
            f'<div class="m-reason">{esc(movie.get("reason", ""))}</div>'
            f'<div class="m-evidence">{esc(movie.get("evidence", ""))}</div>', unsafe_allow_html=True)
        links=[]
        if movie.get("source_url"): links.append(f'[Plot source]({movie["source_url"]})')
        if movie.get("poster_source_url"): links.append(f'[Poster source ({movie.get("poster_license") or "unspecified"})]({movie["poster_source_url"]})')
        if links: st.caption(" · ".join(links))


def show_result(result):
    if not result["movies"]:
        st.info("No matching titles were found in the curated corpus. Try relaxing a requirement; this does not mean no such films exist.")
        return
    for start in range(0, len(result["movies"]), 3):
        cols=st.columns(3)
        for col,movie in zip(cols,result["movies"][start:start+3]):
            with col: movie_card(movie)


def friendly_error(exc):
    if isinstance(exc, ConfigurationError):
        return "The API key is missing. Add OPENAI_API_KEY to the project’s .env file (local) or the app’s Streamlit secrets (deployed), then restart the app."
    if isinstance(exc, DataError):
        return "The movie dataset could not be loaded. Check that movies.csv is in the project folder."
    if isinstance(exc, ProviderError):
        return "The model request failed. Check the API key, model access, API credits, and network connection."
    return "The request could not be completed. Check the requirements and try again."


def parse_warning(query, constraints):
    q=query.lower()
    if any(w in q for w in ["hour","hr"]) and not any(k in constraints for k in ["max_runtime","min_runtime"]):
        return "You mentioned a runtime, but the system did not confirm a specific limit. Try wording it as ‘under 120 minutes.’"
    return None


if "messages" not in st.session_state: st.session_state.messages=[]
if "constraints" not in st.session_state: st.session_state.constraints={}
if "pending_query" not in st.session_state: st.session_state.pending_query=None

with st.sidebar:
    st.markdown("### Your preferences")
    current={k:v for k,v in st.session_state.constraints.items() if v not in ([],"",None) and k!="seen_movie_ids"}
    if current:
        for key,value in current.items():
            shown=" / ".join(map(str,value)) if isinstance(value,list) else value
            st.write(f"**{LABELS.get(key,key)}:** {shown}")
    else: st.caption("No preferences yet. Tell me a theme, genre, or runtime.")
    st.divider()
    compare=st.checkbox("Show a standard AI baseline answer, too")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages=[]; st.session_state.constraints={}; st.rerun()
    st.caption("Recommendations come from the curated movie corpus. The LLM cannot bypass hard filters.")

st.markdown('''<div class="hero"><div class="eyebrow">CINEMUSE · AI FILM CONCIERGE</div>
<h1>What kind of film are you in the mood for?</h1><p>Think of CineMuse as your personal film curator: it listens to what you like and finds a story worth watching from a curated collection.</p></div>''',unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Start with a theme</div>',unsafe_allow_html=True)
ideas=[("Dreamlike sci-fi","Recommend an imaginative, atmospheric science-fiction film."),("Tender romance","I’d like a gentle, not-too-heavy romance."),("Women’s stories","Recommend a film about women’s growth and self-discovery."),("Mystery night","Recommend a mystery film under 120 minutes.")]
idea_cols=st.columns(4)
for col,(label,prompt) in zip(idea_cols,ideas):
    with col:
        if st.button(label,use_container_width=True,key="idea_"+label): st.session_state.pending_query=prompt

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"]=="user": st.write(msg["content"])
        else:
            if msg.get("warning"): st.warning(msg["warning"])
            if msg.get("error"): st.error(msg["error"])
            if msg.get("content"): show_result(msg["content"])
            if msg.get("baseline"): st.markdown("**Standard AI baseline answer**"); st.write(msg["baseline"])

query=st.chat_input("For example: Recommend three sci-fi films under two hours")
if st.session_state.pending_query:
    query=st.session_state.pending_query
    st.session_state.pending_query=None
if query:
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"): st.write(query)
    result=None; error_text=None
    with st.chat_message("assistant"):
        try:
            with st.spinner("Understanding your request and searching the collection…"):
                result=run_agent(query,st.session_state.constraints)
            show_result(result)
        except (ConfigurationError,DataError,ProviderError,ValueError) as exc:
            error_text=friendly_error(exc)
            st.error(error_text)
        except Exception:
            error_text="The request could not be completed. Check the app logs or try again."
            st.error(error_text)
    baseline_text=None
    if compare and result is not None:
        try: baseline_text=run_baseline(query)
        except (ConfigurationError,DataError,ProviderError,ValueError) as exc: baseline_text=friendly_error(exc)
    if result is not None: st.session_state.constraints=result["constraints"]
    st.session_state.messages.append({"role":"assistant","content":result,"warning":parse_warning(query,result["constraints"]) if result else None,"baseline":baseline_text,"error":error_text})
    st.rerun()
