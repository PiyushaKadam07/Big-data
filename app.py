"""
Hybrid Indexing System - Streamlit app
CS63016 · Kent State University
Shahbaz Shaikh · Piyusha Kadam · Mohammed Rameez Usman

Run with:  streamlit run app.py
"""

import html
import os
import random
import re
import statistics
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from dataset_generator import TOPICS, generate_dataset  # noqa: E402
from hybrid_index import (  # noqa: E402
    SAMPLE_DOCS, TOKEN_PATTERN, Document, HybridIndexingSystem, tokenize,
)

st.set_page_config(page_title="Hybrid Indexing", page_icon="🔎", layout="wide")

# ── Look and feel ─────────────────────────────────────────────────────────────

MODES = {
    "hybrid": {"name": "Hybrid", "color": "#0d9488",
               "how": "Checks the directory, then asks only the servers that have your words."},
    "distributed": {"name": "Distributed", "color": "#ea580c",
                    "how": "Asks every server, every time."},
    "centralized": {"name": "Centralized", "color": "#7c3aed",
                    "how": "One big server holds everything."},
}
GREY, INK, MUTED = "#d1d5db", "#111827", "#6b7280"

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  .stMarkdown p.headline { font-size: 1.35rem; line-height: 1.5; margin: 0.5rem 0 1rem; }
  .stMarkdown p.mode-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0; }
  .stMarkdown p.mode-how { color: #6b7280; font-size: 0.9rem; margin: 0 0 0.25rem; min-height: 3.4em; line-height: 1.6; }
  .stMarkdown p.mode-count { font-size: 1.6rem; font-weight: 700; margin: 0.25rem 0 0; }
  .card { display: grid; grid-template-columns: 2.2rem 1fr; gap: 0.75rem; padding: 0.9rem 1rem;
          border: 1px solid #e5e7eb; border-radius: 12px; margin-bottom: 0.6rem; background: #fff; }
  .rank { font-weight: 700; font-size: 1.2rem; color: #0d9488; text-align: center; }
  .stMarkdown p.doc { font-size: 1.05rem; margin: 0 0 0.35rem; color: #111827; }
  .stMarkdown p.why { color: #4b5563; font-size: 0.9rem; margin: 0 0 0.4rem; }
  .meter { height: 6px; background: #f3f4f6; border-radius: 3px; overflow: hidden; max-width: 260px; }
  .meter span { display: block; height: 100%; background: #0d9488; }
  .chip { display: inline-block; padding: 1px 8px; border-radius: 999px; background: #f3f4f6;
          color: #374151; font-size: 0.8rem; margin-left: 6px; }
  mark { background: #fde68a; color: inherit; padding: 0 2px; border-radius: 3px; }
  .stMarkdown p.big-stat { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0; }
  .stMarkdown p.big-label { color: #4b5563; margin: 0.2rem 0 0; }
</style>
""", unsafe_allow_html=True)


def server_name(node_id: str) -> str:
    """node_0 -> Server 1 (people count from one)."""
    return f"Server {int(node_id.split('_')[-1]) + 1}"


# ── Building the system ───────────────────────────────────────────────────────

PLACEMENTS = {
    "By topic (similar documents together)": "topic",
    "Randomly (spread evenly by hashing)": "hash",
}


@st.cache_resource(show_spinner=False)
def build_system(num_nodes: int, strategy: str) -> HybridIndexingSystem:
    system = HybridIndexingSystem(num_nodes=num_nodes, strategy=strategy)
    for i, (content, meta) in enumerate(SAMPLE_DOCS):
        system.add_document(Document(doc_id=f"doc_{i:03d}", content=content, metadata=meta))
    return system


with st.sidebar:
    st.markdown("## 🔎 Hybrid Indexing")
    st.caption("CS63016 · Kent State University")
    st.markdown("---")
    page = st.radio("Go to", ["Try a search", "How it works", "What happens at scale", "Look inside the servers"])
    st.markdown("---")
    st.markdown("**Setup**")
    num_nodes = st.select_slider("Number of servers", options=[2, 4, 6, 8], value=4)
    placement_label = st.radio("How documents are placed on servers", list(PLACEMENTS), index=0,
                               help="Topic placement keeps related documents on the same server, "
                                    "so the directory can rule out more servers.")
    strategy = PLACEMENTS[placement_label]
    st.caption(f"Searching {len(SAMPLE_DOCS)} sample documents.")
    st.markdown("---")
    st.markdown("**Team**  \nShahbaz Shaikh  \nPiyusha Kadam  \nMohammed Rameez Usman")

system = build_system(num_nodes, strategy)


# ── Drawing the journey of a search ───────────────────────────────────────────

def matches_per_node(terms: list[str]) -> dict[str, int]:
    return {nid: len({d for t in terms for d in node.inverted_index.get(t, {})})
            for nid, node in system.nodes.items()}


def journey_svg(mode: str, asked: set[str], hits: dict[str, int], total_hits: int) -> str:
    """A small picture: the search, (the directory), and which servers were asked."""
    color = MODES[mode]["color"]
    W, box_w, box_h, gap, per_row = 340, 76, 56, 8, 4
    parts, cx = [], W / 2
    top_y = 34  # room for the directory box (kept for every mode so the pictures line up)

    def text(x, y, words, size=12, weight=400, fill=INK):
        return (f"<text x='{x}' y='{y}' text-anchor='middle' font-size='{size}' "
                f"font-weight='{weight}' fill='{fill}'>{html.escape(words)}</text>")

    parts.append(f"<rect x='{cx - 60}' y='4' width='120' height='26' rx='13' fill='{INK}'/>")
    parts.append(text(cx, 22, "Your search", 14, 600, "#ffffff"))
    source_y = 30
    if mode == "hybrid":
        parts.append(f"<line x1='{cx}' y1='30' x2='{cx}' y2='44' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<rect x='{cx - 70}' y='44' width='140' height='26' rx='6' fill='#ffffff' "
                     f"stroke='{color}' stroke-width='1.5'/>")
        parts.append(text(cx, 62, "Directory", 14, 600, color))
        source_y = 70

    if mode == "centralized":
        y = 70 + top_y
        parts.append(f"<line x1='{cx}' y1='{source_y}' x2='{cx}' y2='{y}' stroke='{color}' stroke-width='2'/>")
        parts.append(f"<rect x='{cx - 120}' y='{y}' width='240' height='{box_h}' rx='8' fill='{color}' "
                     f"fill-opacity='0.12' stroke='{color}' stroke-width='1.5'/>")
        parts.append(text(cx, y + 23, "One big server", 14, 600))
        parts.append(text(cx, y + 42, f"all {system.total_docs} documents · {total_hits} found", 13, 400, MUTED))
        height = 92 + top_y + box_h + 8
    else:
        ids = list(system.nodes)
        rows = [ids[i:i + per_row] for i in range(0, len(ids), per_row)]
        y0 = 92 + top_y
        for r, row in enumerate(rows):
            row_w = len(row) * box_w + (len(row) - 1) * gap
            for c, nid in enumerate(row):
                x = (W - row_w) / 2 + c * (box_w + gap)
                y = y0 + r * (box_h + 26)
                if nid in asked:
                    parts.append(f"<line x1='{cx}' y1='{source_y}' x2='{x + box_w / 2}' y2='{y}' "
                                 f"stroke='{color}' stroke-width='1.5' opacity='0.8'/>")
                    parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='8' fill='{color}' "
                                 f"fill-opacity='0.12' stroke='{color}' stroke-width='1.5'/>")
                    parts.append(text(x + box_w / 2, y + 23, server_name(nid), 13, 600))
                    found = hits.get(nid, 0)
                    parts.append(text(x + box_w / 2, y + 42, f"{found} found", 13, 400, MUTED))
                else:
                    parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='{box_h}' rx='8' fill='#f9fafb' "
                                 f"stroke='{GREY}' stroke-dasharray='4 3'/>")
                    parts.append(text(x + box_w / 2, y + 23, server_name(nid), 13, 600, "#9ca3af"))
                    parts.append(text(x + box_w / 2, y + 42, "skipped", 13, 400, "#9ca3af"))
        height = y0 + len(rows) * (box_h + 26) - 18
    return (f"<svg viewBox='0 0 {W} {height}' width='100%' role='img' "
            f"aria-label='{MODES[mode]['name']} search path' style='max-width:{W}px'>{''.join(parts)}</svg>")


def highlight(content: str, terms: set[str]) -> str:
    """Wrap the words that matched the search in <mark>."""
    out, last, lower = [], 0, content.lower()
    for m in TOKEN_PATTERN.finditer(lower):
        if m.group() in terms:
            out.append(html.escape(content[last:m.start()]))
            out.append(f"<mark>{html.escape(content[m.start():m.end()])}</mark>")
            last = m.end()
    out.append(html.escape(content[last:]))
    return "".join(out)


def explain(doc: Document, terms: list[str]) -> str:
    """Why a document is in the results, in plain words."""
    stats = system.metadata.stats()
    present = [t for t in dict.fromkeys(terms) if t in set(tokenize(doc.content))]
    words = " and ".join(f"<b>{html.escape(t)}</b>" for t in present)
    why = f"Contains {words}"
    rarest = min(present, key=lambda t: stats.doc_freq.get(t, 0)) if present else None
    if rarest and len(set(terms)) > 1:
        n = stats.doc_freq.get(rarest, 0)
        why += f" · “{html.escape(rarest)}” is a rare word here ({n} of {stats.doc_count} documents), so it counts more"
    return why


# ── Page: Try a search ────────────────────────────────────────────────────────

def page_search():
    st.title("Find documents spread across many servers")
    st.markdown(
        "Large collections are too big for one computer, so they're split across several **servers**. "
        "This compares three ways of searching them. All three find the **same documents** — "
        "the difference is how much work it takes.")

    examples = ["consistent hashing", "NoSQL databases", "fault tolerance replication",
                "distributed indexing nodes", "real time indexing", "B-Tree indexes"]

    def pick():
        if st.session_state.example:
            st.session_state.query = st.session_state.example

    st.session_state.setdefault("query", examples[0])
    st.text_input("What are you looking for?", key="query")
    st.pills("Or try one of these", examples, key="example", on_change=pick)

    query = st.session_state.query.strip()
    terms = tokenize(query)
    if not terms:
        st.info("Type a few words to search.")
        return

    runs = {"hybrid": system.hybrid_search(query, top_k=5),
            "distributed": system.distributed_search(query, top_k=5),
            "centralized": system.centralized_search(query, top_k=5)}
    found = runs["hybrid"]["total_matches"]
    asked = runs["hybrid"]["nodes_queried"]
    n = len(system.nodes)

    if found == 0:
        headline = f"No document contains those words. The directory knew that, so hybrid asked **no servers at all**."
    elif asked < n:
        headline = (f"All three found the same **{found} document{'s' if found != 1 else ''}**. "
                    f"Hybrid asked **{asked} of {n} servers** and skipped **{n - asked}** that had nothing to offer.")
    else:
        headline = (f"All three found the same **{found} documents**. Your words appear on every server, "
                    f"so hybrid had to ask all {n}, the same as distributed. Hybrid saves the most on specific searches.")
    headline = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", headline)
    st.markdown(f"<p class='headline'>{headline}</p>", unsafe_allow_html=True)

    hits = matches_per_node(terms)
    hybrid_asked = set(system.metadata.route_query(terms))
    cols = st.columns(3)
    for col, mode in zip(cols, ["hybrid", "distributed", "centralized"]):
        r, info = runs[mode], MODES[mode]
        with col:
            st.markdown(f"<p class='mode-title' style='color:{info['color']}'>{info['name']}</p>"
                        f"<p class='mode-how'>{info['how']}</p>", unsafe_allow_html=True)
            asked_set = hybrid_asked if mode == "hybrid" else set(system.nodes)
            st.markdown(journey_svg(mode, asked_set, hits, r["total_matches"]), unsafe_allow_html=True)
            servers = "1 server" if mode == "centralized" else f"{r['nodes_queried']} of {n} servers"
            st.markdown(f"<p class='mode-count'>Asked {servers}</p>", unsafe_allow_html=True)

    st.markdown("### What it found")
    results = runs["hybrid"]["results"]
    if not results:
        st.write("Nothing matched. Try fewer or different words.")
        return
    best = results[0].score
    term_set = set(terms)
    for i, r in enumerate(results, 1):
        doc = system.get_document(r.doc_id)
        topic = doc.metadata.get("category", "")
        st.markdown(
            f"<div class='card'><div class='rank'>{i}</div><div>"
            f"<p class='doc'>{highlight(doc.content, term_set)}</p>"
            f"<p class='why'>{explain(doc, terms)}</p>"
            f"<div class='meter' title='Match strength'><span style='width:{100 * r.score / best:.0f}%'></span></div>"
            f"<p class='why' style='margin-top:0.35rem'>Stored on {server_name(r.node_id)}"
            f"<span class='chip'>{html.escape(topic)}</span></p>"
            f"</div></div>", unsafe_allow_html=True)
    if found > len(results):
        st.caption(f"Showing the best {len(results)} of {found} matches. The bar shows how strong each match is "
                   f"compared with the best one.")
    else:
        st.caption("The bar shows how strong each match is compared with the best one.")


# ── Page: How it works ────────────────────────────────────────────────────────

def page_how():
    st.title("How it works")
    st.markdown("Think of it like a library system with several branches.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"#### <span style='color:{MODES['centralized']['color']}'>Centralized</span>", unsafe_allow_html=True)
        st.markdown("**One giant library** with every book. Easy to search, but it gets slow and crowded "
                    "as the collection grows, and if it closes, everything is gone.")
    with c2:
        st.markdown(f"#### <span style='color:{MODES['distributed']['color']}'>Distributed</span>", unsafe_allow_html=True)
        st.markdown("**Several branch libraries.** It scales, but for every question you phone **every** "
                    "branch, even the ones that don't have anything on your topic.")
    with c3:
        st.markdown(f"#### <span style='color:{MODES['hybrid']['color']}'>Hybrid</span>", unsafe_allow_html=True)
        st.markdown("**Branch libraries plus a catalogue** that says which branches have which subjects. "
                    "You check the catalogue first, then phone only the branches that can help.")

    st.markdown("---")
    st.markdown("### What happens when you search, step by step")
    st.markdown(
        "1. **Your words are cleaned up.** “Consistent Hashing!” becomes `consistent`, `hashing`.\n"
        "2. **The directory is checked.** It's a small list of which words live on which servers. "
        "It doesn't store the documents, so it stays small and fast.\n"
        "3. **Only the useful servers are asked**, all at the same time.\n"
        "4. **Each server scores its own matches.** Documents with more of your words score higher, "
        "and rare words count more than common ones.\n"
        "5. **The answers are merged** into one ranked list.")

    example = "consistent hashing"
    terms = tokenize(example)
    st.markdown(f"### What the directory says about “{example}”")
    for t in terms:
        where = sorted(system.metadata.term_to_nodes.get(t, set()))
        names = ", ".join(server_name(n) for n in where) or "no server"
        st.markdown(f"- **{t}** → {names}")
    skipped = len(system.nodes) - len(system.metadata.route_query(terms))
    st.markdown(f"So hybrid can skip **{skipped} of {len(system.nodes)}** servers for this search.")

    st.markdown("---")
    st.markdown("### Why placement matters")
    st.markdown(
        "The directory can only skip a server if that server has **none** of your words. "
        "If documents are scattered randomly, every topic ends up on every server and there's little to skip. "
        "If documents on the same topic are kept together, most servers can be ruled out. "
        "Switch **How documents are placed** in the sidebar and search again to see the difference, "
        "or open **What happens at scale**.")


# ── Page: What happens at scale ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def scale_experiment(num_docs: int) -> dict:
    """Index the same synthetic documents under different setups and count servers asked."""
    docs = generate_dataset(num_docs, seed=42)
    vocab = [t for topic in TOPICS.values() for t in topic["terms"]]
    queries = random.Random(1).sample(vocab, 60)
    sizes = [2, 4, 8, 16]
    out = {"sizes": sizes, "distributed": sizes, "hash": [], "topic": [], "identical": True}
    for strat in ["hash", "topic"]:
        for n in sizes:
            s = HybridIndexingSystem(num_nodes=n, strategy=strat)
            for d in docs:
                s.add_document(Document(d["id"], d["content"], {"category": d["category"]}))
            asked = []
            for q in queries:
                h = s.hybrid_search(q)
                asked.append(h["nodes_queried"])
                if n == 8:
                    c = s.centralized_search(q)
                    if [r.doc_id for r in h["results"]] != [r.doc_id for r in c["results"]]:
                        out["identical"] = False
            out[strat].append(round(statistics.mean(asked), 1))
    return out


def page_scale():
    st.title("What happens at scale")
    st.markdown("Twenty documents is a toy. Here the system is rebuilt with thousands of documents "
                "and up to 16 servers, and 60 different searches are run each time.")
    num_docs = st.select_slider("How many documents", options=[500, 1000, 2000, 5000], value=2000)
    with st.spinner("Building the indexes and running the searches…"):
        r = scale_experiment(num_docs)

    i16 = r["sizes"].index(16)
    topic16, hash16 = r["topic"][i16], r["hash"][i16]
    saved = round(100 * (1 - topic16 / 16))
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<p class='big-stat' style='color:{MODES['distributed']['color']}'>16</p>"
                f"<p class='big-label'>servers asked per search by <b>distributed</b>, with 16 servers</p>",
                unsafe_allow_html=True)
    c2.markdown(f"<p class='big-stat' style='color:#9ca3af'>{hash16:g}</p>"
                f"<p class='big-label'>asked by <b>hybrid</b> when documents are placed randomly</p>",
                unsafe_allow_html=True)
    c3.markdown(f"<p class='big-stat' style='color:{MODES['hybrid']['color']}'>{topic16:g}</p>"
                f"<p class='big-label'>asked by <b>hybrid</b> when documents are grouped by topic — "
                f"<b>{saved}% fewer</b> than distributed</p>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r["sizes"], y=r["distributed"], name="Distributed (asks every server)",
                             mode="lines+markers", line=dict(color=MODES["distributed"]["color"], width=3)))
    fig.add_trace(go.Scatter(x=r["sizes"], y=r["hash"], name="Hybrid, documents placed randomly",
                             mode="lines+markers", line=dict(color="#9ca3af", width=3)))
    fig.add_trace(go.Scatter(x=r["sizes"], y=r["topic"], name="Hybrid, documents grouped by topic",
                             mode="lines+markers", line=dict(color=MODES["hybrid"]["color"], width=3)))
    fig.update_layout(
        title="Servers asked per search as the system grows (lower is better)",
        xaxis=dict(title="Servers in the system", tickvals=r["sizes"]),
        yaxis=dict(title="Servers asked per search", rangemode="tozero"),
        legend=dict(orientation="h", y=-0.25), height=440, margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    fig.update_xaxes(gridcolor="#f3f4f6")
    fig.update_yaxes(gridcolor="#f3f4f6")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### What this shows")
    st.markdown(
        f"- **Distributed** work grows with every server you add: 16 servers means 16 messages for every search.\n"
        f"- **Hybrid with random placement** helps only a little (about {hash16:g} of 16), because every topic "
        f"ends up on almost every server.\n"
        f"- **Hybrid with topic placement** stays almost flat at about {topic16:g} servers, however big the "
        f"system gets. That's the real benefit of the design.")
    if r["identical"]:
        st.success("Check passed: hybrid and centralized returned exactly the same results for every test search.")
    st.caption("Speed isn't shown in milliseconds here: on one laptop every search takes well under a millisecond. "
               "On a real network, each server asked costs a round trip, so fewer servers asked means faster, "
               "cheaper searches. The documents are generated from templates, so treat this as a demonstration.")


# ── Page: Look inside the servers ─────────────────────────────────────────────

def page_inside():
    st.title("Look inside the servers")
    counts = {server_name(nid): len(node.documents) for nid, node in system.nodes.items()}
    fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), marker_color=MODES["hybrid"]["color"],
                           text=list(counts.values()), textposition="outside"))
    fig.update_layout(title="Documents on each server", height=300, plot_bgcolor="#ffffff",
                      yaxis=dict(visible=False), margin=dict(l=20, r=20, t=50, b=30))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        choice = st.selectbox("Pick a server to see what it holds", list(system.nodes), format_func=server_name)
        node = system.nodes[choice]
        topics = sorted({d.metadata.get("category", "") for d in node.documents.values()})
        st.markdown(f"**{server_name(choice)}** holds {len(node.documents)} documents "
                    f"about: {', '.join(topics)}.")
        for doc in node.documents.values():
            st.markdown(f"<div class='card' style='grid-template-columns:1fr'><div>"
                        f"<p class='doc'>{html.escape(doc.content)}</p>"
                        f"<span class='chip' style='margin-left:0'>{html.escape(doc.metadata.get('category', ''))}</span>"
                        f"</div></div>", unsafe_allow_html=True)
    with right:
        st.markdown("#### Ask the directory")
        word = st.text_input("Which servers have this word?", placeholder="e.g. hashing, b-tree, search")
        if word:
            terms = tokenize(word)
            if not terms:
                st.write("That's a filler word the index ignores. Try another.")
            for t in terms:
                where = sorted(system.metadata.term_to_nodes.get(t, set()))
                if where:
                    st.markdown(f"**{t}** is on {', '.join(server_name(n) for n in where)}.")
                else:
                    st.markdown(f"**{t}** isn't on any server.")


{"Try a search": page_search, "How it works": page_how,
 "What happens at scale": page_scale, "Look inside the servers": page_inside}[page]()
