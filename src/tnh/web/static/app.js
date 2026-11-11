// tn-hittare interface: board, database statistics and model output for the position shown.

import {Chessboard, COLOR, INPUT_EVENT_TYPE} from "./vendor/cm-chessboard/src/Chessboard.js"
import {Markers} from "./vendor/cm-chessboard/src/extensions/markers/Markers.js"
import {Arrows, ARROW_TYPE} from "./vendor/cm-chessboard/src/extensions/arrows/Arrows.js"
import {PromotionDialog} from "./vendor/cm-chessboard/src/extensions/promotion-dialog/PromotionDialog.js"
import {Chess} from "./vendor/chess.js/chess.js"
import {MoveTree, START_FEN} from "./tree.js"

const $ = id => document.getElementById(id)
const LAST_MOVE = {class: "marker-last", slice: "markerSquare"}
const CURSOR_ARROW = {class: "arrow-cursor"}
const MAIA_MIN = 0.01        // Maia moves below this are not listed unless "all legal"
const DELTA_MIN_GAMES = 20   // no Δ column below this many games in the position

const state = {
    tree: new MoveTree(),
    orientation: COLOR.white,
    filters: {y0: "", y1: "", e0: "", e1: "", tempi: null},
    me: 2200, opp: 2200, nodes: 200000, sfauto: false,
    showAll: false,
    sort: {key: "n", desc: true},
    cursor: null,     // uci of the highlighted row
    stats: null, maia: null, engine: null,
    info: null,
}
const cache = {stats: new Map(), maia: new Map(), engine: new Map()}
const inflight = {stats: null, maia: null, engine: null}
const timing = {}

// ---------------------------------------------------------------- URL state

function readUrl() {
    const p = new URLSearchParams(location.hash.slice(1))
    const f = state.filters
    for (const k of ["y0", "y1", "e0", "e1"]) if (p.has(k)) f[k] = p.get(k)
    if (p.has("t")) f.tempi = p.get("t") ? p.get("t").split(",") : []
    if (p.has("me")) state.me = +p.get("me")
    if (p.has("opp")) state.opp = +p.get("opp")
    if (p.has("nodes")) state.nodes = +p.get("nodes")
    if (p.has("sf")) state.sfauto = p.get("sf") === "1"
    if (p.get("o") === "b") state.orientation = COLOR.black
    try {
        state.tree = new MoveTree(p.get("fen") || START_FEN)
    } catch (e) {
        status("bad FEN in URL: " + e.message, true)
        state.tree = new MoveTree()
    }
    for (const m of (p.get("m") || "").split(".").filter(Boolean)) {
        if (!state.tree.play(m)) {
            status("illegal move in URL: " + m, true)
            break
        }
    }
}

function writeUrl() {
    const p = new URLSearchParams()
    if (!state.tree.fromStart) p.set("fen", state.tree.root.fen)
    const path = state.tree.path()
    if (path.length) p.set("m", path.join("."))
    const f = state.filters
    for (const k of ["y0", "y1", "e0", "e1"]) if (f[k] !== "") p.set(k, f[k])
    if (f.tempi) p.set("t", f.tempi.join(","))
    p.set("me", state.me)
    p.set("opp", state.opp)
    if (state.nodes !== 200000) p.set("nodes", state.nodes)
    if (state.sfauto) p.set("sf", "1")
    if (state.orientation === COLOR.black) p.set("o", "b")
    history.replaceState(null, "", "#" + p.toString().replace(/%2C/g, ","))
}

// ---------------------------------------------------------------- formatting

const pct = (x, d = 0) => x == null ? "" : (100 * x).toFixed(d)
const int = x => x == null ? "" : Math.round(x).toString()
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]))

function cpText(sf) {
    if (!sf) return ""
    if (sf.mate) return (sf.mate > 0 ? "M" : "-M") + Math.abs(sf.mate)
    if (sf.cp == null) return "?"
    const v = sf.cp / 100
    return (v > 0 ? "+" : "") + v.toFixed(2)
}

function sparkline(series, totals, flag) {
    const W = 64, H = 14
    const pts = series.map((n, i) => totals[i] >= 3 ? n / totals[i] : null)
    const vals = pts.filter(v => v != null)
    if (vals.length < 2) return ""
    const top = Math.max(0.05, ...vals)
    const step = W / Math.max(1, pts.length - 1)
    let d = "", pen = false
    pts.forEach((v, i) => {
        if (v == null) { pen = false; return }
        const x = (i * step).toFixed(1), y = (H - 1 - (v / top) * (H - 2)).toFixed(1)
        d += (pen ? "L" : "M") + x + " " + y
        pen = true
    })
    const cls = flag > 0 ? "up" : flag < 0 ? "down" : ""
    return `<svg class="spark ${cls}" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><path d="${d}"/></svg>`
}

// ---------------------------------------------------------------- board

let board

function initBoard() {
    board = new Chessboard($("board"), {
        position: state.tree.current.fen,
        orientation: state.orientation,
        assetsUrl: "/static/vendor/cm-chessboard/assets/",
        style: {
            cssClass: "default", showCoordinates: true, animationDuration: 100,
            pieces: {file: "/static/vendor/chessnut/pieces.svg", tileSize: 40},
        },
        extensions: [
            {class: Markers, props: {autoMarkers: null, sprite: "/static/markers.svg"}},
            {class: Arrows, props: {headSize: 6}},
            {class: PromotionDialog},
        ],
    })
    board.enableMoveInput(onBoardInput)
}

function onBoardInput(event) {
    const chess = new Chess(state.tree.current.fen)
    if (event.type === INPUT_EVENT_TYPE.moveInputStarted) {
        return chess.moves({square: event.squareFrom}).length > 0
    }
    if (event.type === INPUT_EVENT_TYPE.validateMoveInput) {
        const legal = chess.moves({square: event.squareFrom, verbose: true}).filter(m => m.to === event.squareTo)
        if (!legal.length) return false
        if (legal[0].promotion) {
            const color = chess.turn() === "w" ? COLOR.white : COLOR.black
            board.showPromotionDialog(event.squareTo, color, r => {
                if (r && r.type === "pieceSelected") play(event.squareFrom + event.squareTo + r.piece.charAt(1))
                else showPosition()
            })
            return true
        }
        setTimeout(() => play(event.squareFrom + event.squareTo))
        return true
    }
    return true
}

function play(uci) {
    if (!state.tree.play(uci)) {
        status("illegal: " + uci, true)
        return
    }
    update()
}

function showPosition() {
    const n = state.tree.current
    board.setPosition(n.fen, true)
    board.removeMarkers(LAST_MOVE)
    if (n.uci) {
        board.addMarker(LAST_MOVE, n.uci.slice(0, 2))
        board.addMarker(LAST_MOVE, n.uci.slice(2, 4))
    }
}

function showArrow(uci) {
    board.removeArrows(CURSOR_ARROW)
    if (uci) board.addArrow(CURSOR_ARROW, uci.slice(0, 2), uci.slice(2, 4))
}

// ---------------------------------------------------------------- move list

function renderMoves() {
    const root = $("moves")
    root.innerHTML = ""
    const cur = state.tree.current
    const span = (n, num) => {
        const s = document.createElement("span")
        s.className = "mv" + (n === cur ? " cur" : "")
        s.dataset.id = n.id
        const {number, white} = MoveTree.moveNumber(n)
        s.textContent = (white ? number + ". " : num ? number + "... " : "") + n.san
        return s
    }
    const walk = (parent, el, forceNum) => {
        let p = parent, num = forceNum
        while (p.children.length) {
            const main = p.children[0]
            el.appendChild(span(main, num))
            num = false
            for (const v of p.children.slice(1)) {
                const box = document.createElement("span")
                box.className = "var"
                box.appendChild(span(v, true))
                walk(v, box, false)
                el.appendChild(box)
                num = true
            }
            p = main
        }
    }
    walk(state.tree.root, root, true)
    if (!state.tree.root.children.length) root.innerHTML = '<span class="dim">no moves</span>'
    const c = root.querySelector(".cur")
    if (c) c.scrollIntoView({block: "nearest"})
}

// ---------------------------------------------------------------- data

function filterQuery() {
    const f = state.filters, p = new URLSearchParams()
    for (const k of ["y0", "y1", "e0", "e1"]) if (f[k] !== "") p.set(k, f[k])
    if (f.tempi) p.set("tempo", f.tempi.join(","))
    return p.toString()
}

function historyMoves() {
    // Maia and the engine want the moves from the start position; after a pasted FEN
    // they get the position alone.
    return state.tree.fromStart ? state.tree.path() : []
}

async function fetchJson(kind, url) {
    if (inflight[kind]) inflight[kind].abort()
    const ctl = new AbortController()
    inflight[kind] = ctl
    const t = performance.now()
    const r = await fetch(url, {signal: ctl.signal})
    const body = await r.json()
    inflight[kind] = null
    timing[kind] = performance.now() - t
    if (!r.ok) throw new Error(`${kind}: ${body.detail || r.status}`)
    return body
}

function load(kind, key, url, onDone) {
    const hit = cache[kind].get(key)
    if (hit) { onDone(hit); return }
    fetchJson(kind, url).then(d => {
        cache[kind].set(key, d)
        onDone(d)
    }).catch(e => {
        if (e.name !== "AbortError") status(e.message, true)
    })
}

function fetchStats() {
    const fen = state.tree.current.fen
    const q = filterQuery()
    const key = fen + "|" + q
    state.stats = null
    load("stats", key, `/api/stats?fen=${encodeURIComponent(fen)}&${q}`, d => {
        if (state.tree.current.fen !== fen || filterQuery() !== q) return
        state.stats = d
        render()
    })
}

function elos() {
    const meWhite = state.orientation === COLOR.white
    return meWhite ? [state.me, state.opp] : [state.opp, state.me]
}

function fetchMaia() {
    const fen = state.tree.current.fen
    const moves = historyMoves()
    const [w, b] = elos()
    const key = [fen, moves.slice(-7).join("."), moves.length < 7, w, b].join("|")
    state.maia = null
    load("maia", key, `/api/maia?fen=${encodeURIComponent(fen)}&moves=${moves.join(",")}&white=${w}&black=${b}`, d => {
        if (state.tree.current.fen !== fen) return
        state.maia = d
        render()
    })
}

function fetchEngine(force) {
    const fen = state.tree.current.fen
    const key = fen.split(" ").slice(0, 4).join(" ") + "|" + state.nodes
    state.engine = cache.engine.get(key) || null
    if (!state.engine && !force && !state.sfauto) return
    if (new Chess(fen).isGameOver()) return
    const moves = historyMoves()
    state.engineBusy = !state.engine
    load("engine", key, `/api/engine?fen=${encodeURIComponent(fen)}&moves=${moves.join(",")}&nodes=${state.nodes}`, d => {
        state.engineBusy = false
        if (state.tree.current.fen !== fen) return
        state.engine = d
        render()
    })
}

// ---------------------------------------------------------------- table

const COLS = [
    {k: "san", h: "Move", num: false, t: "Move. Italic: not played in the database under these filters"},
    {k: "n", h: "Games", t: "Games that played the move, under the filters"},
    {k: "share", h: "%", t: "Share of the position's games"},
    {k: "score", h: "Score", t: "Score for the side to move, %"},
    {k: "draw", h: "Draw", t: "Draw rate, %"},
    {k: "elo", h: "Elo", t: "Average rating of the player who made the move (games with both ratings)"},
    {k: "opp_elo", h: "Opp", t: "Average rating of the opponent"},
    {k: "perf", h: "Perf", t: "Performance: opponents' average + FIDE dp(score)"},
    {k: "last_year", h: "Last", t: "Last year the move was played"},
    {k: "reach", h: "Reach", t: "Games reaching the position after the move by any move order; unfiltered, all tempi"},
    {k: "trend", h: "Trend", num: false, t: "Share per year. ▲/▼: recent share differs from the earlier share beyond chance"},
    {k: "maia", h: "Maia", t: "Maia-3 probability for the side to move at the set ratings, % (blitz scale)"},
    {k: "delta", h: "Δ", t: "Maia − database share, percentage points"},
    {k: "sf", h: "SF", t: "Stockfish after the move, pawns from the side to move's view"},
]

function rows() {
    const fen = state.tree.current.fen
    const chess = new Chess(fen)
    const legal = new Map(chess.moves({verbose: true}).map(m => [m.from + m.to + (m.promotion || ""), m.san]))
    const s = state.stats, mp = state.maia && state.maia.moves, en = state.engine && state.engine.moves
    const byUci = new Map()
    const add = uci => {
        if (!byUci.has(uci)) byUci.set(uci, {uci, san: legal.get(uci) || uci, db: null})
        return byUci.get(uci)
    }
    if (s) for (const m of s.moves) add(m.uci).db = m
    if (mp) for (const [u, p] of Object.entries(mp)) if (p >= MAIA_MIN || state.showAll) add(u)
    if (state.showAll) for (const u of legal.keys()) add(u)
    let bestSf = null
    if (en) {
        for (const [u, v] of Object.entries(en)) if (v.value != null && (!bestSf || v.value > en[bestSf].value)) bestSf = u
        if (bestSf) add(bestSf)
    }
    const total = s ? s.total.n : 0
    const out = []
    for (const r of byUci.values()) {
        const d = r.db || {}
        r.n = d.n || 0
        r.share = total ? (d.n || 0) / total : null
        for (const k of ["score", "draw", "elo", "opp_elo", "perf", "last_year", "reach"]) r[k] = d[k] ?? null
        r.maia = mp ? (mp[r.uci] ?? 0) : null
        r.delta = r.maia != null && total >= DELTA_MIN_GAMES ? r.maia - r.share : null
        r.sfv = en ? en[r.uci] : null
        r.sf = r.sfv ? r.sfv.value : null
        r.best = r.uci === bestSf
        r.trend = d.trend ? d.trend.recent - d.trend.early : null
        out.push(r)
    }
    const {key, desc} = state.sort
    const val = r => key === "san" ? r.san : r[key]
    out.sort((a, b) => {
        const va = val(a), vb = val(b)
        if (va == null && vb == null) return (b.maia || 0) - (a.maia || 0)
        if (va == null) return 1
        if (vb == null) return -1
        const c = typeof va === "string" ? va.localeCompare(vb) : va - vb
        return (desc ? -c : c) || (b.maia || 0) - (a.maia || 0)
    })
    return out
}

function renderHead() {
    const tr = document.querySelector("#tbl thead tr")
    tr.innerHTML = COLS.map(c => {
        const sorted = state.sort.key === c.k ? (state.sort.desc ? " ▾" : " ▴") : ""
        let t = c.t
        if (c.k === "trend" && state.stats && state.stats.trend) t += ". " + state.stats.trend.test
        return `<th data-k="${c.k}" class="${c.num === false ? "" : "num"}" title="${esc(t)}">${c.h}${sorted}</th>`
    }).join("")
}

function trendCell(r, s) {
    const d = r.db
    if (!d || !s || !s.years.length) return ""
    const tr = d.trend
    const flag = tr ? tr.flag : 0
    const svg = sparkline(d.series, s.year_totals, flag)
    let t = `${s.years[0]}–${s.years[s.years.length - 1]}`
    let mark = ""
    if (tr) {
        t += `: ${pct(tr.early, 1)}% before ${s.trend.recent_from}, ${pct(tr.recent, 1)}% since; z=${tr.z.toFixed(1)}, p=${tr.p.toPrecision(2)}`
        mark = flag > 0 ? '<b class="up">▲</b>' : flag < 0 ? '<b class="down">▼</b>' : ""
    } else if (s.trend && !s.trend.tested) {
        t += ": too few games to test"
    }
    return `<span title="${esc(t)}">${svg}${mark}</span>`
}

function deltaCell(r) {
    if (r.delta == null) return ""
    const v = 100 * r.delta
    const a = Math.min(1, Math.abs(v) / 20)
    const bg = v > 0 ? `rgba(40,120,220,${a * 0.45})` : `rgba(220,90,40,${a * 0.45})`
    const txt = Math.round(v) === 0 ? "0" : (v > 0 ? "+" : "") + v.toFixed(0)
    return `<span class="pill" style="background:${bg}">${txt}</span>`
}

function render() {
    const s = state.stats
    const total = $("totals")
    if (s) {
        const t = s.total
        const w = t.n ? t.white_wins / t.n : null, d = t.n ? t.draws / t.n : null, b = t.n ? t.black_wins / t.n : null
        total.innerHTML = `<b title="Sum of the moves' games under the filters">${t.n}</b> games` +
            (t.reach != null ? ` <span class="dim" title="Games reaching this position, unfiltered, including those that ended here">(${t.reach} reach)</span>` : "") +
            (t.n ? ` · W ${pct(w)} D ${pct(d)} B ${pct(b)} · score ${s.white_to_move ? "White" : "Black"} ${pct(t.score, 1)}%` : "") +
            ` <span class="dim mono" title="Zobrist hash">${s.hash}</span>`
        const mo = s.move_orders
        if (mo && mo.length > 1) {
            $("orders").innerHTML = `Transposition: reached by ${mo.length} different last moves · ` +
                mo.slice(0, 6).map(o => `<span title="${esc(o.parent_fen || "")}">${esc(o.san)} ${o.n}</span>`).join(" · ")
        } else {
            $("orders").textContent = mo == null ? "move orders: not available from this source" : ""
        }
    } else {
        total.innerHTML = '<span class="dim">loading…</span>'
    }
    renderHead()
    const rs = rows()
    if (state.cursor && !rs.find(r => r.uci === state.cursor)) state.cursor = null
    const body = document.querySelector("#tbl tbody")
    body.innerHTML = rs.map(r => {
        const cls = [r.db ? "" : "nodb", r.uci === state.cursor ? "cursor" : ""].join(" ")
        const sfTitle = r.sfv ? `expected score ${pct(r.sfv.value, 1)}%` : ""
        return `<tr data-uci="${r.uci}" class="${cls}">` +
            `<td class="san">${esc(r.san)}</td>` +
            `<td class="num">${r.db ? r.n : ""}</td>` +
            `<td class="num">${r.db ? pct(r.share, 1) : ""}</td>` +
            `<td class="num">${pct(r.score)}</td>` +
            `<td class="num">${pct(r.draw)}</td>` +
            `<td class="num">${int(r.elo)}</td>` +
            `<td class="num">${int(r.opp_elo)}</td>` +
            `<td class="num">${int(r.perf)}</td>` +
            `<td class="num">${r.last_year ?? ""}</td>` +
            `<td class="num dim">${r.reach ?? ""}</td>` +
            `<td>${trendCell(r, s)}</td>` +
            `<td class="num maia">${r.maia == null ? "" : pct(r.maia, 1)}</td>` +
            `<td class="num">${deltaCell(r)}</td>` +
            `<td class="num ${r.best ? "best" : ""}" title="${sfTitle}">${cpText(r.sfv)}</td>` +
            `</tr>`
    }).join("")
    if (!rs.length) body.innerHTML = `<tr><td colspan="${COLS.length}" class="dim">${s ? "no games and no model output for this position" : ""}</td></tr>`
    showArrow(state.cursor)
    renderModels()
    renderStatus()
}

function renderModels() {
    const m = state.maia, e = state.engine
    let h = ""
    if (m) {
        const [w, d, l] = m.wdl
        h += `<span title="Win, draw, loss for the side to move. Trained on online blitz: under-predicts draws in classical games">` +
            `<b>${esc(m.model)}</b> · mover ${m.mover_elo} v ${m.opp_elo} · W ${pct(w)} D ${pct(d)} L ${pct(l)}` +
            (state.tree.fromStart ? "" : ' · <span class="warn" title="Position set from a FEN: Maia sees no earlier positions">no move history</span>') + `</span>`
    } else {
        h += '<span class="dim">Maia: loading…</span>'
    }
    h += "<br>"
    if (e) {
        const wdl = e.wdl ? ` · W ${pct(e.wdl[0])} D ${pct(e.wdl[1])} L ${pct(e.wdl[2])}` : ""
        h += `<b>${esc(e.model)}</b>${wdl}`
    } else if (state.engineBusy) {
        h += '<span class="dim">engine: running…</span>'
    } else {
        h += '<span class="dim">engine: e to run</span>'
    }
    $("models").innerHTML = h
}

function renderStatus() {
    const parts = []
    if (state.stats) parts.push(`stats ${state.stats.ms.toFixed(0)} ms server`)
    for (const k of ["stats", "maia", "engine"]) if (timing[k] != null) parts.push(`${k} ${timing[k].toFixed(0)} ms`)
    if (!$("status").classList.contains("err")) $("status").textContent = parts.join(" · ")
}

function status(msg, err = false) {
    const el = $("status")
    el.textContent = msg
    el.classList.toggle("err", err)
    if (err) console.error(msg)
}

// ---------------------------------------------------------------- update

function update() {
    $("status").classList.remove("err")
    state.cursor = null
    showPosition()
    renderMoves()
    $("fen").textContent = state.tree.current.fen
    writeUrl()
    fetchStats()
    fetchMaia()
    fetchEngine(false)
    render()
}

function setFilter() {
    const f = state.filters
    for (const k of ["y0", "y1", "e0", "e1"]) f[k] = $(k).value
    const boxes = [...document.querySelectorAll("#tempi input")]
    const on = boxes.filter(b => b.checked).map(b => b.value)
    f.tempi = on.length === boxes.length ? null : on
    writeUrl()
    fetchStats()
    render()
}

function initControls() {
    const bands = [""]
    for (let e = 1000; e <= 2800; e += 200) bands.push(String(e))
    for (const id of ["e0", "e1"]) {
        $(id).innerHTML = bands.map(b => `<option value="${b}">${b || "any"}</option>`).join("")
        $(id).value = state.filters[id]
    }
    $("y0").value = state.filters.y0
    $("y1").value = state.filters.y1
    const tempi = (state.info && state.info.tempi) || ["classical", "rapid", "blitz", "unknown"]
    $("tempi").innerHTML = "Tempo " + tempi.map(t =>
        `<label title="${t}"><input type="checkbox" value="${t}" ${!state.filters.tempi || state.filters.tempi.includes(t) ? "checked" : ""}>${t[0].toUpperCase() + (t === "unknown" ? "?" : t.slice(1, 3))}</label>`
    ).join("")
    for (const id of ["y0", "y1", "e0", "e1"]) $(id).addEventListener("change", setFilter)
    $("tempi").addEventListener("change", setFilter)
    $("me").value = state.me
    $("opp").value = state.opp
    for (const id of ["me", "opp"]) $(id).addEventListener("change", () => {
        state[id] = +$(id).value
        writeUrl()
        fetchMaia()
        render()
    })
    $("nodes").value = String(state.nodes)
    $("nodes").addEventListener("change", () => { state.nodes = +$("nodes").value; writeUrl(); fetchEngine(true); render() })
    $("sfauto").checked = state.sfauto
    $("sfauto").addEventListener("change", () => { state.sfauto = $("sfauto").checked; writeUrl(); fetchEngine(false); render() })

    $("nav").addEventListener("click", e => { if (e.target.dataset.k) command(e.target.dataset.k) })
    $("moves").addEventListener("click", e => {
        const id = e.target.dataset && e.target.dataset.id
        if (id) { state.tree.goto(+id); update() }
    })
    const tbody = document.querySelector("#tbl tbody")
    tbody.addEventListener("click", e => {
        const tr = e.target.closest("tr")
        if (tr && tr.dataset.uci) play(tr.dataset.uci)
    })
    tbody.addEventListener("mouseover", e => {
        const tr = e.target.closest("tr")
        if (tr && tr.dataset.uci) showArrow(tr.dataset.uci)
    })
    tbody.addEventListener("mouseleave", () => showArrow(state.cursor))
    document.querySelector("#tbl thead").addEventListener("click", e => {
        const k = e.target.closest("th") && e.target.closest("th").dataset.k
        if (!k) return
        state.sort = {key: k, desc: state.sort.key === k ? !state.sort.desc : k !== "san"}
        render()
    })
    $("paste").addEventListener("keydown", e => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault()
            loadText($("paste").value)
        } else if (e.key === "Escape") {
            $("paste").blur()
        }
    })
    $("fen").addEventListener("click", () => navigator.clipboard && navigator.clipboard.writeText(state.tree.current.fen))
    document.addEventListener("paste", e => {
        if (isTyping(e.target)) return
        const text = e.clipboardData.getData("text")
        if (text) { e.preventDefault(); loadText(text) }
    })
    document.addEventListener("keydown", onKey)
}

const FEN_RE = /^\s*([rnbqkpRNBQKP1-8]+\/){7}[rnbqkpRNBQKP1-8]+\s+[wb]\b/

async function loadText(text) {
    text = text.trim()
    if (!text) return
    try {
        if (FEN_RE.test(text)) {
            let fen = text.split(/\s+/).slice(0, 6)
            const defaults = ["", "w", "-", "-", "0", "1"]
            while (fen.length < 6) fen.push(defaults[fen.length])
            state.tree = new MoveTree(fen.join(" "))
        } else {
            const r = await fetch("/api/pgn", {method: "POST", headers: {"Content-Type": "text/plain"}, body: text})
            const d = await r.json()
            if (!r.ok) throw new Error(d.detail || r.status)
            state.tree = MoveTree.fromPgn(d.root_fen, d.tree)
            state.tree.end()
        }
    } catch (e) {
        status("could not load: " + e.message, true)
        return
    }
    $("paste").value = ""
    $("paste").blur()
    update()
}

function isTyping(el) {
    return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")
}

function command(k) {
    const t = state.tree
    switch (k) {
        case "back": t.back(); break
        case "forward": t.forward(); break
        case "start": t.start(); break
        case "end": t.end(); break
        case "delete": t.deleteCurrent(); break
        case "promote": t.promote(); renderMoves(); return
        case "flip":
            state.orientation = state.orientation === COLOR.white ? COLOR.black : COLOR.white
            board.setOrientation(state.orientation)
            writeUrl()
            fetchMaia()   // "me" is the side at the bottom
            render()
            return
        case "engine": fetchEngine(true); render(); return
        case "all": state.showAll = !state.showAll; render(); return
        default: return
    }
    update()
}

function moveCursor(step) {
    const rs = rows()
    if (!rs.length) return
    let i = rs.findIndex(r => r.uci === state.cursor)
    i = i < 0 ? (step > 0 ? 0 : rs.length - 1) : Math.max(0, Math.min(rs.length - 1, i + step))
    state.cursor = rs[i].uci
    render()
    const tr = document.querySelector("#tbl tr.cursor")
    if (tr) tr.scrollIntoView({block: "nearest"})
}

function onKey(e) {
    if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return
    const map = {
        ArrowLeft: "back", ArrowRight: "forward", Home: "start", End: "end",
        f: "flip", e: "engine", a: "all", Delete: "delete", p: "promote",
    }
    if (map[e.key]) { e.preventDefault(); command(map[e.key]); return }
    if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); moveCursor(1) }
    else if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); moveCursor(-1) }
    else if (e.key === "Enter" && state.cursor) { e.preventDefault(); const u = state.cursor; state.cursor = null; play(u) }
    else if (e.key === "/") { e.preventDefault(); $("paste").focus() }
}

// ---------------------------------------------------------------- start

async function main() {
    readUrl()
    try {
        const r = await fetch("/api/info")
        state.info = await r.json()
        const st = state.info.stats
        $("source").textContent = st.source + (st.year_min ? ` · ${st.year_min}–${st.year_max}` : "")
        $("source").title = `${st.positions} positions`
    } catch (e) {
        status("info: " + e.message, true)
    }
    initControls()
    initBoard()
    window.tnh = {state, play, command}   // for the end-to-end test and the console
    update()
}

main()
