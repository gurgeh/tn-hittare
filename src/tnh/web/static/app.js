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
// Choices of presentation; each is one switch here. Decided by the owner on 2026-11-12:
// always from White, classical games as the default tempo.
const CHOICES = {
    view: "white",          // "white": scores, engine and W/D/L from White; "mover": from the side to move
    defaultTempi: ["classical"],   // tempi when the URL names none: null = all, ["classical"] = classical only
    elo: "colours",         // "colours": W-Elo and B-Elo deltas; "chooser": the chooser's delta and the rating-gap delta
    trend: true,            // the per-year share column
}
const API = 3                // the server's /api/info "api" this page needs

const state = {
    tree: new MoveTree(),
    orientation: COLOR.white,
    filters: {y0: "", y1: "", e0: "", e1: "", tempi: CHOICES.defaultTempi},
    white: 2200, black: 2200, nodes: 200000, sfauto: false,
    showAll: false,
    sort: {key: "n", desc: true},
    cursor: null,     // uci of the highlighted row
    stats: null, model: null, engine: null,
    modelBusy: false,
    info: null,
}
const cache = {stats: new Map(), model: new Map(), engine: new Map()}
const inflight = {stats: null, model: null, engine: null}
const timing = {}

// ---------------------------------------------------------------- URL state

function readUrl() {
    const p = new URLSearchParams(location.hash.slice(1))
    const f = state.filters
    for (const k of ["y0", "y1", "e0", "e1"]) if (p.has(k)) f[k] = p.get(k)
    if (p.has("t")) f.tempi = p.get("t") === "all" ? null : p.get("t") ? p.get("t").split(",") : []
    if (p.has("w")) state.white = +p.get("w")
    if (p.has("b")) state.black = +p.get("b")
    if (p.has("me") && !p.has("w")) {   // links from before 2026-11-12: me/opp, me at the bottom
        const meWhite = p.get("o") !== "b"
        state.white = +(meWhite ? p.get("me") : p.get("opp") || 2200)
        state.black = +(meWhite ? p.get("opp") || 2200 : p.get("me"))
    }
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
    if (JSON.stringify(f.tempi) !== JSON.stringify(CHOICES.defaultTempi)) p.set("t", f.tempi ? f.tempi.join(",") : "all")
    p.set("w", state.white)
    p.set("b", state.black)
    if (state.nodes !== 200000) p.set("nodes", state.nodes)
    if (state.sfauto) p.set("sf", "1")
    if (state.orientation === COLOR.black) p.set("o", "b")
    history.replaceState(null, "", "#" + p.toString().replace(/%2C/g, ","))
}

// ---------------------------------------------------------------- formatting

const pct = (x, d = 0) => x == null ? "" : (100 * x).toFixed(d)
const signed = x => x == null ? "" : Math.round(x) === 0 ? "0" : (x > 0 ? "+" : "−") + Math.abs(Math.round(x))
// A share in whole percent; a share that is not zero but rounds to it shows as "<1".
const pctShare = x => x == null ? "" : x > 0 && x < 0.005 ? "<1" : (100 * x).toFixed(0)
const int = x => x == null ? "" : Math.round(x).toString()
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]))

// The side the numbers are from: the side to move, or White (CHOICES.view).
function flipped() {
    return CHOICES.view === "white" && !state.tree.current.fen.includes(" w ")
}
const povScore = x => x == null ? null : flipped() ? 1 - x : x
const povWdl = wdl => flipped() ? [wdl[2], wdl[1], wdl[0]] : wdl
const VIEW = () => CHOICES.view === "white" ? "White" : "the side to move"

function cpText(sf) {
    if (!sf) return ""
    const sign = flipped() ? -1 : 1
    if (sf.mate) return (sign * sf.mate > 0 ? "M" : "-M") + Math.abs(sf.mate)
    if (sf.cp == null) return "?"
    const v = sign * sf.cp / 100
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
            pieces: {file: "/pieces.svg", tileSize: 40},
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
    if (!r.ok) {
        const e = new Error(`${kind}: ${body.detail || r.status}`)
        e.status = r.status
        throw e
    }
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

const modelsOff = () => state.info && state.info.models && state.info.models.disabled
const oldServer = () => state.info && state.info.api !== API

function fetchModel() {
    state.model = null
    if (modelsOff() || oldServer()) return
    const fen = state.tree.current.fen
    if (new Chess(fen).isGameOver()) return
    const moves = historyMoves()
    const q = filterQuery()
    const key = [fen, moves.slice(-8).join("."), moves.length < 8, state.white, state.black, q].join("|")
    state.modelBusy = true
    const url = `/api/model?fen=${encodeURIComponent(fen)}&moves=${moves.join(",")}&white=${state.white}&black=${state.black}&${q}`
    const hit = cache.model.get(key)
    if (hit) { state.model = hit; state.modelBusy = false; return }
    fetchJson("model", url).then(d => {
        cache.model.set(key, d)
        state.modelBusy = false
        if (state.tree.current.fen !== fen || filterQuery() !== q) return
        state.model = d
        render()
    }).catch(e => {
        state.modelBusy = false
        // A newer position superseded this request on the server: not an error.
        if (e.name === "AbortError" || e.status === 409) return
        status(e.message, true)
        render()
    })
}

function fetchEngine(force) {
    if (modelsOff() || oldServer()) return
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

// The columns. A column whose figure comes from the models shows the database figure,
// dimmed, until the models answer (or for good when they are off).
// The rule each shrunk column uses, as the server states it (a fitted rule from
// shrink_rules.json, or the fallback n/(n+k)).
function ruleText(q) {
    const r = state.info && state.info.shrink && state.info.shrink[q]
    if (!r || typeof r !== "object") return ""
    return ` Rule (${r.fallback ? "fallback, not fitted" : "fitted " + r.fitted + " on held-out games"}): ${r.text}; n = the move's games, share = n / the position's games.`
}

function eloCols() {
    if (CHOICES.elo === "chooser") return [
        {k: "elo1", h: "Δ Elo", t: () => "The chooser's rating against the position, predicted for the next game with this move: the mean rating of the side to move in its games minus the mean over all moves from here, shrunk toward the rule's prior (n = games with both ratings)." + ruleText("chooser")},
        {k: "elo2", h: "Δ gap", t: () => "The rating gap (chooser minus opponent) in the games with this move minus the same gap over all moves from here, shrunk the same way: above zero, the move is chosen against relatively weaker opponents." + ruleText("gap")},
    ]
    return [
        {k: "elo1", h: "W-Elo", t: () => "White's rating, predicted for the next game with this move, against White's mean over all moves from here: the move's own difference shrunk toward the rule's prior (n = games with both ratings)." + ruleText("w_elo")},
        {k: "elo2", h: "B-Elo", t: () => "The same for Black." + ruleText("b_elo")},
    ]
}

function cols() {
    const sh = state.info && state.info.shrink || {}
    return [
        {k: "san", h: "Move", num: false, w: 62, t: "Move. Italic: not played in the database under these filters"},
        {k: "n", h: "Games", w: 62, t: "Games that played the move from this position, under the filters"},
        {k: "p_mix", h: "Prob", w: 46, t: `Probability of the move: (games + κ·model 1) / (all games + κ), κ = ${sh.kappa} (not fitted yet: it needs model 1 on the sample). The database share alone in the cell's tooltip`},
        {k: "p_model", h: "Model", w: 50, t: "Model 1's probability for the side to move at the header's ratings"},
        {k: "score", h: "Score", w: 50, t: () => "White's expected score, %: the move's games shrunk toward model 2 for the position after it (header ratings); until model 2 answers, toward the position's own mean." + ruleText("score")},
        {k: "draw", h: "Draw", w: 46, t: () => "Draw probability, %, shrunk the same way." + ruleText("draw")},
        ...eloCols().map(c => ({...c, w: 58})),
        {k: "last_year", h: "Last", w: 44, t: "Last year the move was played"},
        {k: "reach", h: "Reach", w: 62, t: "Games reaching the position after the move by any move order; unfiltered, all tempi"},
        ...(CHOICES.trend ? [{k: "trend", h: "Trend", num: false, w: 82, t: "Share per year. ▲/▼: recent share differs from the earlier share beyond chance"}] : []),
        {k: "sf", h: "SF", w: 52, t: () => `Stockfish after the move, pawns from ${VIEW()}'s view`},
    ]
}

function rows() {
    const fen = state.tree.current.fen
    const chess = new Chess(fen)
    const legal = new Map(chess.moves({verbose: true}).map(m => [m.from + m.to + (m.promotion || ""), m.san]))
    const s = state.stats, md = state.model && state.model.moves, en = state.engine && state.engine.moves
    const pList = (state.info && state.info.p_list) || 0.01
    const byUci = new Map()
    const add = uci => {
        if (!byUci.has(uci)) byUci.set(uci, {uci, san: legal.get(uci) || uci, db: null, m: null})
        return byUci.get(uci)
    }
    if (s) for (const m of s.moves) add(m.uci).db = m
    if (md) for (const [u, v] of Object.entries(md)) if (v.p_model >= pList || v.n > 0 || state.showAll) add(u).m = v
    if (md) for (const r of byUci.values()) r.m = md[r.uci] || r.m
    if (state.showAll) for (const u of legal.keys()) add(u)
    let bestSf = null
    if (en) {
        for (const [u, v] of Object.entries(en)) if (v.value != null && (!bestSf || v.value > en[bestSf].value)) bestSf = u
        if (bestSf) add(bestSf)
    }
    const total = s && s.status === "ok" ? s.total.played : 0
    const flip = !fen.includes(" w ") && CHOICES.view === "mover"
    const out = []
    for (const r of byUci.values()) {
        const d = r.db || {}, m = r.m
        r.n = total ? d.n || 0 : null          // 0 is data only where the position has games
        r.share = total ? (d.n || 0) / total : null
        for (const k of ["last_year", "reach", "n_rated"]) r[k] = d[k] ?? null
        if (d.white_wins != null) {
            r.dbScore = d.n ? (d.white_wins + 0.5 * d.draws) / d.n : null
            r.dbDraw = d.n ? d.draws / d.n : null
        } else {   // a server from before api 3 sends the mover's score only
            r.dbScore = d.score == null ? null : fen.includes(" w ") ? d.score : 1 - d.score
            r.dbDraw = d.draw ?? null
        }
        r.p_mix = m ? m.p_mix : null
        r.p_model = m ? m.p_model : null
        // Before model 2: the database shrunk toward the position (servers from api 3 on).
        r.preScore = d.score_shrunk ?? r.dbScore
        r.preDraw = d.draw_shrunk ?? r.dbDraw
        r.score = m ? m.score_mix : r.preScore
        r.draw = m ? m.draw_mix : r.preDraw
        if (flip && r.score != null) r.score = 1 - r.score
        r.elo1 = CHOICES.elo === "chooser" ? d.mover_elo_delta ?? null : d.w_elo_delta ?? null
        r.elo2 = CHOICES.elo === "chooser" ? d.gap_delta ?? null : d.b_elo_delta ?? null
        r.sfv = en ? en[r.uci] : null
        r.sf = r.sfv ? povScore(r.sfv.value) : null
        r.best = r.uci === bestSf
        r.trend = d.trend ? d.trend.recent - d.trend.early : null
        out.push(r)
    }
    const {key, desc} = state.sort
    const val = r => key === "san" ? r.san : key === "n" ? r.n ?? -1 : r[key]
    const tie = (a, b) => (b.p_model || 0) - (a.p_model || 0)
    out.sort((a, b) => {
        const va = val(a), vb = val(b)
        if (va == null && vb == null) return tie(a, b)
        if (va == null) return 1
        if (vb == null) return -1
        const c = typeof va === "string" ? va.localeCompare(vb) : va - vb
        return (desc ? -c : c) || tie(a, b)
    })
    return out
}

function renderHead() {
    const cs = cols()
    $("cols").innerHTML = cs.map(c => `<col style="width:${c.w}px">`).join("")
    const tr = document.querySelector("#tbl thead tr")
    tr.innerHTML = cs.map(c => {
        const sorted = state.sort.key === c.k ? (state.sort.desc ? " ▾" : " ▴") : ""
        let t = typeof c.t === "function" ? c.t() : c.t
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

function pending() {
    return modelsOff() ? "models off" : oldServer() ? "the server is older than this page: restart it"
        : state.modelBusy ? "models computing: the database alone for now" : "the database alone"
}

function probCell(r) {
    const db = r.share == null ? "no games in the position" : `database ${pctShare(r.share)}% (${r.n ?? 0} of ${state.stats.total.played})`
    if (r.p_mix != null) return `<td class="num" title="${esc(db + `; model ${pct(r.p_model, 1)}%`)}">${pctShare(r.p_mix)}</td>`
    return `<td class="num pending" title="${esc(db + "; " + pending())}">${pctShare(r.share)}</td>`
}

function mixCell(r, kind) {
    const m = r.m
    const v = kind === "score" ? r.score : r.draw
    const dbv = kind === "score" ? r.dbScore : r.dbDraw
    const pre = kind === "score" ? r.preScore : r.preDraw
    const db = dbv == null ? "no games" : `database ${pct(dbv, 1)}% of ${r.n}`
    if (m && v != null) {
        const mv = kind === "score" ? m.score_model : m.draw_model
        return `<td class="num" title="${esc(`${db}; model 2 ${pct(mv, 1)}%`)}">${pct(v)}</td>`
    }
    const shrunk = pre != null && pre !== dbv ? `; shrunk toward the position ${pct(pre, 1)}%` : ""
    // With the models off this is the final figure; otherwise it waits for model 2.
    const cls = modelsOff() && shrunk ? "num" : "num pending"
    return `<td class="${cls}" title="${esc(db + shrunk + "; " + pending())}">${pct(pre)}</td>`
}

function eloCell(r, which) {
    const d = r.db
    const v = which === 1 ? r.elo1 : r.elo2
    if (!d || v == null) return `<td class="num"></td>`
    const t = state.stats.total
    let tip
    if (CHOICES.elo === "chooser") {
        const chooser = state.stats.white_to_move ? "White" : "Black"
        tip = which === 1
            ? `${chooser} (chooser) here ${int(state.stats.white_to_move ? d.white_elo : d.black_elo)}, position ${int(state.stats.white_to_move ? t.white_elo : t.black_elo)}`
            : `gap chooser − opponent: this move ${signed(d.gap)}, position ${signed(t.gap)}`
    } else {
        tip = which === 1 ? `White here ${int(d.white_elo)}, position ${int(t.white_elo)}` : `Black here ${int(d.black_elo)}, position ${int(t.black_elo)}`
    }
    const raw = CHOICES.elo === "chooser" ? (which === 1 ? d.mover_elo_raw : d.gap_raw) : (which === 1 ? d.w_elo_raw : d.b_elo_raw)
    if (raw != null) tip += `; unshrunk ${signed(raw)}`
    return `<td class="num" title="${esc(`${tip}; ${d.n_rated} games with both ratings`)}">${signed(v)}</td>`
}

function totalsLine(s) {
    const t = s.total, table = state.info ? state.info.stats.table : "the"
    if (s.status === "none") return `<b>No games.</b> Fewer than two games in the ${table} table reach this position.`
    if (s.status === "ended") return `<b>${t.reach}</b> games reach this position; none played on from it (they ended here or at ply 80).`
    if (s.status === "filtered") return `<b>No games under these filters.</b> ${t.reach} games reach this position without them.`
    const pos = state.model && state.model.position
    const dbScore = t.white_score ?? (t.white_wins + 0.5 * t.draws) / t.played   // ?? : servers before api 3
    const dbDraw = t.draw ?? t.draws / t.played
    const score = pos && pos.score_mix != null ? pos.score_mix : dbScore
    const draw = pos && pos.draw_mix != null ? pos.draw_mix : dbDraw
    const cls = pos && pos.score_mix != null ? "" : "pending"
    const tip = `database ${pct(dbScore, 1)}% / draws ${pct(dbDraw, 1)}% of ${t.played}` +
        (pos && pos.score_model != null ? `; model 2 ${pct(pos.score_model, 1)}% / ${pct(pos.draw_model, 1)}%` : `; ${pending()}`)
    const elo = t.white_elo === undefined ? "" : CHOICES.elo === "chooser"
        ? `gap ${s.white_to_move ? "White" : "Black"} − opponent ${signed(t.gap)}`
        : `White ${int(t.white_elo)} · Black ${int(t.black_elo)}`
    return `<b title="Games that played a move from here, under the filters (a game that comes back and plays another move counts for both)">${t.played}</b> games` +
        ` <span class="dim" title="Games that reached this position: all years, bands and tempi, each game once">(${t.reach} reached)</span>` +
        ` · <span class="${cls}" title="${esc(tip)}">White ${pct(score, 1)}% · draw ${pct(draw)}%</span>` +
        (elo ? ` · <span title="Mean ratings over the moves' games with both ratings (${t.n_rated})">${elo}</span>` : "")
}

function render() {
    const s = state.stats
    if (s) {
        $("totals").innerHTML = totalsLine(s) + ` <span class="dim mono" title="Zobrist hash">${s.hash}</span>`
        const mo = s.move_orders
        if (mo && mo.length > 1) {
            $("orders").innerHTML = `Transposition: reached by ${mo.length} different last moves · ` +
                mo.slice(0, 6).map(o => `<span title="${esc(o.parent_fen || "")}">${esc(o.san)} ${o.n}</span>`).join(" · ")
        } else {
            $("orders").textContent = ""
        }
    } else {
        $("totals").innerHTML = '<span class="dim">loading…</span>'
    }
    renderHead()
    const rs = rows()
    if (state.cursor && !rs.find(r => r.uci === state.cursor)) state.cursor = null
    const body = document.querySelector("#tbl tbody")
    body.innerHTML = rs.map(r => {
        const cls = [r.db ? "" : "nodb", r.uci === state.cursor ? "cursor" : ""].join(" ")
        const sfTitle = r.sfv ? `expected score ${pct(r.sf, 1)}%` : ""
        return `<tr data-uci="${r.uci}" class="${cls}">` +
            `<td class="san">${esc(r.san)}</td>` +
            `<td class="num${r.db ? "" : " dim"}">${r.n ?? ""}</td>` +
            probCell(r) +
            `<td class="num model">${r.p_model == null ? "" : pctShare(r.p_model)}</td>` +
            mixCell(r, "score") + mixCell(r, "draw") +
            eloCell(r, 1) + eloCell(r, 2) +
            `<td class="num">${r.last_year ?? ""}</td>` +
            `<td class="num dim">${r.reach ?? ""}</td>` +
            (CHOICES.trend ? `<td>${trendCell(r, s)}</td>` : "") +
            `<td class="num ${r.best ? "best" : ""}" title="${sfTitle}">${cpText(r.sfv)}</td>` +
            `</tr>`
    }).join("")
    if (!rs.length) body.innerHTML = `<tr><td colspan="${cols().length}" class="dim">${s ? "no moves to show" : ""}</td></tr>`
    showArrow(state.cursor)
    renderModels()
    renderStatus()
}

function renderModels() {
    const m = state.model, e = state.engine
    let h
    if (oldServer()) {
        h = '<span class="warn">The server is older than this page: restart it (the database columns still work).</span>'
    } else if (modelsOff()) {
        h = '<span class="dim">models off (--no-models): Prob is the database share; Score, Draw and the rating columns are the database shrunk by the fitted rules</span>'
    } else if (m) {
        const c = m.cost
        h = `<span title="Model 1: moves for the side to move. Model 2: win, draw, loss of each position. Both at White ${m.white}, Black ${m.black} (FIDE)">` +
            `<b>${esc(m.move_model)}</b> + <b>${esc(m.value_model)}</b> · W ${m.white} B ${m.black}</span>` +
            ` <span class="dim" title="Model 2 positions: asked, computed now (the rest were cached)">· ${c.value_queries} positions, ${c.value_computed} new, ${(c.total_ms / 1000).toFixed(1)} s</span>` +
            (state.tree.fromStart ? "" : ' · <span class="warn" title="Position set from a FEN: the models see no earlier positions">no move history</span>')
    } else if (state.modelBusy) {
        h = '<span class="dim">models: computing…</span>'
    } else {
        h = '<span class="dim">models: —</span>'
    }
    if (e) {
        const ew = e.wdl ? povWdl(e.wdl) : null
        h += ` · <b>${esc(e.model)}</b>` + (ew ? ` W ${pct(ew[0])} D ${pct(ew[1])} L ${pct(ew[2])}` : "")
    } else if (state.engineBusy) {
        h += ' · <span class="dim">engine: running…</span>'
    }
    $("models").innerHTML = h
}

function renderStatus() {
    const parts = []
    if (state.stats) parts.push(`stats ${state.stats.ms.toFixed(0)} ms server`)
    for (const k of ["stats", "model", "engine"]) if (timing[k] != null) parts.push(`${k} ${timing[k].toFixed(0)} ms`)
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
    fetchModel()
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
    fetchModel()
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
    $("white").value = state.white
    $("black").value = state.black
    for (const id of ["white", "black"]) $(id).addEventListener("change", () => {
        state[id] = +$(id).value
        writeUrl()
        fetchModel()
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
        $("source").textContent = `${st.table} table · ${st.games.toLocaleString("en")} games` +
            (st.year_min ? ` · ${st.year_min}–${st.year_max}` : "")
        $("source").title = `EdgeDB(name="${st.table}"): ${st.why}` + (st.built ? `; files from ${st.built}` : "")
    } catch (e) {
        status("info: " + e.message, true)
    }
    initControls()
    initBoard()
    window.tnh = {state, play, command, render, CHOICES}   // for the end-to-end test and the console
    update()
}

main()
