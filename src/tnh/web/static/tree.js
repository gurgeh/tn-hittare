// The move tree kept in the browser: a root position and variations under it.
// Pure logic on top of chess.js; no DOM.

import {Chess} from "./vendor/chess.js/chess.js"

export const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

let nextId = 1

function node(parent, uci, san, fen) {
    return {id: nextId++, parent, uci, san, fen, children: []}
}

export class MoveTree {
    constructor(rootFen = START_FEN) {
        new Chess(rootFen)   // throws on a bad FEN
        this.root = node(null, null, null, rootFen)
        this.current = this.root
    }

    get fromStart() {
        return this.root.fen === START_FEN
    }

    // Plays `move` ({from, to, promotion} or a UCI or SAN string) from the current node.
    // Follows an existing child with the same move, else adds a new variation.
    // Returns the new current node, or null if the move is illegal.
    play(move) {
        const chess = new Chess(this.current.fen)
        let m
        try {
            if (typeof move === "string" && /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move)) {
                m = chess.move({from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4]})
            } else {
                m = chess.move(move)
            }
        } catch (e) {
            return null
        }
        const uci = m.from + m.to + (m.promotion || "")
        let child = this.current.children.find(c => c.uci === uci)
        if (!child) {
            child = node(this.current, uci, m.san, chess.fen())
            this.current.children.push(child)
        }
        this.current = child
        return child
    }

    back() {
        if (this.current.parent) this.current = this.current.parent
        return this.current
    }

    forward() {
        if (this.current.children.length) this.current = this.current.children[0]
        return this.current
    }

    start() {
        this.current = this.root
        return this.current
    }

    end() {
        while (this.current.children.length) this.current = this.current.children[0]
        return this.current
    }

    goto(id) {
        const n = this.find(id)
        if (n) this.current = n
        return this.current
    }

    find(id, n = this.root) {
        if (n.id === id) return n
        for (const c of n.children) {
            const r = this.find(id, c)
            if (r) return r
        }
        return null
    }

    // Removes the current node and its subtree; the parent becomes current.
    deleteCurrent() {
        const n = this.current
        if (!n.parent) return n
        n.parent.children = n.parent.children.filter(c => c !== n)
        this.current = n.parent
        return this.current
    }

    // Makes the current node's line the main line at every branch above it.
    promote() {
        for (let n = this.current; n.parent; n = n.parent) {
            const sib = n.parent.children
            sib.splice(sib.indexOf(n), 1)
            sib.unshift(n)
        }
    }

    // UCI moves from the root to the current node.
    path(n = this.current) {
        const out = []
        for (; n.parent; n = n.parent) out.push(n.uci)
        return out.reverse()
    }

    // Adds a tree from the server's PGN parse ({uci, san, children}) under the root.
    static fromPgn(rootFen, tree) {
        const t = new MoveTree(rootFen)
        const add = (parent, items) => {
            for (const it of items) {
                t.current = parent
                const c = t.play(it.uci)
                if (!c) throw new Error("illegal move in PGN: " + it.san)
                if (it.comment) c.comment = it.comment
                add(c, it.children)
            }
        }
        add(t.root, tree)
        t.current = t.root
        return t
    }

    // Fullmove number and side of the move that leads to `n`.
    static moveNumber(n) {
        const parts = n.parent.fen.split(" ")
        return {number: parseInt(parts[5], 10), white: parts[1] === "w"}
    }
}
