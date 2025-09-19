// wrapper.js
import readline from 'node:readline';
import { initialState, applyPlacement, getSnapshot } from '../tetris_game/src/state.js';

let gameState = null;

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,   // handle \r\n and \n
});

function send(obj) {
  // One line per response; Python does readline()
  process.stdout.write(JSON.stringify(obj) + '\n');
}

function snapshotResponse(state) {
  const s = getSnapshot(state);
  return {
    board: s.board,
    currentPiece: s.cur,
    nextPiece: s.queue?.[0] ?? null,
    score: s.score,
    lines: s.lines,
    done: !!s.toppedOut,
  };
}

rl.on('line', (line) => {
  if (!line.trim()) return;

  let cmd;
  try {
    cmd = JSON.parse(line);
  } catch (e) {
    // Always respond so Python never blocks
    return send({ error: 'BAD_JSON', message: e.message });
  }

  try {
    switch (cmd.action) {
      case 'reset': {
        gameState = initialState({ seed: cmd.seed });
        return send({ ok: true, ...snapshotResponse(gameState) });
      }
      case 'step': {
        if (!gameState) return send({ error: 'NO_STATE', message: 'Call reset first' });
        try {
          applyPlacement(gameState, cmd.placement);
        } catch (e) {
          // Invalid move or game exception — still reply
          return send({ error: 'STEP_FAILED', message: e.message });
        }
        return send({ ok: true, ...snapshotResponse(gameState) });
      }
      default:
        return send({ error: 'UNKNOWN_ACTION', action: cmd.action });
    }
  } catch (e) {
    // Belt-and-suspenders: never leave Python waiting
    return send({ error: 'UNHANDLED', message: e.message });
  }
});

if (command.action === 'step') {
  try {
    // Validate inputs first
    if (!gameState || gameState.toppedOut) {
      process.stdout.write(JSON.stringify({
        board: gameState?.board || [],
        currentPiece: null,
        nextPiece: null,
        score: gameState?.score || 0,
        lines: gameState?.lines || 0,
        done: true
      }) + '\n');
      return;
    }

    applyPlacement(gameState, command.placement);
    const snapshot = getSnapshot(gameState);
    // ... rest of normal response
  } catch (error) {
    // If placement fails, just end the game
    process.stdout.write(JSON.stringify({
      board: gameState.board,
      currentPiece: null,
      nextPiece: null,
      score: gameState.score,
      lines: gameState.lines,
      done: true  // End game on any error
    }) + '\n');
  }
}

// Clean exit if Python closes stdin
rl.on('close', () => process.exit(0));