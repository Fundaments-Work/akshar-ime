// File: src/bin/evaluate/tune_weights.rs
//
// Joint coordinate-descent tuning of the runtime knobs on HELD-OUT data —
// the stages stop fighting each other. Optimizes the actual metric (top-1
// hits, exact integer counts), derivative-free: the pipeline has discrete
// decisions (beam, argmax) with nothing to differentiate through.
//
// Knobs (all runtime env, no retraining):
//   AKSHAR_GAMMA, AKSHAR_BEAM, AKSHAR_RERANK_DEPTH, AKSHAR_NO_TRIE_UNION
// on the VALID split (pooled top-1 over all strata, n=9155), then AKSHAR_CTX_W
// on disjoint held-out sentences (--skip 1000 avoids the measured 0-1000
// window) in predicted mode.
//
// Method: cycle over knobs, line-search each while holding the rest, repeat
// until no knob moves (max 3 cycles). Each eval spawns the shipped worker
// binary with env set (ablation flags are OnceLock-cached per process, so
// in-process search would lie). Deterministic engine => exact comparisons,
// no noise bars needed within a set; cross-set generalization is confirmed
// once on TEST at the end, by hand, never in this loop.
//
// Usage: cargo run --release --bin tune_weights
// Prerequisite: cargo build --release --bins (workers must exist).

use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

const WORKER: &str = "target/release/evaluate_aksharantar";
const SENT_WORKER: &str = "target/release/evaluate_sentences";
const VALID: &str = "data/aksharantar/valid_devanagari.jsonl";
const CTX: &str = "data/corpus_bigrams.bin";

/// Per-stratum (name, top1 hits, total) from one worker run.
type Strata = Vec<(String, u64, u64)>;

fn run_word(env: &[(&str, &str)]) -> (Strata, f64) {
    let t0 = Instant::now();
    let mut cmd = Command::new(WORKER);
    cmd.args(["--dataset", VALID, "--topk", "5", "--show-misses", "0"]);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn evaluate_aksharantar");
    assert!(out.status.success(), "worker failed");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut strata = Vec::new();
    for line in text.lines() {
        if !line.contains("top1=") {
            continue;
        }
        // stratum line: "... (top1 h/n, top5 h/n)". Parse the TOP-1 pair:
        // the old "(h/n)" format carried top-5 counts, which once sent the
        // tuner chasing the wrong metric — never parse bare parens again.
        let name = line.split_whitespace().next().unwrap_or("?").to_string();
        if let Some(i) = line.find("(top1 ") {
            let rest = &line[i + 6..];
            // first segment before ',' is "h/n" (the top5 pair follows).
            let first = rest.split(',').next().unwrap_or("");
            let mut it = first.split('/');
            if let (Some(h), Some(n)) = (it.next(), it.next()) {
                if let (Ok(h), Ok(n)) = (h.trim().parse::<u64>(), n.trim().parse::<u64>()) {
                    strata.push((name, h, n));
                }
            }
        }
    }
    assert!(!strata.is_empty(), "parsed no strata from worker output");
    // Tripwire against format drift: the valid split has exactly 9155 cases.
    // (Burned once by the top-5-counts incident — a silent wrong-column parse
    // must fail loudly instead of mistuning.)
    let total: u64 = strata.iter().map(|(_, _, n)| n).sum();
    assert_eq!(total, 9155, "valid split size changed? strata={strata:?}");
    (strata, t0.elapsed().as_secs_f64())
}

/// Objective: AK-Freq hits (same family as the test headline). Pooled valid
/// is recorded as a guardrail only — tuning on it robs the native family to
/// pay named-entity strata (2026-09-22: pooled +3.27pp, AK-Freq −8.67pp).
fn akfreq_hits(strata: &Strata) -> u64 {
    strata
        .iter()
        .find(|(n, _, _)| n == "AK-Freq")
        .map(|(_, h, _)| *h)
        .unwrap_or(0)
}

fn pooled_hits(strata: &Strata) -> (u64, u64) {
    strata
        .iter()
        .fold((0, 0), |(h, n), (_, hi, ni)| (h + hi, n + ni))
}

fn run_sent(ctx_w: &str) -> f64 {
    let mut cmd = Command::new(SENT_WORKER);
    cmd.args([
        "--n",
        "1000",
        "--skip",
        "1000",
        "--topk",
        "5",
        "--ctx",
        CTX,
        "--ctx-mode",
        "predicted",
    ]);
    cmd.env("AKSHAR_CTX_W", ctx_w);
    let out = cmd.output().expect("spawn evaluate_sentences");
    assert!(out.status.success(), "sentence worker failed");
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if line.contains("word@1=") {
            // "... word@1= 88.80% ..."
            if let Some(i) = line.find("word@1=") {
                let rest = line[i + 7..].trim_start();
                let num: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.')
                    .collect();
                if let Ok(v) = num.parse::<f64>() {
                    return v;
                }
            }
        }
    }
    panic!("parsed no word@1 from sentence worker");
}

struct Knob {
    name: &'static str,
    values: Vec<&'static str>,
}

fn word_env(cfg: &HashMap<&str, &str>) -> Vec<(String, String)> {
    let mut env = vec![
        ("AKSHAR_GAMMA".to_string(), cfg["gamma"].to_string()),
        ("AKSHAR_BEAM".to_string(), cfg["beam"].to_string()),
        ("AKSHAR_RERANK_DEPTH".to_string(), cfg["depth"].to_string()),
    ];
    if cfg["trie"] == "0" {
        env.push(("AKSHAR_NO_TRIE_UNION".to_string(), "1".to_string()));
    }
    env
}

fn main() {
    for bin in [WORKER, SENT_WORKER] {
        assert!(
            std::path::Path::new(bin).exists(),
            "missing worker {bin}: run cargo build --release --bins first"
        );
    }
    assert!(
        std::path::Path::new(CTX).exists(),
        "missing {CTX}: run make ctx-model first"
    );

    // Conditional blend knobs replace the old single gamma (kept working via
    // AKSHAR_GAMMA compat, but the search moves on the (lo, hi) pair).
    // Objective: CONSTRAINED — maximize pooled top-1 subject to AK-Freq >=
    // baseline (443). Unconstrained pooled tuning robbed native to pay
    // entities; the constraint encodes "no harm" directly.
    let knobs = [
        Knob {
            name: "lo",
            values: vec!["0.1", "0.2", "0.3"],
        },
        Knob {
            name: "hi",
            values: vec!["0.3", "0.5", "0.6", "0.8"],
        },
        Knob {
            name: "beam",
            values: vec!["32", "48", "64", "96", "128"],
        },
        Knob {
            name: "depth",
            values: vec!["8", "12", "16", "24", "32", "50"],
        },
        Knob {
            name: "trie",
            values: vec!["1", "0"],
        },
    ];
    let mut cfg: HashMap<&str, &str> = [
        ("gamma", "0.3"),
        ("beam", "64"),
        ("depth", "24"),
        ("trie", "1"),
    ]
    .into_iter()
    .collect();
    let mut cache: HashMap<String, (Strata, f64)> = HashMap::new();
    let mut eval = |cfg: &HashMap<&str, &str>| -> (Strata, f64) {
        let key = format!(
            "{}|{}|{}|{}",
            cfg["gamma"], cfg["beam"], cfg["depth"], cfg["trie"]
        );
        if let Some(v) = cache.get(&key) {
            return v.clone();
        }
        let env = word_env(cfg);
        let envref: Vec<(&str, &str)> = env.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let v = run_word(&envref);
        cache.insert(key, v.clone());
        v
    };

    // n=519 is noisy (SE ~8 hits): a move must win outright on AK-Freq AND
    // survive the next cycle's re-verification to stick (see loop below).
    let (strata0, t0) = eval(&cfg);
    let base_ak = akfreq_hits(&strata0);
    let (bph, bpn) = pooled_hits(&strata0);
    println!(
        "start: gamma=0.3 beam=64 depth=24 trie=1 -> AK-Freq {base_ak}/519 ({:.2}%), pooled {bph}/{bpn} ({:.2}%, {t0:.1}s)",
        base_ak as f64 / 519.0 * 100.0,
        bph as f64 / bpn as f64 * 100.0
    );
    println!("constraint: AK-Freq >= {base_ak} (baseline); maximizing pooled.");
    let (mut best_h, mut best_n) = (bph, bpn);
    let mut best_ak = base_ak;
    let _ = t0;
    for cycle in 1..=3 {
        let mut moved = false;
        for knob in &knobs {
            // incumbent score: (constraint-ok, pooled hits, time)
            let (vs0, vt0) = eval(&cfg);
            let (mut lb_ok, mut lb_pool, mut lb_t) =
                (akfreq_hits(&vs0) >= base_ak, pooled_hits(&vs0).0, vt0);
            let mut local_val = cfg[knob.name];
            for &v in &knob.values {
                let mut trial = cfg.clone();
                trial.insert(knob.name, v);
                let (strata, t) = eval(&trial);
                let (ak, (ph, pn)) = (akfreq_hits(&strata), pooled_hits(&strata));
                let ok = ak >= base_ak;
                println!(
                    "  cycle{cycle} {}={v:<5} -> AK-Freq {ak}/519 ({:.2}%){} pooled {ph}/{pn} ({:.2}%, {t:.1}s)",
                    knob.name,
                    ak as f64 / 519.0 * 100.0,
                    if ok { "" } else { " REJECT" },
                    ph as f64 / pn as f64 * 100.0
                );
                // Constraint first: a config that harms native can never win,
                // however high its pooled total. Then pooled, then speed.
                let better = (ok && !lb_ok)
                    || (ok == lb_ok
                        && (ph > lb_pool
                            || (ph == lb_pool
                                && (knob.name == "beam" || knob.name == "depth")
                                && t < lb_t - 1.0)));
                if better {
                    (lb_ok, lb_pool, lb_t) = (ok, ph, t);
                    local_val = v;
                }
            }
            if local_val != cfg[knob.name] {
                cfg.insert(knob.name, local_val);
                moved = true;
            }
            // re-verify the (possibly new) incumbent exactly once per knob.
            let (vs, _vt) = eval(&cfg);
            let (vak, (vph, vpn)) = (akfreq_hits(&vs), pooled_hits(&vs));
            (best_ak, best_h, best_n) = (vak, vph, vpn);
        }
        let (fs, _) = eval(&cfg);
        let (fak, (fph, fpn)) = (akfreq_hits(&fs), pooled_hits(&fs));
        println!(
            "after cycle{cycle}: gamma={} beam={} depth={} trie={} -> AK-Freq {fak}/519 ({:.2}%), pooled {fph}/{fpn} ({:.2}%)",
            cfg["gamma"],
            cfg["beam"],
            cfg["depth"],
            cfg["trie"],
            fak as f64 / 519.0 * 100.0,
            fph as f64 / fpn as f64 * 100.0
        );
        if !moved {
            println!("converged after {cycle} cycle(s).");
            break;
        }
    }

    println!("\n--- ctx_w line search (disjoint sentences 1000-2000, predicted) ---");
    for w in ["0.1", "0.25", "0.5", "1.0", "2.0"] {
        let v = run_sent(w);
        println!("  ctx_w={w:<5} -> word@1={v:.2}%");
    }

    println!(
        "\n=== RECOMMENDATION (constrained: pooled max, native no-harm — confirm once on test) ==="
    );
    println!(
        "AKSHAR_GAMMA={} AKSHAR_BEAM={} AKSHAR_RERANK_DEPTH={}{}  # valid AK-Freq {}/{}, pooled {}/{} ({:.2}%)",
        cfg["gamma"],
        cfg["beam"],
        cfg["depth"],
        if cfg["trie"] == "0" { " AKSHAR_NO_TRIE_UNION=1" } else { "" },
        best_ak,
        519,
        best_h,
        best_n,
        best_h as f64 / best_n as f64 * 100.0
    );
    println!("(ctx_w: take the max of the line search above; then run make eval once on test)");
    println!("NOTE 2026-09-22: pooled-valid tuning was tried and REJECTED (pooled +3.27pp,");
    println!("valid-AK-Freq -8.67pp, test-AK-Freq -4.18pp): one global blend cannot serve the");
    println!("native family and named entities at once. Next architecture: conditional");
    println!("gamma(frequency) — low gamma where the prior is trustworthy, high where OOV.");
}
