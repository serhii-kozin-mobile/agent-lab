"""Review experiment (2026-10-26), NOT part of the published numbers.
The chip-conservation guardrail anchors on the first acted read of a hand without checking it, so a
misread there poisons the rest of the hand. Variant: the first read of a hand only becomes the anchor
if two frames in a row agree on the chip total (and hole cards); otherwise re-capture (max 3 frames),
then fall back. Same seeds, same hands, same policy. Usage: python review_anchor_fix.py 3000"""
from __future__ import annotations
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from agent import VisionAgent
from game import play_hand
from run_matches import BASE_SEED, THRESHOLDS

class AnchoredVisionAgent(VisionAgent):
    def _problems(self, p, conf):
        probs = super()._problems(p, conf)
        key = (self._hand, len(self.log))           # one key per decision
        if self._mem.get("total") is None:
            total = p.pot + p.hero_stack + p.villain_stack + p.hero_bet + p.villain_bet
            prev = self._pending.get(key) if hasattr(self, "_pending") else None
            if not hasattr(self, "_pending"): self._pending = {}
            self._pending[key] = (total, tuple(p.hero_cards))
            if prev is None or prev != (total, tuple(p.hero_cards)):
                probs.append("anchor_unconfirmed")
        return probs

def main(n, cond="realistic_mix"):
    a = AnchoredVisionAgent(cond, mode="hardened", guard=True, base_seed=BASE_SEED, thresholds=THRESHOLDS)
    rows = [dict(hand=i, result_bb=play_hand(i, BASE_SEED, a)[0]) for i in range(n)]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "review_anchor_fix")
    h = pd.DataFrame(rows); d = pd.DataFrame(a.log)
    h.to_csv(os.path.join(out, f"hands_{cond}.csv"), index=False); d.to_csv(os.path.join(out, f"decisions_{cond}.csv.gz"), index=False)
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    allh = pd.read_csv(os.path.join(R, "hands.csv")); gt = allh[allh.run == "gt"].set_index("hand").result_bb.loc[h.hand]
    old = allh[allh.run == f"hardened_guarded__{cond}"].set_index("hand").result_bb.loc[h.hand]
    from game import hand_setup
    style = pd.Series({i: hand_setup(i, BASE_SEED)["opp_style"] for i in h.hand})
    def ci(x): x = np.asarray(x, float); return [float(x.mean() * 100), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)) * 100)]
    new = h.set_index("hand").result_bb
    res = dict(hands=n, condition=cond,
               fallback_rate=float(d.fallback.mean()), wrong_state_rate=float(1 - d.state_correct.mean()),
               acted_on_wrong_state=float((~d[~d.fallback].state_correct).sum() / len(d)),
               action_mismatch=float((d.gt_action != d.vis_action).mean()), frames_per_decision=float(d.frames.mean()),
               gap_all=ci(new - gt), gap_station=ci((new - gt)[style == "station"]), gap_tag=ci((new - gt)[style == "tag"]),
               original_gap_station=ci((old - gt)[style == "station"]), original_gap_all=ci(old - gt))
    json.dump(res, open(os.path.join(out, f"summary_{cond}.json"), "w"), indent=1); print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000, sys.argv[2] if len(sys.argv) > 2 else "realistic_mix")
