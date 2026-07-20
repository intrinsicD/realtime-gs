# Iteration 2 topology-repair preregistration — independent review (initial FAIL)

Reviewed before implementation or fresh-root access on 2026-07-17.

Verdict: **FAIL PENDING PAPER REPAIR**. The reviewer found five ambiguities that could make a
later positive topology attribution invalid:

1. The proposed method had 600 updates while the no-topology control stopped at 400.
2. An unconstrained own-view bijection could contradict a representative's exact spawning
   observation and the stated three-view recovery loss.
3. Held-out release was described per arm instead of once after every fitting-side decision.
4. Shuffled-score direction, representative selection, and rejected-topology metric values were
   not fully defined.
5. Bijection scope, exact-track definitions, component metrics, and denominators were ambiguous.

Required disposition: continue the paired hard-min control to 600 updates; hard-fix each
representative's source assignment in its own view; release held-out data once after all roots and
arms are frozen; define the shuffled score formula and rejection semantics; and define every
assignment/track/component denominator. No scientific arm, threshold, fresh root, or outcome was
opened. The preregistration was amended in place before implementation and must receive a new
independent PASS review.
