# Message Template for Recruiters

Copy-paste this into emails/LinkedIn when sending the link:

---

## Version 1 (Technical Deep Dive)

Hi [Name],

Following up on the [specific role] opportunity. I've been building a production-grade RL training infrastructure from scratch to deepen my understanding of modern training pipelines.

The project covers the full stack:
- **Environment Design**: Deterministic 5×5 grid MDP with structured rewards
- **Training Loop**: REINFORCE policy gradient optimization (Sutton & Barto)
- **Post-training**: DPO (Direct Preference Optimization) for preference learning
- **Efficiency**: LoRA and QLoRA for parameter-efficient fine-tuning
- **Evaluation**: Continuous harness measuring reliability on unseen goals
- **Infrastructure**: Docker + K8s for reproducible deployment

Every component is built from first principles with rigorous math. I understand the "why" behind each piece, not just the code.

This directly aligns with your focus on [post-training | evaluation systems | agent reliability]. The repo is public if you'd like to review:

github.com/anthonylewallen/gridworld

Current status: Phase 0 complete (working config + logging), Phase 1 taught (MDP fundamentals), building Phase 2+ this week.

Happy to discuss the design or any specific components.

Best,
Anthony

---

## Version 2 (Brief)

Hi [Name],

Building production RL infrastructure to demonstrate understanding of training pipelines, post-training optimization, and deployment systems.

Covers: environment design, REINFORCE, DPO, LoRA fine-tuning, evaluation harness, Docker/K8s.

Repo: github.com/anthonylewallen/gridworld

Phase 0-1 complete, Phase 2+ in progress. Would love to discuss the approach.

Best,
Anthony

---

## Which to Use

- **Technical roles** (Willie, Reducto, NewtonX): Use Version 1 (shows depth)
- **Fast-turnaround recruiting**: Use Version 2 (quick, credible)
- **Follow-ups**: Use Version 2 (they've already heard pitch once)

---

## Timing

- **Send after pushing to GitHub** (have a clean repo to show)
- **Monday morning** (better response rate than Friday evening)
- **Personalize [Name] and [specific role]** (shows you read the job posting)

---

## What They're Looking For

When they click the link, they see:
- ✅ Clean code structure (modular, organized)
- ✅ Production thinking (config, logging, Docker, K8s)
- ✅ Rigor (math in design.md, equations in README, references to papers)
- ✅ Progress (Phase 0 working, Phase 1-8 documented, on track)
- ✅ Ownership (every line is yours, you understand it all)

This is way more impressive than "completed a tutorial."

---

## Expected Response

Good recruiters will:
1. Check the repo (takes 5 min)
2. See it's serious (production thinking, not toy project)
3. Bump you up to hiring manager
4. Schedule a call

Bad recruiters will:
- Copy-paste form response
- Not look at repo
- Move on

Filter accordingly.
