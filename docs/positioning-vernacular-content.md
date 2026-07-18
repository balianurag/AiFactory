# Positioning & ICP — Vernacular Content Engine

_Vertical chosen: regional content creators. Method: product-strategist + product-manager-toolkit skills, grounded in the AiFactory engine._

## The product (what AiFactory becomes here)
A creator/D2C brand uploads **one** long-form video/podcast and gets back, in minutes:
dubbed clips, burned-in captions, thumbnails, and platform-tuned posts — in Indian languages.
AiFactory's multi-agent pipeline is the backend (workers: transcribe → translate/dub → clip →
caption → thumbnail → per-platform post copy). The cost-efficiency engine keeps per-video cost
low enough for price-sensitive creators — that's the moat vs Opus Clip et al.

## Positioning statement
> For **regional-language creators and D2C brands in India** who **struggle to turn one video into
> a week of multi-platform, multi-language content**, the Vernacular Content Engine turns a single
> upload into ready-to-post regional clips, captions, and thumbnails **in ~10 minutes for the price
> of a chai** — unlike Opus Clip and global tools that ignore Indic languages and Indian platforms.

## ICP (start narrow, expand later)
- **Who:** regional creators (Marathi / Hindi / Tamil / Telugu / Bengali), 10k–500k followers,
  producing long-form (YouTube / podcast), 1–4 uploads/week.
- **Pain:** repurposing is manual, slow, English-tool-centric; dubbing/subtitling in their language
  is expensive or unavailable; posting across Reels / Shorts / ShareChat / Moj eats hours.
- **Also:** D2C brands wanting regional reach without a content team.

## The wedge — do NOT boil the ocean
Ship the narrowest slice that delivers a "wow" first:
**ONE language + ONE platform + ONE content type.**
e.g. _"Hindi YouTube long-form → 5 Instagram Reels with burned-in Hindi captions + titles."_
Prove that end-to-end on a real creator's video before adding 5 languages.

## Differentiation
- Indic-language dubbing/subtitling (global tools skip this)
- Indian-platform tuning (Reels / Shorts / ShareChat / Moj), not just US formats
- Affordability via the cost-efficiency engine — a real, demonstrable price edge

## Q3 2026 OKR (realistic for pre-launch solo + weekends + ₹1L)
_The skill's default OKR ("150000% growth") is nonsense for pre-launch; this is grounded._

**Objective: Prove the engine has real pull with regional creators.**
- **KR1** — 5 creator interviews done this week; ≥3 rank repurposing a top-3 weekly pain.
- **KR2** — 1 working demo (one language, one platform, one content type) processes a real
  creator's video end-to-end.
- **KR3** — 3 creators try it on their own content; ≥1 says they'd pay.

## START HERE (this week, before any code)
**5 creator interviews.** Distribution beats build. Questions to ask (keep it about their pain,
not your product):
1. Walk me through what you do after recording a video — step by step.
2. Which part eats the most time each week?
3. What have you tried for repurposing / dubbing / captions? What did you drop and why?
4. If it existed, what would "one upload → a week of regional content" be worth to you monthly?
5. Whose content do you admire for multi-language reach — how do they do it?

After the interviews: run `product-manager-toolkit/scripts/customer_interview_analyzer.py` on your
notes to extract pain themes and feature requests, then `rice_prioritizer.py` to rank what to build first.
