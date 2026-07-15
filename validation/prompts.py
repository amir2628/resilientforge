"""Real-world validation prompts (Step 3). Written by hand to cover a
believable range a real user might ask a web-search agent — NOT designed to
trigger any of the 5 existing failure-injection scenarios (those are all
about a single-argument calendar-event-style tool: natural-language dates,
malformed JSON, missing fields, wrong types, ambiguous fix candidates —
none of that applies to a free-text `query: str` search tool). Whatever
failures show up here come from the real DuckDuckGo search API, the real
local model's tool-calling, and real content on the web — not anything
scripted.

Mix: factual lookups, ambiguous/vague queries, and multi-step questions
that likely require more than one search.
"""

PROMPTS: list[str] = [
    # -- factual lookups --------------------------------------------------
    "What is the boiling point of water at sea level in Celsius?",
    "Who wrote the novel 'One Hundred Years of Solitude'?",
    "What is the capital of Mongolia?",
    "When was the Eiffel Tower completed?",
    "What is the speed of light in a vacuum, in km/s?",
    "Who is the current Secretary-General of the United Nations?",
    "What programming language was Django written in?",
    "How tall is the Burj Khalifa?",
    "What year did the Berlin Wall fall?",
    "Who painted 'The Starry Night'?",
    "What is the largest desert in the world by area?",
    "What currency is used in Switzerland?",
    # -- ambiguous / vague queries -----------------------------------------
    "What's that new thing everyone's talking about in AI this week?",
    "Tell me about the situation in the housing market right now.",
    "What's a good book to read?",
    "How's the weather been affecting crops lately?",
    "What happened in tech news recently?",
    "Is it a good time to learn a new programming language?",
    "What are people saying about electric cars these days?",
    "Give me an update on the space industry.",
    # -- multi-step / comparison questions (likely >1 search) ---------------
    "Compare the population of Lisbon and Porto, and tell me which is bigger.",
    "Which is older, the University of Oxford or the University of Bologna?",
    "What's the difference between Python's asyncio and threading modules?",
    "Compare the GDP of Vietnam and the Philippines.",
    "Who has won more World Cups, Brazil or Germany?",
    "What's the distance between Tokyo and Osaka, and how long does the "
    "bullet train take?",
    "Compare the battery range of the Tesla Model 3 and the Chevy Bolt.",
    "Between Rust and Go, which one is generally considered faster for "
    "systems programming, and why?",
    "What's the tallest building in the world, and how does it compare in "
    "height to the Eiffel Tower?",
    "Find out who the founder of LangChain is, and what company they work "
    "at now.",
    "What's the most recent stable version of Python, and when was it "
    "released?",
    "How many moons does Jupiter have, and how does that compare to Saturn?",
    "What is the exchange rate trend between the US dollar and the "
    "Japanese yen this year?",
    "Who is the current president of Brazil, and when did they take "
    "office?",
    "What's the tallest mountain in Africa, and how does its height "
    "compare to Mount Fuji?",
]

assert len(PROMPTS) >= 30, f"expected at least 30 prompts, got {len(PROMPTS)}"
