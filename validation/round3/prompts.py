"""Round 2 real-world validation prompts. All NEW — none reused from
round 1's validation/prompts.py. Split roughly evenly across three groups
(search-only, extraction-only, both-in-sequence), written to widen real
variety (obscure topics, non-English terms, very long/rambling questions,
very short ones) rather than to engineer any specific failure. Real URLs
in the extraction group are genuine, existing pages picked for organic
diversity (language, age, format) — no synthetic test endpoints
(httpbin.org and similar were used only to characterize the tool during
development, never put in front of the agent).
"""

# -- search-only: no URL given, model must use the search tool -------------
SEARCH_PROMPTS: list[str] = [
    "What is ikigai and where does the concept come from?",
    "Explique-moi ce qu'est le concept de 'hygge' danois.",
    "What is the Voynich manuscript?",
    "Tuvan throat singing — what is it and where is it practiced?",
    "What's the deal with the Bloomsbury Group?",
    "saudade",
    "Who was Hypatia of Alexandria?",
    "What is the current status of the Kessler syndrome as a real orbital debris risk?",
    "How does mycorrhizal networking between trees actually work — is the 'wood wide web' real?",
    "What is Ubuntu (the philosophy, not the OS)?",
    "この前読んだ量子もつれの記事、要点を教えて",
    "Tell me about the Antikythera mechanism.",
    "I've been hearing a lot about something called 'degrowth economics' lately and I honestly "
    "don't understand what it actually proposes beyond just 'consume less' — can you explain "
    "what the actual policy ideas are and who the major proponents are?",
    "qanat",
    "What's a katabatic wind?",
    "Who invented the moka pot?",
    "Explain the Ship of Theseus paradox and how modern philosophers have responded to it.",
    "What happened to the Library of Alexandria, really?",
    "gyre",
    "What's the current scientific consensus on the health effects of intermittent fasting?",
    "Tell me about the Great Emu War.",
    "What is a palimpsest?",
]

# -- extraction-only: a real, specific URL is given -------------------------
EXTRACTION_PROMPTS: list[str] = [
    "Summarize the content of https://en.wikipedia.org/wiki/Quantum_computing",
    "What does this page say? https://eo.wikipedia.org/wiki/Esperanto",
    "Can you read https://ja.wikipedia.org/wiki/量子コンピュータ and tell me the gist in English?",
    "What's on this page: https://www.iana.org/domains/reserved",
    "Read http://info.cern.ch/hypertext/WWW/TheProject.html and tell me what it's about — it's "
    "apparently a historically significant page.",
    "Pull the text from https://www.gutenberg.org/files/1342/1342-0.txt and tell me what book "
    "this is and how it opens.",
    "What's covered on https://www.python.org/",
    "Summarize https://go.dev/",
    "https://www.rust-lang.org/ — what's the pitch on this homepage?",
    "Read https://news.ycombinator.com/ and tell me what's being discussed today.",
    "https://www.nasa.gov/ — summarize the front page.",
    "What does https://www.who.int/ currently highlight?",
    "Summarize https://www.un.org/en/",
    "Can you get the abstract from https://arxiv.org/abs/2301.00001",
    "Try reading https://arxiv.org/pdf/2301.00001 directly and tell me what's in it.",
    "https://www.lemonde.fr/ — qu'est-ce qui fait l'actualité aujourd'hui ?",
    "Was steht gerade auf https://www.spiegel.de/ ?",
    "Read https://www.bbc.com/news and summarize the top stories.",
    "What's on https://www.w3.org/",
    "Get the text of https://simple.wikipedia.org/wiki/Quantum_computing — I want the simple "
    "version explained even more simply if you can.",
    "https://www.iana.org/domains/reserved — extract this and just list the domains mentioned.",
    "Read this for me please: https://www.nasa.gov/",
]

# -- both in sequence: model must find something, then read/visit it -------
BOTH_PROMPTS: list[str] = [
    "Find the official website of the Python Software Foundation and summarize its homepage.",
    "Find the homepage of the Rust project and tell me what its main selling points are.",
    "Search for the official site of the World Health Organization and summarize what's "
    "currently on its front page.",
    "Find a recent Wikipedia article about black holes and summarize it.",
    "Look up the Esperanto Wikipedia page and tell me what it says, even though you'll need to "
    "translate.",
    "Find the arXiv page for a paper about transformer architectures and read its abstract.",
    "Search for the official NASA website and tell me what they're currently featuring.",
    "Find Le Monde's website and tell me the top story, in English.",
    "Look up information about the Antikythera mechanism, then try to find and read a page that "
    "goes into more depth than the summary.",
    "Find out where I can read the full text of Pride and Prejudice for free online, then pull "
    "the opening lines.",
    "Search for information on katabatic winds and then find and read a more detailed source on "
    "the topic.",
    "Find the Hacker News homepage and summarize what's trending.",
    "I want to know more about the Voynich manuscript — find a good source and read it in full.",
    "Search for 'Ship of Theseus' and then read the most relevant page you find in full.",
    "Find the German news site Spiegel and summarize today's top headline.",
    "Look up degrowth economics, then find a source that discusses actual policy proposals in "
    "detail and read it.",
    "Find the W3C's website and summarize its homepage.",
    "Search for information about the Great Emu War, then read a detailed account of it.",
]

PROMPTS: list[str] = SEARCH_PROMPTS + EXTRACTION_PROMPTS + BOTH_PROMPTS

assert len(PROMPTS) >= 60, f"expected at least 60 prompts, got {len(PROMPTS)}"
assert len(SEARCH_PROMPTS) >= 15 and len(EXTRACTION_PROMPTS) >= 15, "groups should be roughly even"
