"""Hand-written seed content for the synthetic benchmark.

Each seed is a prompt with a reference answer broken into key facts, one plausible false
claim (used to plant hallucinations), and machine-checkable constraints. Seeds marked
`ambiguous` are underspecified or subjective; the generator gives them noisier annotations.
"""

from __future__ import annotations

SEEDS: list[dict] = [
    # ------------------------------------------------------------------ factual QA
    {
        "category": "factual_qa",
        "prompt": "Why is the sky blue? Answer in under 80 words.",
        "constraints": {"max_words": 80, "format": "paragraph"},
        "facts": [
            "The sky looks blue because of Rayleigh scattering of sunlight by air molecules.",
            "Shorter wavelengths like blue scatter much more strongly than longer red wavelengths.",
            "We see blue rather than violet partly because our eyes are more sensitive to blue and some violet light is absorbed high in the atmosphere.",
        ],
        "false": "The sky is blue mainly because it reflects the color of the oceans.",
    },
    {
        "category": "factual_qa",
        "prompt": "What causes the seasons on Earth?",
        "constraints": {"max_words": 100},
        "facts": [
            "Seasons are caused by the tilt of Earth's axis, about 23.5 degrees.",
            "The hemisphere tilted toward the Sun gets more direct sunlight and longer days, so it has summer.",
            "Distance is not the cause; Earth is actually closest to the Sun in early January.",
        ],
        "false": "Earth has summer when its orbit brings it much closer to the Sun.",
    },
    {
        "category": "factual_qa",
        "prompt": "Who wrote Pride and Prejudice, and when was it first published?",
        "constraints": {"max_words": 60},
        "facts": [
            "Pride and Prejudice was written by Jane Austen.",
            "It was first published in 1813.",
            "It was published anonymously, credited to the Author of Sense and Sensibility.",
        ],
        "false": "Austen co-wrote the novel with her sister Cassandra.",
    },
    {
        "category": "factual_qa",
        "prompt": "What is the boiling point of water at sea level, and how does altitude change it?",
        "constraints": {"max_words": 90},
        "facts": [
            "At sea level, water boils at 100 degrees Celsius, or 212 degrees Fahrenheit.",
            "At higher altitude the air pressure is lower, so water boils at a lower temperature.",
            "The boiling point drops by roughly 1 degree Celsius for every 300 meters of elevation.",
        ],
        "false": "Water boils at a higher temperature on mountains because the air is thinner.",
    },
    {
        "category": "factual_qa",
        "prompt": "What is photosynthesis?",
        "constraints": {"max_words": 90},
        "facts": [
            "Photosynthesis is how plants, algae, and some bacteria turn light energy into chemical energy.",
            "It uses carbon dioxide and water to make glucose.",
            "Oxygen is released as a byproduct, and the process takes place in chloroplasts using chlorophyll.",
        ],
        "false": "Photosynthesis takes place mainly in the plant's roots.",
    },
    {
        "category": "factual_qa",
        "prompt": "What is the difference between a virus and a bacterium?",
        "constraints": {"max_words": 100},
        "facts": [
            "Bacteria are single-celled living organisms that can reproduce on their own.",
            "Viruses are much smaller and can only replicate inside a host cell.",
            "Antibiotics can treat bacterial infections but do not work against viruses.",
        ],
        "false": "Antibiotics are the standard treatment for viral infections such as the flu.",
    },
    {
        "category": "factual_qa",
        "prompt": "When did the Berlin Wall fall, and why was it significant?",
        "constraints": {"max_words": 90},
        "facts": [
            "The Berlin Wall fell on 9 November 1989.",
            "It had been the main symbol of the Cold War division between East and West.",
            "Its fall led to German reunification in October 1990.",
        ],
        "false": "The wall was brought down by a NATO military operation.",
    },
    {
        "category": "factual_qa",
        "prompt": "What does DNS do on the internet?",
        "constraints": {"max_words": 100},
        "facts": [
            "DNS translates human-readable domain names into IP addresses.",
            "It is a hierarchy of root, top-level domain, and authoritative name servers.",
            "Resolvers cache answers for a time set by each record's TTL to speed up lookups.",
        ],
        "false": "DNS also encrypts all traffic between your browser and the website.",
    },
    {
        "category": "factual_qa",
        "prompt": "What is the speed of light in a vacuum?",
        "constraints": {"max_words": 70},
        "facts": [
            "The speed of light in a vacuum is exactly 299,792,458 meters per second, about 300,000 km per second.",
            "It is a universal physical constant, written as c.",
            "No object with mass can reach the speed of light.",
        ],
        "false": "Light travels faster in water than it does in a vacuum.",
    },
    {
        "category": "factual_qa",
        "prompt": "What is the largest planet in our solar system? Give two facts about it.",
        "constraints": {"max_words": 80},
        "facts": [
            "Jupiter is the largest planet in the solar system.",
            "It is a gas giant made mostly of hydrogen and helium.",
            "Its mass is more than twice that of all the other planets combined.",
        ],
        "false": "Jupiter has a solid rocky surface that spacecraft can land on.",
    },
    {
        "category": "factual_qa",
        "prompt": "What is inflation in economics?",
        "constraints": {"max_words": 90},
        "facts": [
            "Inflation is a general rise in prices that reduces the purchasing power of money.",
            "It is commonly measured with price indices such as the Consumer Price Index.",
            "Many central banks target about 2 percent inflation and adjust interest rates to manage it.",
        ],
        "false": "Inflation means the total amount of goods an economy produces is shrinking.",
    },
    {
        "category": "factual_qa",
        "prompt": "What do mitochondria do in a cell?",
        "constraints": {"max_words": 80},
        "facts": [
            "Mitochondria produce most of the cell's ATP through cellular respiration.",
            "They have their own DNA, separate from the cell nucleus.",
            "They also help regulate cell death and calcium signaling.",
        ],
        "false": "Mitochondria are found only in plant cells.",
    },
    # ------------------------------------------------------------------ summarization
    {
        "category": "summarization",
        "prompt": (
            "Summarize the following announcement in no more than 50 words.\n\n"
            "The Riverside Public Library will extend its weekday hours starting March 3, staying open "
            "until 9 p.m. Monday through Thursday. The change follows a community survey in which 68 "
            "percent of respondents asked for evening access. The extended hours are funded by a two-year "
            "state grant, and the library will hire four part-time staff to cover the new shifts."
        ),
        "constraints": {"max_words": 50, "format": "paragraph"},
        "facts": [
            "Riverside Public Library will stay open until 9 p.m. Monday through Thursday starting March 3.",
            "The change responds to a survey where 68 percent of respondents wanted evening hours.",
            "A two-year state grant funds it, including four new part-time staff.",
        ],
        "false": "The library will also start charging a small fee for evening visits.",
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize this study in exactly 3 bullet points.\n\n"
            "Researchers at a university sleep lab tracked 120 adults for six weeks. Participants who slept "
            "at least seven hours a night scored 20 percent higher on memory recall tests than those who "
            "slept under six hours. The team noted the study was observational, so it cannot prove that "
            "sleep directly causes better memory, and called for a controlled trial."
        ),
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "A six-week study of 120 adults linked sleeping seven or more hours to 20 percent better memory recall.",
            "The study was observational, so it cannot show that sleep causes better memory.",
            "The researchers called for a controlled trial to test the effect.",
        ],
        "false": "The researchers concluded that sleep deprivation causes permanent memory loss.",
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize this earnings report in one paragraph of at most 60 words.\n\n"
            "Northwind Outfitters reported third-quarter revenue of $412 million, up 9 percent from a year "
            "earlier, driven mainly by online sales. Net profit fell 4 percent to $31 million because of "
            "higher shipping costs. The company kept its full-year revenue forecast unchanged and said it "
            "plans to open 12 new stores next year."
        ),
        "constraints": {"max_words": 60, "format": "paragraph"},
        "facts": [
            "Northwind's third-quarter revenue rose 9 percent to $412 million, mostly from online sales.",
            "Net profit fell 4 percent to $31 million due to higher shipping costs.",
            "The company kept its full-year forecast and plans 12 new stores next year.",
        ],
        "false": "The CEO announced she will step down at the end of the year.",
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize this news item in exactly 3 bullet points.\n\n"
            "The city council approved a $2.3 million plan to restore Mill Creek Park. The project will "
            "replace invasive plants with native species, rebuild the eroded creek banks, and add a "
            "1.5-mile accessible trail. Work begins in April and is expected to take 18 months, during "
            "which the park's north section will be closed."
        ),
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "The council approved $2.3 million to restore Mill Creek Park.",
            "The plan replaces invasive plants with native species, rebuilds creek banks, and adds a 1.5-mile accessible trail.",
            "Work starts in April, lasts about 18 months, and closes the north section meanwhile.",
        ],
        "false": "A new parking garage will be built at the park entrance.",
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize these release notes for end users in exactly 3 bullet points.\n\n"
            "Version 4.2 of the Tidepool editor adds a built-in terminal panel and support for split views. "
            "It fixes a bug that caused the editor to freeze when opening files larger than 50 MB. Support "
            "for the legacy plugin API has been removed; plugin authors must migrate to the new API before "
            "upgrading."
        ),
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "Tidepool 4.2 adds a built-in terminal panel and split views.",
            "It fixes a freeze when opening files larger than 50 MB.",
            "The legacy plugin API was removed, so plugins must migrate to the new API.",
        ],
        "false": "Version 4.2 also doubles the editor's startup speed.",
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize this survey in at most 60 words.\n\n"
            "A survey of 2,000 office workers found that 61 percent prefer a hybrid schedule, 24 percent "
            "want to work fully remotely, and 15 percent prefer being in the office full time. Respondents "
            "most often cited commute time as the reason for wanting remote days, while those preferring "
            "the office cited easier collaboration. Managers in the survey were more likely than staff to "
            "favor full-time office work."
        ),
        "constraints": {"max_words": 60},
        "facts": [
            "Of 2,000 office workers surveyed, 61 percent prefer hybrid, 24 percent fully remote, and 15 percent full-time office.",
            "Commute time was the main reason for wanting remote days, while office supporters cited collaboration.",
            "Managers favored full-time office work more often than staff did.",
        ],
        "false": "Most workers said they would quit if required to return to the office.",
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize this notice in no more than 50 words.\n\n"
            "The Harbor Street Bridge will close to all vehicle traffic from June 10 to August 22 for deck "
            "repairs. Pedestrians and cyclists can still cross using a protected lane on the east side. "
            "Drivers are advised to use the Fifth Avenue Bridge as a detour, and two extra bus routes will "
            "run during the closure."
        ),
        "constraints": {"max_words": 50},
        "facts": [
            "Harbor Street Bridge closes to vehicles from June 10 to August 22 for deck repairs.",
            "Pedestrians and cyclists can still cross on a protected east-side lane.",
            "Drivers should detour via the Fifth Avenue Bridge, and two extra bus routes will run.",
        ],
        "false": "Tolls on the bridge will increase after it reopens.",
    },
    # ------------------------------------------------------------------ instruction following
    {
        "category": "instruction_following",
        "prompt": "Give exactly 3 tips for better password security as bullet points.",
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "Use a long, unique password for every account.",
            "Store passwords in a reputable password manager.",
            "Turn on two-factor authentication wherever it is offered.",
        ],
        "false": "Writing your password on a sticky note on your monitor is a safe backup.",
    },
    {
        "category": "instruction_following",
        "prompt": "Explain recursion to a 10-year-old in under 60 words.",
        "constraints": {"max_words": 60, "format": "paragraph"},
        "facts": [
            "Recursion is when you solve a problem by solving a smaller version of the same problem.",
            "It is like nesting dolls: you open one doll to find a smaller one inside, again and again.",
            "Every recursion needs a stopping point, like the tiniest doll that does not open.",
        ],
        "false": "Recursion means a computer repeats the exact same problem forever without changing it.",
        "ambiguous": True,
    },
    {
        "category": "instruction_following",
        "prompt": "Write a product description for a reusable steel water bottle in exactly 3 bullet points.",
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "Double-wall insulated steel keeps drinks cold for 24 hours or hot for 12.",
            "A leak-proof lid makes it safe to toss in any bag.",
            "The reusable design cuts down on single-use plastic bottles.",
        ],
        "false": "It is certified to purify water from any source instantly.",
    },
    {
        "category": "instruction_following",
        "prompt": "List the 3 main steps for making pour-over coffee as bullet points.",
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "Heat water to about 93 to 96 degrees Celsius and grind the beans medium-fine.",
            "Pour a little water over the grounds and let them bloom for about 30 seconds.",
            "Pour the rest slowly in circles, finishing in about 3 minutes.",
        ],
        "false": "Use water heated to 120 degrees Celsius in an open kettle for the best extraction.",
    },
    {
        "category": "instruction_following",
        "prompt": "Write a polite email declining a meeting invitation for Thursday. Keep it under 80 words.",
        "constraints": {"max_words": 80, "format": "paragraph"},
        "facts": [
            "Thank you for the invitation to Thursday's meeting.",
            "Unfortunately, I have a prior commitment and cannot attend.",
            "Could you share the notes afterward, or suggest another time that works?",
        ],
        "false": "As I mentioned to your manager, I will be on leave for the whole month.",
    },
    {
        "category": "instruction_following",
        "prompt": "Summarize the rules of tic-tac-toe in exactly 3 bullet points.",
        "constraints": {"format": "bullets", "num_bullets": 3},
        "facts": [
            "Two players take turns marking X or O in a 3-by-3 grid.",
            "The first player to get three marks in a row, column, or diagonal wins.",
            "If all nine squares fill with no line of three, the game is a draw.",
        ],
        "false": "Each player may erase one of the opponent's marks once per game.",
    },
    {
        "category": "instruction_following",
        "prompt": "Describe what a hash table is in one paragraph of no more than 70 words. Do not include code.",
        "constraints": {"max_words": 70, "format": "paragraph"},
        "facts": [
            "A hash table stores key-value pairs for fast lookup.",
            "A hash function maps each key to an index in an underlying array.",
            "Collisions are handled with chaining or open addressing, giving average constant-time operations.",
        ],
        "false": "Hash tables automatically keep their keys in sorted order.",
    },
    {
        "category": "instruction_following",
        "prompt": "List 3 benefits of regular exercise as bullet points, under 45 words total.",
        "constraints": {"max_words": 45, "format": "bullets", "num_bullets": 3},
        "facts": [
            "Strengthens the heart and lowers blood pressure.",
            "Improves mood and reduces symptoms of anxiety.",
            "Helps maintain a healthy weight and better sleep.",
        ],
        "false": "Removes the need for any vaccinations.",
    },
    # ------------------------------------------------------------------ explanation
    {
        "category": "explanation",
        "prompt": "How does binary search work?",
        "constraints": {"max_words": 120},
        "facts": [
            "Binary search works on a sorted array.",
            "It compares the target with the middle element and discards the half that cannot contain it.",
            "Because the range halves each step, it runs in O(log n) time.",
        ],
        "false": "Binary search works just as well on unsorted data.",
    },
    {
        "category": "explanation",
        "prompt": "How do vaccines work?",
        "constraints": {"max_words": 120},
        "facts": [
            "A vaccine exposes the immune system to a harmless piece or weakened form of a pathogen.",
            "The body responds by making antibodies and memory cells.",
            "If the real pathogen appears later, the immune system recognizes it and responds faster.",
        ],
        "false": "Vaccines work by directly killing viruses that are already in your bloodstream.",
    },
    {
        "category": "explanation",
        "prompt": "Explain how to reverse a singly linked list in place.",
        "constraints": {"max_words": 120},
        "facts": [
            "Walk the list while keeping three pointers: previous, current, and next.",
            "At each node, point current's next back to previous, then advance all three.",
            "It takes O(n) time and O(1) extra space.",
        ],
        "false": "Reversing a linked list requires copying it into a new array first.",
    },
    {
        "category": "explanation",
        "prompt": "What does git rebase do, and when should you avoid it?",
        "constraints": {"max_words": 120},
        "facts": [
            "Git rebase replays your commits on top of another branch's latest commit.",
            "This produces a linear history without merge commits.",
            "It rewrites commit hashes, so avoid rebasing branches others have already pulled.",
        ],
        "false": "Rebase permanently deletes the branch you rebase onto.",
    },
    {
        "category": "explanation",
        "prompt": "How does compound interest work?",
        "constraints": {"max_words": 120},
        "facts": [
            "Compound interest is earned on both the principal and previously earned interest.",
            "The formula is A = P(1 + r/n)^(nt).",
            "Longer time periods and more frequent compounding make the balance grow faster.",
        ],
        "false": "Compound interest is calculated only on the original principal every year.",
    },
    {
        "category": "explanation",
        "prompt": "How does a refrigerator keep food cold?",
        "constraints": {"max_words": 120},
        "facts": [
            "Refrigerant absorbs heat from inside the fridge as it evaporates.",
            "A compressor pressurizes the refrigerant gas, raising its temperature.",
            "The heat is released through condenser coils on the back or bottom of the fridge.",
        ],
        "false": "A refrigerator makes cold air by burning a small amount of gas.",
    },
    {
        "category": "explanation",
        "prompt": "What is overfitting in machine learning, and how do you prevent it?",
        "constraints": {"max_words": 120},
        "facts": [
            "Overfitting happens when a model learns noise in the training data instead of the general pattern.",
            "It scores well on training data but poorly on new, unseen data.",
            "Regularization, more data, cross-validation, and early stopping help prevent it.",
        ],
        "false": "Training for more epochs is the standard way to fix overfitting.",
    },
    # ------------------------------------------------------------------ open-ended
    {
        "category": "open_ended",
        "prompt": "What's the best programming language?",
        "constraints": {"max_words": 120},
        "facts": [
            "There is no single best language; it depends on what you want to build.",
            "Python is popular for data work and scripting, JavaScript for web front ends, and C or Rust for systems programming.",
            "Consider the ecosystem, available libraries, and what your team already knows.",
        ],
        "false": "Python is the fastest programming language for every kind of task.",
        "ambiguous": True,
    },
    {
        "category": "open_ended",
        "prompt": "Write a short poem about autumn, no more than 50 words.",
        "constraints": {"max_words": 50},
        "facts": [
            "Red and amber leaves drift down to the cold ground.",
            "The days grow short as summer slips away.",
            "A quiet chill settles over the evening fields.",
        ],
        "false": "Snow melts away as the spring flowers bloom today.",
        "ambiguous": True,
    },
    {
        "category": "open_ended",
        "prompt": "Is it better to rent or buy a home?",
        "constraints": {"max_words": 120},
        "facts": [
            "It depends on how long you plan to stay, your finances, and the local housing market.",
            "Buying builds equity over time but comes with down payments, maintenance, and transaction costs.",
            "Renting offers flexibility and lower upfront costs.",
        ],
        "false": "Buying a home is always cheaper than renting in the long run.",
        "ambiguous": True,
    },
    {
        "category": "open_ended",
        "prompt": "Tell me about Mercury.",
        "constraints": {"max_words": 120},
        "facts": [
            "Mercury is the smallest planet and the closest to the Sun.",
            "Mercury is also a chemical element, symbol Hg, and the only metal that is liquid at room temperature.",
            "In Roman mythology, Mercury was the messenger of the gods.",
        ],
        "false": "Mercury is the hottest planet in the solar system.",
        "ambiguous": True,
    },
    {
        "category": "open_ended",
        "prompt": "Make this message sound more professional: 'hey, cant make it tmrw, something came up'",
        "constraints": {"max_words": 50},
        "facts": [
            "Hello, I'm sorry, but I won't be able to make it tomorrow.",
            "Something unexpected has come up.",
            "Could we find another time that works for you?",
        ],
        "false": "I have been admitted to the hospital with a sudden illness.",
        "ambiguous": True,
    },
    {
        "category": "open_ended",
        "prompt": "How long should a cover letter be?",
        "constraints": {"max_words": 100},
        "facts": [
            "A cover letter should usually fit on one page.",
            "Aim for roughly 250 to 400 words.",
            "Three or four short paragraphs is a common structure.",
        ],
        "false": "A strong cover letter should be at least two full pages.",
    },
]
