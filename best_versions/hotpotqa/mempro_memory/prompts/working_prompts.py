def make_prompt(summary: str, question: str) -> str:
    """Build the final HotpotQA answer prompt from research_summary."""
    return f"""You are a careful HotpotQA short-answer extractor.
Use only the given Context.
Answer with ONLY the final answer string; no extra words.

Rules:
- Return the shortest complete phrase that answers the exact Question.
- Do not answer with the bridge entity when the Question asks for its role, date, place, work, organization, or other related fact.
- Do not answer with an entity explicitly excluded by words such as "aside from", "besides", "other than", "except", or "not"; answer the remaining matching entity.
- If the Question asks who/which person/this man/this woman, answer the person's name, not a description, citizenship, date, or job.
- If the Question asks for a position, role, title, work, organization, road, event, type, place, date, or number, output only that requested span.
- If the Question asks what government position was held and the Context lists several offices, choose the office that directly names a government position rather than a less direct biographical post.
- Prefer the common short form of an office, person, place, work, organization, or concept when it is sufficient; avoid unnecessary formal expansions.
- If the Question asks "formerly known as what", answer the former name, not the current name.
- If the Context gives both total capacity and seated capacity, and the Question asks how many people an arena can seat, use the seated capacity phrase.
- If the Question asks how many hypermarkets or stores at a specified time, copy the exact count phrase tied to that time; do not substitute a total company store count.
- If the Question asks for a count of a typed object, output the number and keep the object word only when it is part of the natural answer phrase.
- If the Question asks for the inhabitant or inhabitants of a place and the Context gives a dated population or inhabitants count, answer the count phrase rather than a demonym.
- If the Question asks when someone was born and the Context includes a full day-month-year birth date, output the full date, not only the year.
- If the Question asks "what was the election" or asks for an event/work/series title, output the title/name of the event/work/series, not the date or outcome.
- If the Question asks "what other film" after naming one film as the bridge clue, do not answer that bridge film; answer the additional film listed with the same writer/director.
- If the Question asks where a band, organization, or group "hails from", use the formation/origin place when the Context provides it.
- If the Question asks for "other writers" and the Context gives a "was written by" list containing the main artist plus other names, output only the other names from that written-by list. Do not add producers, performers, collaborations, or writers of different songs.
- For a list of names, keep the natural conjunction before the final item when it appears in the Context.
- If the Question asks what campaign two groups embrace, include the word "campaign" when the Context names it as a campaign.
- If the Context names a vehicle, aircraft, weapon, office, or other typed entity with a useful type word, keep the type word unless the Question asks only for a model/name.
- If the Question asks what an event secured, caused, established, or achieved, output the shortest noun phrase that names that outcome instead of adding explanatory clauses.
- If the Context says "X, also known as Y", answer X unless the Question asks for an alias.
- If a modifier or relative clause in the Question identifies one item from a list, answer only that item, not the whole list. For example, if the Question asks which war "had over 60 million casualties", answer the war that has that property.
- For place answers, match the granularity implied by the Question and Context; do not always choose the most specific sub-place.
- Preserve essential qualifiers already in the answer phrase, such as "more than", "at least", "million", "seated", or "inhabitants". Do not add unnecessary units.
- For yes/no questions, answer exactly "yes" or "no".
- Do not answer "not provided" if the Context contains a plausible answer.

Question:
{question}

Context:
{summary}

Answer:
"""
