def make_final_summarize_prompt(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, provide the most specific supported date/time phrase. Use the format \"15 July 2023\" for explicit dates, but keep relative phrases such as \"last week\", \"the previous Friday\", or \"the weekend before 22 July 2023\" when that is what the summary supports. Do not replace a relative phrase with its anchor date unless the summary explicitly says the event happened on that anchor date. Only provide one year, date, or time phrase, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.
For "why" questions, answer with the direct cause or motivation from the summary. If the summary gives specific reasons after words like "because", "since", "as", "after", "motivated by", "wanted", or "dreaming of", keep all direct reasons rather than only the first trigger phrase or a vague background phrase. If both a long-term motivation and a triggering event explain the action, include both.
For non-date questions that ask for plural items, examples, events, values, reasons, or "some" things, include all directly relevant specific items from the summary, separated by commas. Do not choose only one representative item when several answer the question. If the question contains a qualifier such as "through", "by", "after", "on", or "about", answer only the fact tied to that qualifier rather than nearby background facts; phrases introduced by "specifically", "by", or "through" are usually the best answer span.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def make_final_summarize_prompt_category3(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""


def build_final_summarize_prompt(category: int | None, summary: str, question: str) -> str:
    if category == 3:
        return make_final_summarize_prompt_category3(summary, question)
    return make_final_summarize_prompt(summary, question)
