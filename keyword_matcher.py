"""
keyword_matcher.py — Stage 4 of the pipeline.

Extracts meaningful keywords from text (stripping stopwords) and computes
a Jaccard overlap score between the reference paper and a candidate document.
"""

import re
from config import KEYWORD_THRESHOLD, MIN_KEYWORD_LENGTH

# ── Mega stopword list (from RAGtocompare.py, kept as-is) ────────────────────
ENGLISH_MEGA_STOPWORDS = {
    "a", "about", "above", "across", "after", "afterwards", "again",
    "against", "all", "almost", "alone", "along", "already", "also",
    "although", "always", "am", "among", "amongst", "an", "and",
    "another", "any", "anybody", "anyhow", "anyone", "anything",
    "anyway", "anywhere", "are", "aren't", "around", "as", "at",
    "back", "be", "became", "because", "become", "becomes",
    "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "both",
    "bottom", "but", "by",
    "call", "can", "can't", "cannot", "cant", "co", "con",
    "could", "couldn't", "couldnt",
    "de", "describe", "detail", "did", "didn't", "didnt", "do",
    "does", "doesn't", "doesnt", "doing", "don't", "done", "dont",
    "down", "due", "during",
    "each", "eg", "eight", "either", "eleven", "else", "elsewhere",
    "empty", "enough", "etc", "even", "ever", "every", "everybody",
    "everyone", "everything", "everywhere", "except",
    "few", "fifteen", "fifty", "fill", "find", "fire", "first",
    "five", "for", "former", "formerly", "forty", "found", "four",
    "from", "front", "full", "further",
    "get", "gets", "getting", "give", "given", "gives", "go",
    "goes", "going", "gone", "got", "gotten",
    "had", "hadn't", "hadnt", "has", "hasn't", "hasnt", "have",
    "haven't", "havent", "having", "he", "he'd", "he'll", "he's",
    "hed", "hell", "hence", "her", "here", "here's", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "hes", "hi",
    "him", "himself", "his", "how", "how's", "howbeit", "however",
    "hundred",
    "i", "i'd", "i'll", "i'm", "i've", "id", "ie", "if", "ignored",
    "ill", "im", "immediate", "in", "inasmuch", "inc", "indeed",
    "indicate", "indicated", "indicates", "inner", "insofar",
    "instead", "into", "inward", "is", "isn't", "isnt", "it",
    "it'd", "it'll", "it's", "itd", "itll", "its", "itself", "ive",
    "just",
    "keep", "keeps", "kept", "know", "known", "knows",
    "last", "lately", "later", "latter", "latterly", "least",
    "less", "lest", "let", "let's", "lets", "like", "liked",
    "likely", "little", "look", "looking", "looks", "ltd",
    "made", "mainly", "make", "makes", "many", "may", "maybe",
    "me", "mean", "meanwhile", "merely", "might", "mine", "more",
    "moreover", "most", "mostly", "much", "must", "mustn't",
    "mustnt", "my", "myself",
    "name", "namely", "nd", "near", "nearly", "necessary", "need",
    "needs", "neither", "never", "nevertheless", "new", "next",
    "nine", "no", "nobody", "non", "none", "noone", "nor",
    "normally", "not", "nothing", "novel", "now", "nowhere",
    "obviously", "of", "off", "often", "oh", "ok", "okay", "old",
    "on", "once", "one", "ones", "only", "onto", "or", "other",
    "others", "otherwise", "ought", "our", "ours", "ourselves",
    "out", "outside", "over", "overall", "own",
    "particular", "particularly", "per", "perhaps", "placed",
    "please", "plus", "possible", "presumably", "probably",
    "provides", "put",
    "que", "quite", "qv",
    "rather", "rd", "re", "really", "reasonably", "regarding",
    "regardless", "regards", "relatively", "respectively", "right",
    "said", "same", "saw", "say", "saying", "says", "second",
    "secondly", "see", "seeing", "seem", "seemed", "seeming",
    "seems", "seen", "self", "selves", "sensible", "sent",
    "serious", "seriously", "seven", "several", "shall", "she",
    "she'd", "she'll", "she's", "shed", "shell", "shes", "should",
    "shouldn't", "shouldnt", "show", "side", "since", "six",
    "some", "somebody", "somehow", "someone", "something",
    "sometime", "sometimes", "somewhat", "somewhere", "soon",
    "sorry", "specified", "specify", "specifying", "still",
    "sub", "such", "sup", "sure",
    "take", "taken", "tell", "tends", "th", "than", "thank",
    "thanks", "thanx", "that", "that's", "thats", "the", "their",
    "theirs", "them", "themselves", "then", "thence", "there",
    "there's", "thereafter", "thereby", "therefore", "therein",
    "theres", "thereupon", "these", "they", "they'd", "they'll",
    "they're", "they've", "theyd", "theyll", "theyre", "theyve",
    "thick", "thin", "third", "this", "thorough", "thoroughly",
    "those", "though", "three", "through", "throughout", "thru",
    "thus", "to", "together", "too", "took", "top", "toward",
    "towards", "tried", "tries", "truly", "try", "trying",
    "twelve", "twenty", "twice", "two",
    "un", "under", "unfortunately", "unless", "unlikely",
    "until", "unto", "up", "upon", "us", "use", "used",
    "useful", "uses", "using", "usually",
    "value", "various", "very", "via", "viz", "vs",
    "want", "wants", "was", "wasn't", "wasnt", "way", "we",
    "we'd", "we'll", "we're", "we've", "wed", "well", "were",
    "weren't", "werent", "weve", "welcome", "went", "were",
    "what", "what's", "whats", "whatever", "when", "when's",
    "whence", "whenever", "whens", "where", "where's",
    "whereafter", "whereas", "whereby", "wherein", "wheres",
    "whereupon", "wherever", "whether", "which", "while",
    "whither", "who", "who's", "whoever", "whole", "whom",
    "whos", "whose", "why", "why's", "whys", "will", "with",
    "within", "without", "won't", "wonder", "wont", "would",
    "wouldn't", "wouldnt",
    "yes", "yet", "you", "you'd", "you'll", "you're", "you've",
    "youd", "youll", "your", "youre", "yours", "yourself",
    "yourselves", "youve", "zero",
    # exam/admin boilerplate
    "instructions", "instruction", "candidates", "candidate",
    "examination", "exam", "exams", "question", "questions",
    "answer", "answers", "paper", "papers", "booklet", "book",
    "sheet", "sheets", "page", "pages", "section", "sections",
    "part", "parts", "total", "marks", "mark", "marking",
    "maximum", "minimum", "time", "hours", "hour", "minutes",
    "minute", "duration", "allowed", "permitted", "prohibited",
    "forbidden", "read", "carefully", "given", "below", "above",
    "attempt", "attempted", "attempting", "write", "written",
    "writing", "choose", "chosen", "choosing", "select", "selected",
    "selecting", "correct", "incorrect", "option", "options",
    "tick", "circle", "fill", "blank", "blanks", "space", "spaces",
    "subject", "code", "date", "session", "year", "roll",
    "rollno", "rollnumber", "registration", "regno", "enrolment",
    "enrollment", "centre", "center", "invigilator", "signature",
    "sign", "signed", "signing", "father", "mother", "school",
    "college", "institute", "university", "board", "council",
    "authority", "principal", "director", "hod", "professor",
    "teacher", "student", "students", "class", "grade",
    "standard", "syllabus", "chapter", "unit", "topic",
    "lesson", "exercise", "practice", "revision", "test",
    "quiz", "assignment", "homework", "project",
    # social/spam
    "telegram", "channel", "channels", "group", "groups",
    "whatsapp", "instagram", "facebook", "twitter", "youtube",
    "tiktok", "snapchat", "linkedin", "reddit", "discord",
    "signal", "wechat", "messenger", "username", "handle",
    "profile", "account", "accounts", "admin", "admins",
    "moderator", "moderators", "member", "members", "join",
    "joined", "joining", "leave", "left", "leaving", "invite",
    "invited", "inviting", "invitation", "invitations",
    "bio", "status", "story", "stories", "reel", "reels",
    "live", "stream", "streaming", "broadcast",
    # single letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
    "w", "x", "y", "z",
    # common adverbs / filler
    "actually", "basically", "certainly", "clearly", "definitely",
    "essentially", "frankly", "honestly", "hopefully",
    "interestingly", "literally", "obviously", "personally",
    "possibly", "practically", "presumably", "probably",
    "really", "seriously", "simply", "surely", "technically",
    "theoretically", "totally", "truly", "typically", "usually",
    "virtually", "well",
    # abbreviations
    "etc", "eg", "ie", "vs", "viz", "cf", "al", "ed", "eds",
    "vol", "vols", "no", "nos", "pp", "fig", "figs",
    "eq", "eqs", "ref", "refs", "sec", "secs", "ch", "chs",
    "app", "apps", "approx", "est", "min", "max", "avg",
    "std", "dev", "var", "corr", "sig", "prob", "info",
    "misc", "temp", "orig", "rev", "ver", "dept", "univ",
    "inst", "assoc", "soc", "govt", "natl", "intl",
    # greetings
    "hello", "hey", "hi", "greetings", "welcome", "thanks",
    "thank", "regards", "sincerely", "best", "kind",
    "please", "sorry", "excuse", "pardon", "oops", "wow",
    "oh", "ah", "um", "uh", "er", "hmm", "hmmm", "yeah",
    "yep", "nope", "nah", "ok", "okay", "alright",
    "sure", "fine", "good", "bad", "great", "awesome",
    "amazing", "nice", "cool", "interesting", "weird",
    "strange", "funny", "sad", "happy", "angry", "excited",
}


def extract_keywords(text: str) -> set[str]:
    """
    Tokenise *text*, remove stopwords and short tokens, return a set of
    meaningful keywords (all lowercase).

    This fixes the original RAGtocompare.py bug where the entire chunk list
    was being added to the set instead of individual tokens.
    """
    tokens = re.findall(r"\b[a-zA-Z]{%d,}\b" % MIN_KEYWORD_LENGTH, text.lower())
    return {t for t in tokens if t not in ENGLISH_MEGA_STOPWORDS}


def jaccard_overlap(ref_keywords: set[str], cand_keywords: set[str]) -> float:
    """
    Jaccard similarity between two keyword sets.
    Returns 0.0 if both sets are empty.
    """
    if not ref_keywords and not cand_keywords:
        return 0.0
    union = ref_keywords | cand_keywords
    if not union:
        return 0.0
    return len(ref_keywords & cand_keywords) / len(union)


def keyword_overlap_ratio(ref_keywords: set[str], cand_keywords: set[str]) -> float:
    """
    What fraction of the *reference* keywords appear in the candidate?
    Useful as an additional signal alongside Jaccard.
    """
    if not ref_keywords:
        return 0.0
    return len(ref_keywords & cand_keywords) / len(ref_keywords)


def passes_keyword_threshold(
    ref_keywords: set[str],
    cand_keywords: set[str],
    threshold: float = KEYWORD_THRESHOLD,
) -> bool:
    """Returns True if the candidate clears the keyword similarity bar."""
    return jaccard_overlap(ref_keywords, cand_keywords) >= threshold


def get_matching_keywords(ref_keywords: set[str], cand_keywords: set[str]) -> set[str]:
    """Return the actual overlapping keyword tokens for snippet extraction."""
    return ref_keywords & cand_keywords


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_ref = """
    The electromagnetic induction experiment demonstrates Faraday's law.
    The solenoid produces magnetic flux when current flows through the coil.
    Lenz's law opposes the change in magnetic flux.
    """
    sample_cand = """
    Faraday discovered electromagnetic induction. The change in magnetic flux
    through a solenoid coil induces an electromotive force (EMF).
    This is the fundamental principle behind electric generators.
    """
    ref_kw = extract_keywords(sample_ref)
    cand_kw = extract_keywords(sample_cand)
    print("Reference keywords:", ref_kw)
    print("Candidate keywords:", cand_kw)
    print("Jaccard overlap:   ", round(jaccard_overlap(ref_kw, cand_kw), 3))
    print("Ref coverage:      ", round(keyword_overlap_ratio(ref_kw, cand_kw), 3))
    print("Passes threshold:  ", passes_keyword_threshold(ref_kw, cand_kw))
