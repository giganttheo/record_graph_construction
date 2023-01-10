import re

# The following code is extracted from gensim.summarization.textcleaner with gensim==3.6

RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)
SEPARATOR = '@'
AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)@(\w)', re.UNICODE)
UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)@(\w)', re.UNICODE)

def get_sentences(text):
    """Sentence generator from provided text. Sentence pattern set
    in :const:`RE_SENTENCE`.

    Parameters
    ----------
    text : str
        Input text.

    Yields
    ------
    str
        Single sentence extracted from text.

    Example
    -------
    >>> text = "Does this text contains two sentences? Yes, it does."
    >>> for sentence in get_sentences(text):
    >>>     print(sentence)
    Does this text contains two sentences?
    Yes, it does.

    """
    for match in RE_SENTENCE.finditer(text):
        yield match.group()

def replace_with_separator(text, separator, regexs):
    """Get text with replaced separator if provided regular expressions were matched.

    Parameters
    ----------
    text : str
        Input text.
    separator : str
        The separator between words to be replaced.
    regexs : list of `_sre.SRE_Pattern`
        Regular expressions used in processing text.

    Returns
    -------
    str
        Text with replaced separators.

    """
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result

def replace_abbreviations(text):
    """Replace blank space to '@' separator after abbreviation and next word.

    Parameters
    ----------
    text : str
        Input sentence.

    Returns
    -------
    str
        Sentence with changed separator.

    Example
    -------
    >>> replace_abbreviations("God bless you, please, Mrs. Robinson")
    God bless you, please, Mrs.@Robinson

    """
    return replace_with_separator(text, SEPARATOR, [AB_SENIOR, AB_ACRONYM])

def undo_replacement(sentence):
    """Replace `@` separator back to blank space after each abbreviation.

    Parameters
    ----------
    sentence : str
        Input sentence.

    Returns
    -------
    str
        Sentence with changed separator.

    Example
    -------
    >>> undo_replacement("God bless you, please, Mrs.@Robinson")
    God bless you, please, Mrs. Robinson

    """
    return replace_with_separator(sentence, r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])


def split_sentences(text):
    """Split and get list of sentences from given text. It preserves abbreviations set in
    :const:`~gensim.summarization.textcleaner.AB_SENIOR` and :const:`~gensim.summarization.textcleaner.AB_ACRONYM`.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list of str
        Sentences of given text.

    Example
    -------
    >>> from gensim.summarization.textcleaner import split_sentences
    >>> text = '''Beautiful is better than ugly.
    ... Explicit is better than implicit. Simple is better than complex.'''
    >>> split_sentences(text)
    ['Beautiful is better than ugly.',
    'Explicit is better than implicit.',
    'Simple is better than complex.']

    """
    processed = replace_abbreviations(text)
    return [undo_replacement(sentence) for sentence in get_sentences(processed)]
