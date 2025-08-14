def split_into_passages(text, chunk_size=500, overlap=50):
    """
    Split a text into passages of specified size with overlap.

    Args:
        text (str): The input text to split.
        chunk_size (int): Number of words per passage.
        overlap (int): Number of overlapping words between passages.

    Returns:
        list: A list of passage strings.
    """
    words = text.split()
    passages = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        passage = " ".join(words[start:end])
        passages.append(passage)
        start = end - overlap
    return passages
