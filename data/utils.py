import typing as tp
import re

RE_PARAGRAPH = re.compile(r"^\s*$", re.MULTILINE)

# helper functions
def get_paragraphs(desc: str) -> tp.List[str]:
    """Return paragraphs in a description
    """
    desc = desc.strip()
    return [_.strip() for _ in RE_PARAGRAPH.split(desc)]

def cleanup_description(
    desc: str,
    remove_empty_lines: bool = True,
    remove_url: bool = True,
    keep_only_first_paragraph: bool = False,
    filter_with_sentencebert: bool = True,
    title: str = None,
    similarity_threshold: float = .5,
    sentence_bert_model: tp.Union["SentenceTransformer", str] = 'sentence-transformers/all-MiniLM-L6-v2'
) -> str:
    """Cleanup description of video

    1. removes url
    2. remove paragraphs that do not have high similarity with title via sentencebert
    """

    desc = desc.strip()
    if remove_url:
        desc = re.sub(r'https?://\S+', "", desc)

    if filter_with_sentencebert:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        assert title is not None, "Title cannot be None if filter with sentence_bert"
        if isinstance(sentence_bert_model, str):
            sentence_bert_model = SentenceTransformer(sentence_bert_model)
        def filter_by_similarity(ttl: str, desc: str, threshold: float) -> bool:
            """Compute SentenceBert similarity between title and description
            """
            ttl_embedding = sentence_bert_model.encode(ttl)
            desc_embeddings = [sentence_bert_model.encode(_) for _ in desc.split('\n')]
            sim = max([
                cosine_similarity(ttl_embedding[None,:], _desc_embed[None,:]).item()
                for _desc_embed in desc_embeddings
            ])
            if sim < threshold:
                return False
            return True
        desc = "\n".join(list(filter(
            lambda paragraph: filter_by_similarity(title, paragraph, similarity_threshold),
            get_paragraphs(desc)
        )))

    if keep_only_first_paragraph:
        # the following line does 3 things
        # 1. It compiles multiline regex pattern that matches empty lines
        # 2. It splits `desc` at each empty line
        # 3. It keeps the first match and strips the result
        desc = get_paragraphs(desc)[0]

    if remove_empty_lines:
        desc = "\n".join(get_paragraphs(desc))

    return desc
