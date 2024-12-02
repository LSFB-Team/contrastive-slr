import torch


def get_lcc_embedding(voc_size, background_size=4, hidden=1024, M=4):
    """Learnt Contrastive Concept (LCC) embedding can be see as an
    embedding of the class label of each signs. The shape of the embedding is
    (H, V', M) where H is the hidden size, V' is the size of the vocabulary (V) +
    an arbitrary number of "background" movement (hold, not signing, ...) and M is
    the number of variants for each gloss.

    params:
    voc_size: int, the size of the vocabulary
    background_size: int, the number of background movements
    hidden: int, the hidden size of the embedding
    M: int, the number of variants for each gloss
    """

    E = torch.nn.Parameter(
        torch.empty(hidden, voc_size + background_size, M)
    )  # H x V' x M
    torch.nn.init.xavier_uniform_(E, gain=1)
    E.requires_grad = True

    return E
