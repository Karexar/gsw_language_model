import csv
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def mask_tokens(
    tokens: torch.Tensor,
    tokeniser: PreTrainedTokenizer,
    prob: float = 0.15,
    ignore_label: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares tokens for masked language modelling (MLM)
    80% Mask, 10% random, 10% original
    """
    labels = tokens.clone()
    probability_matrix = torch.full_like(labels, prob, dtype=torch.float)
    special_tokens_mask = torch.tensor(
        [
            tokeniser.get_special_tokens_mask(label, already_has_special_tokens=True)
            for label in labels.tolist()
        ],
        dtype=torch.bool,
    )
    # Set the probabilities of the special tokens to 0.
    probability_matrix[special_tokens_mask] = 0.0
    mask_indices = torch.bernoulli(probability_matrix).to(torch.bool)
    # Non-masked labels are not used for the loss
    labels[~mask_indices] = ignore_label
    # 80% of the masked inputs are replaced with the Mask token.
    replace_indices = (
        torch.bernoulli(torch.full_like(labels, 0.8, dtype=torch.float)).to(torch.bool)
        & mask_indices
    )
    tokens[replace_indices] = tokeniser.convert_tokens_to_ids(tokeniser.mask_token)
    # 10% of the masked inputs (i.e. 50% of the remaining ones) are replaced with
    # a random word.
    random_indices = (
        torch.bernoulli(torch.full_like(labels, 0.5, dtype=torch.float)).to(torch.bool)
        & mask_indices
        & ~replace_indices
    )

    tokens[random_indices] = torch.randint_like(
        tokens[random_indices], 0, len(tokeniser)
    )

    # The remaining 10% are left as is
    return tokens, labels


class TextDataset(Dataset):
    """Dataset of text"""

    def __init__(
        self,
        path: str,
        tokeniser: PreTrainedTokenizer,
        use_special: bool = True,
        manual_special: bool = False,
        block_size: int = 512,
        name: Optional[str] = None,
    ):
        """
        Args:
            path (string): Path to fiel with the text
            tokeniser (PreTrainedTokenizer): Tokeniser used for the model.
            use_special (bool): Whether the tokeniser uses speical tokens.
                Mainly to avoid getting spammed by warnings.
                [Default: True]
            manual_special (bool): Whether to manually add special tokens to the start
                and end of the sequence rather than using the tokeniser's specific
                implementation. Needed when the XLNetTokenizer is used for the GPT-2
                model.
                [Default: False]
            block_size (int): Size of the blocks of text [Default: 512]
            name (string, optional): Name of the dataset
                [Default: Name of the ground truth file and its parent directory]
        """
        super(TextDataset, self).__init__()
        self.block_size = min(block_size, tokeniser.max_len_single_sentence)
        self.path = path
        self.tokeniser = tokeniser
        if name is None:
            filename = os.path.splitext(os.path.basename(path))[0]
            self.name = filename
        else:
            self.name = name

        if manual_special:
            assert (
                tokeniser.bos_token_id is not None
                and tokeniser.eos_token_id is not None
            ), (
                "tokeniser must have set a bos_token and eos_token "
                "when using manual_special=True"
            )

        with open(path, "r", encoding="utf8") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            tokenised_ids: List[int] = []
            if manual_special:
                tokenised_ids.append(tokeniser.eos_token_id)

            for line in reader:
                #print(line[0])
                encoded = tokeniser.encode(line[0], add_special_tokens=False)
                #print(max(tokeniser.added_tokens_encoder.values()))
                #print(encoded)
                #raise Exception
                tokenised_ids.extend(encoded)
                if manual_special:
                    tokenised_ids.append(tokeniser.eos_token_id)

        self.text_blocks: List[List[int]] = []
        # Group into blocks of text, discarding the last incomplete text.
        for i in range(0, len(tokenised_ids) - self.block_size + 1, self.block_size):
            token_block = tokenised_ids[i : i + self.block_size]
            if use_special:
                token_block = (
                    [tokeniser.bos_token_id] + token_block + [tokeniser.eos_token_id]
                    if manual_special
                    else tokeniser.build_inputs_with_special_tokens(token_block)
                )
            self.text_blocks.append(token_block)

    def __len__(self) -> int:
        return len(self.text_blocks)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.tensor(self.text_blocks[i])
