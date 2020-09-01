import argparse
import csv
import json
import os
import tempfile

import torch
from halo import Halo
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from subword_nmt.learn_bpe import learn_bpe

seed = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        type=str,
        help="Path to TSV file of the sentences",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_dir",
        type=str,
        help="Output directory [Default: data/<basename-of-input-file>/]",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(seed),
    )

    return parser.parse_args()


def main():
    options = parse_args()
    torch.manual_seed(options.seed)
    basename = os.path.splitext(os.path.basename(options.input))[0]
    out_dir = options.out_dir or "data/{}/".format(basename)
    spinner = Halo(spinner="dots", placement="right")

    with open(options.input, "r", encoding="utf8") as fd:
        reader = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        lines = [[line[0]] for line in reader]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_full = os.path.join(out_dir, "{}.tsv".format(basename))
    with open(output_full, "w", encoding="utf8") as fd:
        writer = csv.writer(fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        writer.writerows(lines)

    vocab_size = 32000
    spiece_out = os.path.join(out_dir, "spiece")
    spiece_args = (
        "--input={} "
        "--model_prefix={} "
        "--vocab_size={} "
        "--character_coverage=1.0"
    ).format(output_full, spiece_out, vocab_size)
    SentencePieceTrainer.Train(spiece_args)
    # Load the generated vocabulary
    with open("{}.vocab".format(spiece_out), "r", encoding="utf8") as fd:
        reader = csv.reader(
            fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
        )
        vocab = [line[0] for line in reader]
    # Remove the special tokens <unk>, <s>, </s>
    vocab = vocab[3:]

    # Convert to BERT style
    bert_vocab = [
        v[1:] if v.startswith("▁") else "##{}".format(v) for v in vocab if v != "▁"
    ]
    # Add BERT's special tokens to the beginning
    bert_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + bert_vocab
    # Fill up with unused tokens
    pad_size = vocab_size - len(bert_vocab)
    bert_vocab += ["unused{}".format(i) for i in range(pad_size)]
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf8") as fd:
        writer = csv.writer(
            fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
        )
        writer.writerows([[b] for b in bert_vocab])

    # Convert to GPT-2 style
    # Unfortunately it's slow and tedious.
    spinner.start(text="Generating BPE vocabulary")
    gpt2_vocab = ["Ġ{}".format(v[1:]) if v.startswith("▁") else v for v in vocab]
    # Add the GPT-2 special token to the end
    gpt2_vocab.append("<|endoftext|>")
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf8") as fd:
        json.dump({v: i for i, v in enumerate(gpt2_vocab)}, fd, ensure_ascii=False)
    spiece_processor = SentencePieceProcessor()
    spiece_processor.Load("{}.model".format(spiece_out))
    # Encode the whole text
    encoded = [
        [" ".join(spiece_processor.EncodeAsPieces(line[0])).replace("▁", "Ġ")]
        for line in lines
    ]
    tmp_encoded_fd, tmp_encoded_path = tempfile.mkstemp()
    tmp_bpe_fd, tmp_bpe_path = tempfile.mkstemp()
    try:
        # Write the encoded text to a temporary file.
        with os.fdopen(tmp_encoded_fd, "w", encoding="utf8") as fd:
            writer = csv.writer(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            writer.writerows(encoded)
        learn_bpe(
            open(tmp_encoded_path, "r", encoding="utf8"),
            open(tmp_bpe_path, "w", encoding="utf8"),
            num_symbols=vocab_size,
        )
        with open(tmp_bpe_path, "r", encoding="utf8") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            seen = set()
            merges = []
            for line in reader:
                # Get rid of the </w> tokens
                line = line[0].replace("</w>", "")
                # Remove duplicates (due to </w> tokens)
                if line not in seen:
                    seen.add(line)
                    merges.append([line])
        with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf8") as fd:
            writer = csv.writer(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            writer.writerows(merges)
    finally:
        os.remove(tmp_encoded_path)
        os.remove(tmp_bpe_path)
    spinner.stop()


if __name__ == "__main__":
    main()
