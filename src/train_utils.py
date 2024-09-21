from .transformer import subsequent_mask, Transformer, Config, DecoderOnlyTransformer

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import evaluate
import pandas as pd

from typing import Callable, Optional, List, Tuple, Union
from tqdm.auto import tqdm

T = torch.Tensor
M = nn.Module


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, source: T, target: Optional[T] = None, pad: int = 2):
        # source shape: (batch_size, seq_len)
        # target shape: (batch_size, seq_len)
        self.source = source
        self.source_mask = (source != pad).unsqueeze(-2)
        if target is not None:
            # the model predicts the next word using the previous words and hence the last token is not required
            # the last token is also removed to align the target shape with the output
            self.target = target[:, :-1]
            # the model predicts the next word, so the target is shifted by one
            # use target_y while computing loss
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.ntokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(target: T, pad: int) -> T:
        "Create a mask to hide padding and future words."
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = target_mask & subsequent_mask(target.size(-1)).type_as(
            target_mask.data
        )
        return target_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    def __repr__(self) -> str:
        return f"TrainState(step={self.step}, accum_step={self.accum_step}, samples={self.samples}, tokens={self.tokens})"

    def __str__(self) -> str:
        string = f"""Steps: {self.step}
Accumulation Steps: {self.accum_step}
Samples: {self.samples}
Tokens: {self.tokens}"""
        return string


def bce_crit(output: T, target: T, tokens: Optional[int] = None) -> T:
    out = output.contiguous().view(-1, output.size(-1))
    tar = target.contiguous().view(-1)
    if tokens is None:
        tokens = 1  # assume a single token if not provided
    return torch.nn.functional.cross_entropy(out, tar, ignore_index=0) / tokens


class TransformerTrainer:
    """Train a transformer model"""

    def __init__(
        self,
        model: Union[Transformer, DecoderOnlyTransformer],
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[T, T, Optional[int]], T],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        source_tokenizer: Optional[Tokenizer] = None,
        target_tokenizer: Optional[Tokenizer] = None,
        wandb=None,
        wandb_log_freq: int = 100,
        max_len: int = 32,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.total_batches = 0
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.wandb = wandb
        self.wandb_log_freq = wandb_log_freq
        self.max_len = max_len
        self.decoder_only = True if isinstance(model, DecoderOnlyTransformer) else False

    def run_epoch(
        self,
        data_iter: List[Batch],
        mode: str = "train",
        accum_iter: int = 1,
        train_state: TrainState = TrainState(),
        log_freq: int = 40,
        max_batches: Optional[int] = None,
    ):
        """Train a single epoch"""
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        for i, batch in enumerate(tqdm(data_iter)):
            if self.decoder_only:
                # source and target are same but the target is shifted by one
                out = self.model.forward(batch.target, batch.target_mask)
            else:
                out = self.model.forward(
                    batch.source, batch.target, batch.source_mask, batch.target_mask
                )
            out_g = self.model.generate(out)
            loss_node: T = self.criterion(out_g, batch.target_y, batch.ntokens)
            loss = loss_node.item()
            if mode == "train":
                loss_node.backward()
                train_state.step += 1
                train_state.samples += batch.source.shape[0]
                train_state.tokens += batch.ntokens

                if i % accum_iter == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    n_accum += 1
                    train_state.accum_step += 1
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % log_freq == 1 and mode == "train":
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                msg = f"Batch Step: {i}, Loss*100: {100*(loss / batch.ntokens ):6f}, Tokens / Sec: {tokens / elapsed :4f}, Learning Rate: {lr}"
                print(msg)
                start = time.time()
                tokens = 0
            del loss
            del loss_node
            self.total_batches += 1
            if max_batches is not None and self.total_batches >= max_batches:
                break

            if self.total_batches % self.wandb_log_freq == 1 and mode == "train":
                self.log_sample_to_wandb(wandb=self.wandb)
                # will log to wandb if wandb is not None
                # otherwise will print the log
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "total_batches": self.total_batches,
                            "total_tokens": total_tokens,
                            "loss": total_loss / total_tokens,
                        }
                    )

        return total_loss / total_tokens, train_state

    def train(
        self,
        data_loader: List[Batch],
        num_epochs: int,
        eval_loader: Optional[List[Batch]] = None,
        log_freq: int = 40,
        max_batches: Optional[int] = None,
    ) -> None:
        self.total_batches = 0
        self.model.train()
        train_states = []
        losses = []
        eval_states = []
        eval_losses = []

        for epoch in range(num_epochs):
            train_state = TrainState()
            loss, train_state = self.run_epoch(
                data_loader,
                mode="train",
                train_state=train_state,
                log_freq=log_freq,
                max_batches=max_batches,
            )
            losses.append(loss)
            train_states.append(train_state)

            if eval_loader is not None:
                eval_state = TrainState()
                self.model.eval()
                with torch.no_grad():
                    eval_loss, eval_state = self.run_epoch(
                        eval_loader,
                        mode="eval",
                        train_state=eval_state,
                        log_freq=log_freq,
                    )
                eval_losses.append(eval_loss)
                eval_states.append(eval_state)
            progress_str = f"Epoch: {epoch}, Loss: {loss:4f}"
            if eval_loader is not None:
                progress_str += f", Eval Loss: {eval_loss:4f}"
            print(progress_str)

            if max_batches is not None and self.total_batches >= max_batches:
                break

        if eval_loader is not None:
            return losses, train_states, eval_losses, eval_states
        return losses, train_states

    def log_sample_to_wandb(self, wandb=None):
        pass  # Must be implemented in the child class


def greedy_decode(
    text: str,
    model: M,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_len: int = 32,
    task: str = "translation",
    prefix: str = "",
    device: Optional[str] = None,
    decoder_only_for_generation: bool = True,
):
    """A greedy decoder for the transformer model

    Parameters
    ----------
    text : str
        The input text to be translated or generated
    model : M
        The transformer model
    source_tokenizer : Tokenizer
        The source tokenizer
    target_tokenizer : Tokenizer
        The target tokenizer
    max_len : int, optional
        The maximum length of the output text, by default 32
    task : str, optional
        The task to perform. Either 'translation' or 'generation', by default "translation"
    prefix : str, optional
        The prefix to add to the input text, by default ""
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task not in ["translation", "generation"]:
        msg = f"Task must be either 'translation' or 'generation', but got {task}"
        raise ValueError(msg)
    text = f"[CLS]{prefix}{text}"
    if task == "translation" or not decoder_only_for_generation:
        text = f"{text}[SEP]"
    text = [text]

    source_pad_token = source_tokenizer.token_to_id("[PAD]")
    end_token = target_tokenizer.token_to_id("[SEP]")

    source = source_tokenizer.encode_batch(text)
    ids = [enc.ids for enc in source][0]
    attension_masks = [enc.attention_mask for enc in source][0]
    ids = [[i for i, m in zip(ids, attension_masks) if m == 1]]
    source = torch.tensor(ids).to(device)
    source_length = source.shape[1]

    if source_length > max_len and task == "generation":
        msg = f"Input text is too long. Max length is {max_len}, but the input text is {source_length}"
        raise ValueError(msg)

    source_mask = (source != source_pad_token).unsqueeze(-2)  # should be all True

    if task == "generation":
        ys = source  # start with the source
    else:
        ys = (
            torch.zeros(1, 1)
            .fill_(target_tokenizer.token_to_id("[CLS]"))
            .type_as(source.data)
        )

    if task == "translation" or not decoder_only_for_generation:
        memory = model.encode(source, source_mask)

        out_f = lambda s, s_m: model.decode(memory, s, source_mask, s_m)
    else:
        out_f = lambda s, s_m: model.decode(s, s_m)  # decoder only for generation

    for i in range(max_len - 1):
        out = out_f(ys, subsequent_mask(ys.size(1)).type_as(source.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(source.data).fill_(next_word)], dim=1
        )
        if next_word == end_token:
            # if the next word is the end token, then break
            break

    rank = len(ys.shape)
    if rank == 1:
        ys = ys.unsqueeze(0)

    return target_tokenizer.decode_batch(ys.tolist()), ys


class TransformerTrainerForTranslation(TransformerTrainer):
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[T, T, Optional[int]], T],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        source_tokenizer: Optional[Tokenizer] = None,
        target_tokenizer: Optional[Tokenizer] = None,
        wandb=None,
        wandb_log_freq: int = 100,
        max_len: int = 32,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            scheduler,
            device,
            source_tokenizer,
            target_tokenizer,
            wandb,
            wandb_log_freq,
            max_len,
        )

    def translate(self, text: str, **kwargs):
        return greedy_decode(
            text=text,
            model=self.model,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            max_len=self.max_len,
            task="translation",
            prefix="",
            device=self.device,
        )

    def log_sample_to_wandb(self, wandb=None):
        samples = [
            "I am a stick",
            "Journey before destination",
            "Long live the transformer",
            "There is always another secret",
            "The most important step a person can take is always the next one",
            "The purpose of a storyteller is not to tell you how to think",
            "but to give you questions to think upon",
            "The only time a story is dead is when it lives only in the mind of its creator",
        ]
        ground_truths = [
            "Je suis un baton",
            "Le voyage avant la destination",
            "Vive le transformateur",
            "Il y a toujours un autre secret",
            "La démarche la plus importante qu'une personne puisse faire est toujours la prochaine",
            "Le but d'un conteur n'est pas de vous dire comment penser",
            "mais pour vous donner des questions à réfléchir",
            "Le seul moment où une histoire est morte est lorsqu'elle vit uniquement dans l'esprit de son créateur",
        ]
        translations = []
        translations_ids = []
        for sample in tqdm(samples, desc="Translating...", disable=True):
            translation, translation_ids = self.translate(sample)
            translations.append(translation[0])
            translations_ids.append(translation_ids[0].tolist())
        log_dict = {
            "translations": translations,
            "samples": samples,
            "ground_truths": ground_truths,
            #             "tokens": translations_ids,
        }
        if wandb is None:
            from pprint import pprint

            pprint(log_dict)
        else:
            try:
                bleu = evaluate.load("bleu")
                score = bleu.compute(predictions=translations, references=ground_truths)
                wandb.log({"bleu_score": score["bleu"]})
            except:  # possibly ZeroDevisionError when target length is zero for initial batches
                pass
            text_table = wandb.Table(
                columns=["ground_truth", "translation", "source", "tokens"]
            )
            for i in range(len(samples)):
                text_table.add_data(
                    ground_truths[i], translations[i], samples[i], translations_ids[i]
                )
            wandb.log({"translations": text_table})


class TransformerTrainerForGeneration(TransformerTrainer):
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[T, T, Optional[int]], T],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
        wandb=None,
        wandb_log_freq: int = 100,
        max_len: int = 32,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            scheduler,
            device,
            tokenizer,
            tokenizer,
            wandb,
            wandb_log_freq,
            max_len,
        )

    def generate(self, text: str, **kwargs):
        return greedy_decode(
            text=text,
            model=self.model,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            max_len=self.max_len,
            task="generation",
            prefix="",
            device=self.device,
        )

    def log_sample_to_wandb(self, wandb=None):
        sample_taylor_swift_lyrics = [
            "salt air and the rust on your door",
            "no other sadness in the world would do"
            "make sure nobody sees you leave hood over your head keep your eyes down",
            "there i was again tonight forcing laughter faking smiles same old tired lonely place",
            "i said remember this moment in the back of my mind",
            "hey stephen i know looks can be deceiving but i know i saw a light in you",
            "we could leave the christmas lights up 'til january",
            "life was a willow and it bent right to your wind",
        ]

        generations = []
        generation_ids = []
        for sample in tqdm(
            sample_taylor_swift_lyrics, desc="Generating...", disable=True
        ):
            generation, generation_id = self.generate(sample)
            generations.append(generation[0])
            generation_ids.append(generation_id[0].tolist())
        log_dict = {
            "generations": generations,
            # "samples": sample_taylor_swift_lyrics,
        }
        if wandb is None:
            from pprint import pprint

            pprint(log_dict)
        else:
            text_table = wandb.Table(columns=["samples", "generations", "tokens"])
            for i in range(len(sample_taylor_swift_lyrics)):
                text_table.add_data(
                    sample_taylor_swift_lyrics[i],
                    generations[i],
                    generation_ids[i],
                )
            wandb.log({"generation": text_table})


class TranslationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path: str,
        source_tokenizer_path: str,
        target_tokenizer_path: str,
        source_column: str,
        target_column: str,
        max_len: int = 32,
        device: Optional[str] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.df = pd.read_csv(dataset_path)
        self.target_tokenizer: Tokenizer = Tokenizer.from_file(target_tokenizer_path)
        self.source_tokenizer: Tokenizer = Tokenizer.from_file(source_tokenizer_path)
        self.target_tokenizer.enable_truncation(max_length=max_len)
        self.source_tokenizer.enable_truncation(max_length=max_len)
        self.target_tokenizer.enable_padding(
            pad_id=self.target_tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=max_len,
        )
        self.source_tokenizer.enable_padding(
            pad_id=self.source_tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=max_len,
        )
        self.max_len = max_len
        self.source_column = source_column
        self.target_column = target_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[self.source_column][idx], self.df[self.target_column][idx]

    def collate_fn(self, batch) -> Batch:
        src, tgt = zip(*batch)
        # add start and end tokens
        src = ["[CLS] " + s + " [SEP]" for s in src]
        tgt = ["[CLS] " + t + " [SEP]" for t in tgt]
        src_encodings = self.source_tokenizer.encode_batch(src)
        tgt_encodings = self.target_tokenizer.encode_batch(tgt)
        src_encodings = torch.tensor([enc.ids for enc in src_encodings]).to(self.device)
        tgt_encodings = torch.tensor([enc.ids for enc in tgt_encodings]).to(self.device)

        return Batch(
            src_encodings,
            tgt_encodings,
            pad=self.source_tokenizer.token_to_id("[PAD]"),
        )

    def get_dataloader(self, batch_size: int, shuffle: bool = False):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn
        )

    def decode(self, tokens: T, tokenizer: Tokenizer) -> str:
        rank = len(tokens.shape)
        if rank == 1:
            tokens = tokens.unsqueeze(0)

        return tokenizer.decode_batch(tokens.tolist())

    def decode_source(self, tokens: T) -> str:
        return self.decode(tokens, self.source_tokenizer)

    def decode_target(self, tokens: T) -> str:
        return self.decode(tokens, self.target_tokenizer)

    def decode_batch(self, batch: Batch) -> Tuple[List[str], List[str]]:
        return (
            [self.decode_source(enc) for enc in batch.source],
            [self.decode_target(enc) for enc in batch.target],
        )


class GenerationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_path: str,
        tokenizer_path: str,
        text_column: str,
        max_len: int = 64,
        device: Optional[str] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.df = pd.read_csv(dataset_path)

        self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
            length=max_len,
        )
        self.max_len = max_len
        self.text_column = text_column
        self.lyrics = self.split()

    def split(self):
        import re

        lyrics: List[str] = self.df[self.text_column].tolist()
        lyrics = " ".join(lyrics)
        # remove any character that are not numbers, alphabets, or punctuations
        lyrics = re.sub(r"[^a-zA-Z0-9.,!?]+", " ", lyrics)
        lyrics_list = lyrics.split(" ")

        max_len = int(self.max_len * 0.66)  # one word may have multiple tokens
        lyrics_final = [
            lyrics_list[i : i + max_len] for i in range(0, len(lyrics_list), max_len)
        ]

        lyrics_final = [" ".join(lyric) for lyric in lyrics_final]
        return lyrics_final

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        return self.lyrics[idx]

    def collate_fn(self, batch) -> Batch:
        src = ["[CLS] " + s + " [SEP]" for s in batch]
        src_encodings = self.tokenizer.encode_batch(src)
        src_encodings = torch.tensor([enc.ids for enc in src_encodings]).to(self.device)

        return Batch(
            source=src_encodings,
            target=src_encodings,
            pad=self.tokenizer.token_to_id("[PAD]"),
        )

    def get_dataloader(self, batch_size: int, shuffle: bool = False):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn
        )

    def decode(self, tokens: T, tokenizer: Tokenizer) -> str:
        rank = len(tokens.shape)
        if rank == 1:
            tokens = tokens.unsqueeze(0)

        return tokenizer.decode_batch(tokens.tolist())

    def decode_text(self, tokens: T) -> str:
        return self.decode(tokens, self.tokenizer)

    def decode_batch(self, batch: Batch) -> List[str]:
        return [self.decode_text(enc) for enc in batch.source]


class TransformerTrainerConfig(Config):
    ALLOWED_KEYS = [
        "model",
        "optimizer",
        "criterion",
        "scheduler",
        "device",
        "source_tokenizer",
        "target_tokenizer",
        "wandb",
        "wandb_log_freq",
        "max_len",
    ]
    ConfigFor = TransformerTrainer


class TransformerTrainerForGenerationConfig(Config):
    ALLOWED_KEYS = [
        "model",
        "optimizer",
        "criterion",
        "scheduler",
        "device",
        "tokenizer",
        "wandb",
        "wandb_log_freq",
        "max_len",
    ]
    ConfigFor = TransformerTrainerForGeneration


class TransformerTrainerForTranslationConfig(Config):
    ALLOWED_KEYS = [
        "model",
        "optimizer",
        "criterion",
        "scheduler",
        "device",
        "source_tokenizer",
        "target_tokenizer",
        "wandb",
        "wandb_log_freq",
        "max_len",
    ]
    ConfigFor = TransformerTrainerForTranslation


class TranslationDatasetConfig(Config):
    ALLOWED_KEYS = [
        "dataset_path",
        "source_tokenizer_path",
        "target_tokenizer_path",
        "source_column",
        "target_column",
        "max_len",
        "device",
    ]
    ConfigFor = TranslationDataset


class GenerationDatasetConfig(Config):
    ALLOWED_KEYS = [
        "dataset_path",
        "tokenizer_path",
        "text_column",
        "max_len",
        "device",
    ]
    ConfigFor = GenerationDataset
