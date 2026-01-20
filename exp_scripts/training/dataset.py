"""
Dataset module for fine-tuning Qwen 2.5 7B on retrieval task.

The task is: given a query and top-k retrieved articles, the model should output
the gold/relevant article.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class RetrievalTrainingExample:
    """A single training example for retrieval fine-tuning."""

    query: str
    docs: List[
        Dict[str, str]
    ]  # List of {"paragraph_text": str, "title": Optional[str], "idx": str}
    gold_doc_id: str  # ID of the gold document (string like "doc1", "123", etc.)


class RetrievalFineTuningDataset(Dataset):
    """
    Dataset for fine-tuning LLM to retrieve the correct document.

    Each example consists of:
    - Input: query + top-k retrieved documents (formatted as prompt)
    - Output: the gold document content
    """

    def __init__(
        self,
        examples: List[RetrievalTrainingExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 32768,
        model_base_class: str = "Qwen2.5-7B-Instruct",
        include_doc_index: bool = True,
    ):
        """
        Args:
            examples: List of training examples
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            model_base_class: Base model class for prompt formatting
            include_doc_index: Whether to include document ID in the target output
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_base_class = model_base_class.lower()
        self.include_doc_index = include_doc_index

        # Set up prompt formatting based on model
        if "qwen" in self.model_base_class:
            self.user_start = "<|im_start|>user"
            self.assistant_start = "<|im_end|>\n<|im_start|>assistant"
            self.assistant_end = "<|im_end|>"
            self.separator = "\n\n"
        elif "llama" in self.model_base_class:
            self.user_start = "<|start_header_id|>user<|end_header_id|>"
            self.assistant_start = (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
            self.assistant_end = "<|eot_id|>"
            self.separator = " \n\n"
        else:
            # Default format
            self.user_start = "User:"
            self.assistant_start = "\nAssistant:"
            self.assistant_end = ""
            self.separator = "\n\n"

    def __len__(self) -> int:
        return len(self.examples)

    def _format_prompt(self, query: str, docs: List[Dict[str, str]]) -> str:
        """Format the input prompt with query and documents."""
        # Build the user prompt
        prompt = self.user_start + " Here are some paragraphs:"

        for doc in docs:
            paragraph_text = doc["paragraph_text"]
            doc_id = doc.get("idx", "")
            if doc.get("title"):
                paragraph_text = doc["title"] + "\n" + paragraph_text
            doc_str = f"[{doc_id}] {paragraph_text}"
            prompt += self.separator + doc_str

        instruction = "\n\nPlease find information that are relevant to the following query in the paragraphs above."
        prompt += instruction + self.separator + "Query: " + query
        prompt += self.assistant_start

        return prompt

    def _format_target(self, gold_doc_id: str, docs: List[Dict[str, str]]) -> str:
        """Format the target output (the gold document)."""
        # Find the gold document by its ID
        doc = next((d for d in docs if d.get("idx") == gold_doc_id), None)
        if doc is None:
            raise ValueError(f"Gold document ID '{gold_doc_id}' not found in docs")

        if self.include_doc_index:
            # Output with document ID
            target = f"[{gold_doc_id}] "
        else:
            target = ""

        paragraph_text = doc["paragraph_text"]
        if doc.get("title"):
            target += doc["title"] + "\n"
        target += paragraph_text

        return target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format the prompt and target
        prompt = self._format_prompt(example.query, example.docs)
        target = self._format_target(example.gold_doc_id, example.docs)

        # Combine for tokenization
        full_text = prompt + target + self.assistant_end

        # Get prompt length for masking
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens["input_ids"])

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels: only compute loss on the target portion
        labels = encodings["input_ids"].clone()
        labels[0, :prompt_length] = -100  # Ignore prompt tokens

        # Mask padding tokens (pad_token_id equals eos_token_id after we set it)
        pad_token_id = self.tokenizer.pad_token_id
        labels[0, :] = torch.where(
            encodings["input_ids"][0] == pad_token_id,
            torch.tensor(-100),
            labels[0, :]
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class RetrievalFineTuningDatasetWithIndex(Dataset):
    """
    Alternative dataset format where the model learns to output just the document ID.

    This can be easier for the model to learn and more efficient for inference.
    """

    def __init__(
        self,
        examples: List[RetrievalTrainingExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 32768,
        model_base_class: str = "Qwen2.5-7B-Instruct",
    ):
        """
        Args:
            examples: List of training examples
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            model_base_class: Base model class for prompt formatting
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_base_class = model_base_class.lower()

        # Set up prompt formatting based on model
        if "qwen" in self.model_base_class:
            self.user_start = "<|im_start|>user"
            self.assistant_start = "<|im_end|>\n<|im_start|>assistant"
            self.assistant_end = "<|im_end|>"
            self.separator = "\n\n"
        elif "llama" in self.model_base_class:
            self.user_start = "<|start_header_id|>user<|end_header_id|>"
            self.assistant_start = (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
            self.assistant_end = "<|eot_id|>"
            self.separator = " \n\n"
        else:
            self.user_start = "User:"
            self.assistant_start = "\nAssistant:"
            self.assistant_end = ""
            self.separator = "\n\n"

    def __len__(self) -> int:
        return len(self.examples)

    def _format_prompt(self, query: str, docs: List[Dict[str, str]]) -> str:
        """Format the input prompt with query and documents."""
        prompt = self.user_start + " Here are some paragraphs:"

        for doc in docs:
            paragraph_text = doc["paragraph_text"]
            doc_id = doc.get("idx", "")
            if doc.get("title"):
                paragraph_text = doc["title"] + "\n" + paragraph_text
            doc_str = f"[{doc_id}] {paragraph_text}"
            prompt += self.separator + doc_str

        instruction = "\n\nWhich of the above paragraphs is most relevant to the query? Respond with just the ID in brackets."
        prompt += instruction + self.separator + "Query: " + query
        prompt += self.assistant_start

        return prompt

    def _format_target(self, gold_doc_id: str) -> str:
        """Format the target (just the document ID)."""
        return f"[{gold_doc_id}]"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format the prompt and target
        prompt = self._format_prompt(example.query, example.docs)
        target = self._format_target(example.gold_doc_id)

        # Combine for tokenization
        full_text = prompt + target + self.assistant_end

        # Get prompt length for masking
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_length = len(prompt_tokens["input_ids"])

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels: only compute loss on the target portion
        labels = encodings["input_ids"].clone()
        labels[0, :prompt_length] = -100  # Ignore prompt tokens

        # Mask padding tokens
        pad_token_id = self.tokenizer.pad_token_id
        labels[0, :] = torch.where(
            encodings["input_ids"][0] == pad_token_id,
            torch.tensor(-100),
            labels[0, :]
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def load_examples(
    file_path: str,
    query_key: str = "query",
    docs_key: str = "docs",
    gold_idx_key: str = "gold_doc_idx",
) -> List[RetrievalTrainingExample]:
    """
    Load training examples from a JSON or JSONL file.

    Expected format per line (JSONL):
    {
        "query": "What is the capital of France?",
        "docs": [
            {"idx": "doc1", "paragraph_text": "...", "title": "..."},
            {"idx": "doc2", "paragraph_text": "...", "title": "..."},
            ...
        ],
        "gold_doc_idx": ["doc1"]  # or "doc1" for single gold
    }

    Args:
        file_path: Path to the JSON/JSONL file
        query_key: Key name for the query field
        docs_key: Key name for the documents field
        gold_idx_key: Key name for the gold document ID field

    Returns:
        List of RetrievalTrainingExample objects
    """
    import json

    examples = []
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                gold_ids = data[gold_idx_key]
                # Handle both single string and list of strings
                if isinstance(gold_ids, str):
                    gold_ids = [gold_ids]
                examples.extend(
                    [
                        RetrievalTrainingExample(
                            query=data[query_key],
                            docs=data[docs_key],
                            gold_doc_id=gold_id,
                        )
                        for gold_id in gold_ids
                    ]
                )
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for example in data:
                gold_ids = example[gold_idx_key]
                # Handle both single string and list of strings
                if isinstance(gold_ids, str):
                    gold_ids = [gold_ids]
                examples.extend(
                    [
                        RetrievalTrainingExample(
                            query=example[query_key],
                            docs=example[docs_key],
                            gold_doc_id=gold_id,
                        )
                        for gold_id in gold_ids
                    ]
                )
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return examples
