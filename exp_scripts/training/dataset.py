"""
Dataset module for fine-tuning Qwen 2.5 7B on retrieval task.

The task is: given a query and top-k retrieved articles, the model should output
the gold/relevant article.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


@dataclass
class RetrievalTrainingExample:
    """A single training example for retrieval fine-tuning."""

    query: str
    docs: List[
        Dict[str, str]
    ]  # List of {"paragraph_text": str, "title": Optional[str], "idx": str}
    gold_doc_id: str  # ID of the gold document (string like "doc1", "123", etc.)


def formatting_conversations(
    query: str,
    docs: List[Dict[str, str]],
    gold_doc_id: str,
    separator: str = "\n",
    include_doc_index: bool = True,
) -> List[Dict[str, str]]:
    """
    Format the prompt for Qwen model.

    Args:
        query: The user query
        docs: List of retrieved documents
        gold_doc_id: The gold document ID
        separator: Separator between documents
        include_doc_index: Whether to include document ID in output

    Returns:
        Formatted text string
    """
    # Build the user prompt
    prompt = "Here are some paragraphs:\n\n### Paragraphs:"

    for doc in docs:
        paragraph_text = doc["paragraph_text"]
        doc_id = doc.get("idx", "")
        if doc.get("title"):
            paragraph_text = doc["title"] + "\n" + paragraph_text
        doc_str = f"[{doc_id}] {paragraph_text}"
        prompt += separator + doc_str

    instruction = "\n\nPlease find information that are relevant to the following query in the paragraphs above."
    prompt += instruction + separator + "Query: " + query

    # Add the gold document ID as the target
    if include_doc_index:
        target = f"[{gold_doc_id}]"
    else:
        # Find the gold document content
        doc = next((d for d in docs if d.get("idx") == gold_doc_id), None)
        if doc is None:
            raise ValueError(f"Gold document ID '{gold_doc_id}' not found in docs")
        target = f"[{gold_doc_id}] " + doc["paragraph_text"]

    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]

    return conversation


def formatting_prompts(
    query: str,
    docs: List[Dict[str, str]],
    gold_doc_id: str,
    separator: str = "\n",
    include_doc_index: bool = True,
) -> str:
    """
    Format as a single text string (for non-chat models).

    Args:
        query: The user query
        docs: List of retrieved documents
        gold_doc_id: The gold document ID
        separator: Separator between documents
        include_doc_index: Whether to include document ID in output

    Returns:
        Formatted text string
    """
    # Build the user prompt
    prompt = "Here are some paragraphs:\n\n### Paragraphs:"

    for doc in docs:
        paragraph_text = doc["paragraph_text"]
        doc_id = doc.get("idx", "")
        if doc.get("title"):
            paragraph_text = doc["title"] + "\n" + paragraph_text
        doc_str = f"[{doc_id}] {paragraph_text}"
        prompt += separator + doc_str

    instruction = "\n\nPlease find information that are relevant to the following query in the paragraphs above."
    prompt += instruction + separator + "Query: " + query

    # Add the gold document ID as the target
    if include_doc_index:
        target = f"[{gold_doc_id}]"
    else:
        # Find the gold document content
        doc = next((d for d in docs if d.get("idx") == gold_doc_id), None)
        if doc is None:
            raise ValueError(f"Gold document ID '{gold_doc_id}' not found in docs")
        target = f"[{gold_doc_id}] " + doc["paragraph_text"]

    return prompt + "\n" + target


def load_examples_as_dataset(
    file_path: str,
    query_key: str = "query",
    docs_key: str = "docs",
    gold_idx_key: str = "gold_doc_idx",
    model_base_class: str = "Qwen2.5-7B-Instruct",
    include_doc_index: bool = True,
    tokenizer: Optional[PreTrainedTokenizer] = None,
):
    """
    Load training examples from a JSON or JSONL file and return a datasets.Dataset.

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
        model_base_class: Base model class for prompt formatting
        include_doc_index: Whether to include document ID in the target output

    Returns:
        datasets.Dataset with formatted "text" column
    """
    import json

    from datasets import Dataset

    examples = []
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data)
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Convert to datasets.Dataset
    dataset = Dataset.from_list(examples)

    # Define the formatting function based on model type
    def formatting_prompts_func(examples_batch):
        queries = examples_batch[query_key]
        docs_list = examples_batch[docs_key]
        gold_ids_list = examples_batch[gold_idx_key]

        texts = []
        model_base = model_base_class.lower()

        # Set up separator based on model
        for query, docs, gold_ids in zip(queries, docs_list, gold_ids_list):
            # Handle both single string and list of strings
            if isinstance(gold_ids, str):
                gold_ids = [gold_ids]

            # Create a training example for each gold ID
            for gold_id in gold_ids:
                text = formatting_prompts(
                    query=query,
                    docs=docs,
                    gold_doc_id=gold_id,
                    separator="\n",
                    include_doc_index=include_doc_index,
                )
                texts.append(text)

        return {"text": texts}

    def formatting_conversations_func(examples_batch):
        queries = examples_batch[query_key]
        docs_list = examples_batch[docs_key]
        gold_ids_list = examples_batch[gold_idx_key]

        conversations = []

        for query, docs, gold_ids in zip(queries, docs_list, gold_ids_list):
            # Handle both single string and list of strings
            if isinstance(gold_ids, str):
                gold_ids = [gold_ids]
            for gold_id in gold_ids:
                conversation = formatting_conversations(
                    query=query,
                    docs=docs,
                    gold_doc_id=gold_id,
                    separator="\n",
                    include_doc_index=include_doc_index,
                )
                conversations.append(conversation)

        return {"conversations": conversations}

    # First, create conversations column
    dataset = dataset.map(
        formatting_conversations_func,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Check if this is a chat model (Qwen 2.5, Qwen 3, Llama 3, etc.)
    model_base = model_base_class.lower()
    is_chat_model = (
        "qwen" in model_base and ("2" in model_base or "3" in model_base)
    ) or "llama-3" in model_base or "instruct" in model_base

    if is_chat_model:
        # For chat models, return dataset with "conversations" column
        # SFTTrainer will apply chat_template automatically
        return dataset
    else:
        # For non-chat models, apply chat template manually to create "text" column
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required for non-chat models to create text column"
            )

        def apply_chat_template_func(examples_batch):
            texts = tokenizer.apply_chat_template(
                examples_batch["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": texts}

        dataset = dataset.map(
            apply_chat_template_func,
            batched=True,
            remove_columns=["conversations"],
        )
        return dataset
