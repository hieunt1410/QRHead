"""
Dataset module for fine-tuning Qwen 2.5 / Qwen 3 on retrieval task.

The task is: given a query and top-k retrieved articles, the model should output
the gold/relevant article.

Qwen 2.5: Uses custom prompt format (no chat template)
Qwen 3: Uses chat template with conversations format
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


def format_prompt_text(
    query: str,
    docs: List[Dict[str, str]],
    gold_doc_id: str,
    separator: str = "\n",
    include_doc_index: bool = True,
) -> str:
    """
    Format as a single text string (for Qwen 2.5 without chat template).

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

    return prompt + "\n" + "The answer is: " + target


def format_conversations(
    query: str,
    docs: List[Dict[str, str]],
    gold_doc_id: str,
    separator: str = "\n",
    include_doc_index: bool = True,
) -> List[Dict[str, str]]:
    """
    Format as conversations (for Qwen 3 with chat template).

    Args:
        query: The user query
        docs: List of retrieved documents
        gold_doc_id: The gold document ID
        separator: Separator between documents
        include_doc_index: Whether to include document ID in output

    Returns:
        List of conversation messages with role and content
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


def load_examples_as_dataset(
    file_path: str,
    query_key: str = "query",
    docs_key: str = "docs",
    gold_idx_key: str = "gold_doc_idx",
    model_base_class: str = "Qwen2.5-7B-Instruct",
    include_doc_index: bool = True,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    enable_thinking: bool = False,  # Qwen 3 specific: enable thinking mode for reasoning
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
        model_base_class: Base model class (used to detect Qwen 3 for chat template)
        include_doc_index: Whether to include document ID in the target output
        tokenizer: Tokenizer for applying chat template (required for Qwen 3)
        enable_thinking: For Qwen 3, enable thinking/reasoning mode

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

    # Detect model type
    model_lower = model_base_class.lower()
    is_qwen3 = "qwen3" in model_lower or "qwen-3" in model_lower or "qwen 3" in model_lower

    if is_qwen3:
        # Qwen 3: Use chat template with conversations format
        if tokenizer is None:
            raise ValueError("tokenizer is required for Qwen 3")

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
                    conversation = format_conversations(
                        query=query,
                        docs=docs,
                        gold_doc_id=gold_id,
                        separator="\n",
                        include_doc_index=include_doc_index,
                    )
                    conversations.append(conversation)

            return {"conversations": conversations}

        # Create conversations column
        dataset = dataset.map(
            formatting_conversations_func,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Apply chat template to create text column
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }

        if enable_thinking:
            chat_template_kwargs["enable_thinking"] = True

        texts = tokenizer.apply_chat_template(
            list(dataset["conversations"]),
            **chat_template_kwargs,
        )

        dataset = Dataset.from_dict({"text": texts})

    else:
        # Qwen 2.5 and others: Use direct text formatting (no chat template)
        def formatting_prompts_func(examples_batch):
            queries = examples_batch[query_key]
            docs_list = examples_batch[docs_key]
            gold_ids_list = examples_batch[gold_idx_key]

            texts = []

            for query, docs, gold_ids in zip(queries, docs_list, gold_ids_list):
                # Handle both single string and list of strings
                if isinstance(gold_ids, str):
                    gold_ids = [gold_ids]
                for gold_id in gold_ids:
                    text = format_prompt_text(
                        query=query,
                        docs=docs,
                        gold_doc_id=gold_id,
                        separator="\n",
                        include_doc_index=include_doc_index,
                    )
                    texts.append(text)

            return {"text": texts}

        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            remove_columns=dataset.column_names,
        )

    return dataset
