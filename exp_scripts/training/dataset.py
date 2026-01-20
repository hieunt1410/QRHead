"""
Dataset module for fine-tuning Qwen 2.5 7B on retrieval task.

The task is: given a query and top-k retrieved articles, the model should output
the gold/relevant article.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RetrievalTrainingExample:
    """A single training example for retrieval fine-tuning."""

    query: str
    docs: List[
        Dict[str, str]
    ]  # List of {"paragraph_text": str, "title": Optional[str], "idx": str}
    gold_doc_id: str  # ID of the gold document (string like "doc1", "123", etc.)


def format_qwen_prompt(
    query: str,
    docs: List[Dict[str, str]],
    gold_doc_id: str,
    separator: str = "\n",
    include_doc_index: bool = True,
) -> str:
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

    # Add the assistant start and the expected response
    prompt += "<|im_end|>\n<|im_start|>assistant"

    # Add the gold document ID as the target
    if include_doc_index:
        target = f"[{gold_doc_id}]"
    else:
        # Find the gold document content
        doc = next((d for d in docs if d.get("idx") == gold_doc_id), None)
        if doc is None:
            raise ValueError(f"Gold document ID '{gold_doc_id}' not found in docs")
        target = f"[{gold_doc_id}] " + doc["paragraph_text"]

    full_text = prompt + target + "<|im_end|>"

    return full_text


def load_examples_as_dataset(
    file_path: str,
    query_key: str = "query",
    docs_key: str = "docs",
    gold_idx_key: str = "gold_doc_idx",
    model_base_class: str = "Qwen2.5-7B-Instruct",
    include_doc_index: bool = True,
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
        if "qwen" in model_base:
            separator = "\n"
        elif "llama" in model_base:
            separator = " \n"
        else:
            separator = "\n"

        for query, docs, gold_ids in zip(queries, docs_list, gold_ids_list):
            # Handle both single string and list of strings
            if isinstance(gold_ids, str):
                gold_ids = [gold_ids]

            # Create a training example for each gold ID
            for gold_id in gold_ids:
                text = format_qwen_prompt(
                    query=query,
                    docs=docs,
                    gold_doc_id=gold_id,
                    separator=separator,
                    include_doc_index=include_doc_index,
                )
                texts.append(text)

        return {"text": texts}

    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

    return dataset


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
