"""Dataset loaders (lightweight, CPU-friendly).

This module avoids importing torchvision at import-time. Callers should handle
installing torchvision if they want to use the provided CIFAR-10 loader.
"""
from typing import Tuple, Any, Dict, List
import os


def get_cifar10_loaders(root: str = "./data", batch_size: int = 64, num_workers: int = 0) -> Tuple[Any, Any]:
    """Return train and test dataloaders for CIFAR-10.

    Requires `torch` and `torchvision` to be installed. This function imports them lazily.
    """
    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch and torchvision are required to load CIFAR-10") from exc

    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


class AGNewsDataset:
    """AG News dataset handler for topic classification (4 classes)."""
    
    CLASSES = ['World', 'Sports', 'Business', 'Sci-Tech']
    
    def __init__(self, root: str = "./data", vocab_size: int = 20000):
        import torch
        from torchtext import datasets as text_datasets
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
        
        self.root = root
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = get_tokenizer('basic_english')
        
        os.makedirs(root, exist_ok=True)
        
        # Load dataset
        train_iter = text_datasets.AG_NEWS(root=root, split='train')
        test_iter = text_datasets.AG_NEWS(root=root, split='test')
        
        # Build vocab
        def yield_tokens(data_iter):
            for label, text in data_iter:
                yield self.tokenizer(text)
        
        # Rebuild train_iter since it was consumed
        train_iter = text_datasets.AG_NEWS(root=root, split='train')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
    
    def get_loaders(self, batch_size: int = 64) -> Tuple[Any, Any]:
        """Return train and test dataloaders for AG News."""
        import torch
        from torchtext import datasets as text_datasets
        
        def collate_batch(batch):
            """Collate function for variable-length sequences."""
            labels = []
            texts = []
            max_len = 0
            
            for label, text in batch:
                tokens = self.tokenizer(text)
                if len(tokens) > 0:
                    tokens = tokens[:150]  # Limit to 150 tokens
                    max_len = max(max_len, len(tokens))
                    labels.append(label - 1)  # 0-indexed
                    texts.append(tokens)
            
            # Pad sequences
            padded_texts = []
            for tokens in texts:
                if max_len > 0:
                    padded = tokens + [self.vocab['<pad>']] * (max_len - len(tokens))
                    padded_texts.append(self.vocab(padded[:max_len]))
            
            if len(padded_texts) == 0:
                return None, None
            
            return torch.tensor(labels), torch.stack(padded_texts).float()
        
        train_iter = text_datasets.AG_NEWS(root=self.root, split='train')
        test_iter = text_datasets.AG_NEWS(root=self.root, split='test')
        
        train_loader = torch.utils.data.DataLoader(
            train_iter, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_batch
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_iter,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
        
        return train_loader, test_loader


def get_ag_news_loaders(root: str = "./data", batch_size: int = 64) -> Tuple[Any, Any]:
    """Return train and test dataloaders for AG News (4-class topic classification).
    
    Requires `torch` and `torchtext` to be installed.
    """
    try:
        import torch
        from torchtext import datasets as text_datasets
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except Exception as exc:
        raise RuntimeError("torch and torchtext are required to load AG News") from exc
    
    os.makedirs(root, exist_ok=True)
    
    tokenizer = get_tokenizer('basic_english')
    
    # Load and build vocab
    train_iter = text_datasets.AG_NEWS(root=root, split='train')
    
    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    # Collate function
    def collate_batch(batch):
        """Process variable-length sequences."""
        labels = []
        texts = []
        max_len = 0
        
        for label, text in batch:
            tokens = tokenizer(text)[:150]  # Max 150 tokens
            if len(tokens) > 0:
                max_len = max(max_len, len(tokens))
                labels.append(label - 1)  # Convert to 0-indexed
                texts.append(tokens)
        
        if not texts or max_len == 0:
            return None, None
        
        # Pad and convert to indices
        padded = []
        for tokens in texts:
            pad_len = max_len - len(tokens)
            indexed = vocab(tokens) + [vocab['<pad>']] * pad_len
            padded.append(indexed)
        
        return torch.tensor(labels, dtype=torch.long), torch.tensor(padded, dtype=torch.float32)
    
    # Create loaders
    train_iter = text_datasets.AG_NEWS(root=root, split='train')
    test_iter = text_datasets.AG_NEWS(root=root, split='test')
    
    train_loader = torch.utils.data.DataLoader(
        train_iter,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_iter,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    return train_loader, test_loader, vocab
