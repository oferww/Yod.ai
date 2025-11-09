"""
Semantic RAG system using ChromaDB for embeddings and metadata filtering.
Combines semantic matching with hard spec filtering using ChromaDB's built-in capabilities.
"""

import json
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from langchain_cohere import CohereRerank


class SemanticRAG:
    """
    Semantic Retrieval-Augmented Generation for laptop descriptions using ChromaDB.
    Stores laptop descriptions with full specs as metadata for hybrid filtering.
    """
    
    def __init__(self, products: List[Dict[str, Any]], persist_dir: str = "/app/chroma_data", cohere_api_key: Optional[str] = None, api_call_logger=None):
        """
        Initialize the semantic RAG system with ChromaDB.
        
        Args:
            products: List of product dictionaries with full laptop specs
            persist_dir: Directory to persist ChromaDB embeddings
            cohere_api_key: Optional Cohere API key for reranking
            api_call_logger: Optional callback function to log API calls (e.g., chatbot._log_api_call)
        """
        self.products = products
        self.sku_to_product = {p.get('SKU'): p for p in products}
        self.persist_dir = persist_dir
        self.cohere_api_key = cohere_api_key
        self.api_call_logger = api_call_logger
        
        # Use langchain's CohereRerank for consistency with the rest of the codebase
        self.reranker = None
        print(f"[SEMANTIC_RAG] Cohere API key: {'Yes (length={})'.format(len(cohere_api_key)) if cohere_api_key else 'None/Empty'}")
        if cohere_api_key:
            try:
                self.reranker = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-english-v3.0")
                print(f"[SEMANTIC_RAG] Reranker initialized successfully with model: rerank-english-v3.0")
            except Exception as e:
                print(f"[SEMANTIC_RAG] WARNING: Failed to initialize reranker: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[SEMANTIC_RAG] WARNING: No Cohere API key provided, reranking disabled")
        
        # Create persist directory if it doesn't exist
        try:
            os.makedirs(persist_dir, exist_ok=True)
            print(f"[CHROMADB] Created persist directory: {persist_dir}")
        except Exception as e:
            print(f"[CHROMADB] Warning: Could not create directory {persist_dir}: {e}")
        
        # Initialize ChromaDB client with persistent storage
        # Embeddings are stored on disk and reused on startup
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            print(f"[CHROMADB] Initialized PersistentClient at {persist_dir}")
        except Exception as e:
            print(f"[CHROMADB] Error with PersistentClient: {e}, falling back to EphemeralClient")
            self.client = chromadb.EphemeralClient()
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="laptops",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Populate collection with products (only adds if not already present)
        self._populate_collection()
        
        # Force download and initialization of the embedding model
        # This ensures the model is downloaded and cached at startup, not on first query
        # Must happen AFTER collection is populated
        print(f"[CHROMADB] Pre-loading embedding model (this may take 10-15 seconds)...")
        try:
            # Access the actual embedding function to force initialization
            embedding_fn = self.collection._embedding_function
            print(f"[CHROMADB] Embedding function found, pre-computing embeddings...")
            
            # Force compute embeddings for a dummy text to trigger download
            if hasattr(embedding_fn, "_model_name"):
                # This is a more aggressive approach to ensure the model is downloaded
                _ = embedding_fn(["This is a test sentence to force model download"])
                print(f"[CHROMADB] Embedding model pre-loaded successfully")
            else:
                # Fallback to query approach
                self.collection.query(query_texts=["test"], n_results=1)
                print(f"[CHROMADB] Embedding model pre-loaded via query")
        except Exception as e:
            print(f"[CHROMADB] Warning: Could not pre-load model: {e}")
    
    def _populate_collection(self):
        """Populate ChromaDB collection with laptop descriptions and metadata.
        
        Only adds products that aren't already in the collection (for persistence).
        """
        if not self.products:
            return
        
        # Check which products are already in the collection
        try:
            existing_ids = set(self.collection.get(include=[])['ids'])
        except Exception:
            existing_ids = set()
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for product in self.products:
            sku = product.get('SKU', '')
            if not sku or sku in existing_ids:
                continue
            
            # Enrich document with all relevant specs for better embeddings
            description = product.get('Description', '')
            name = product.get('Name', '')
            cpu = product.get('CPU', '')
            gpu = product.get('GPU', '')
            ram = product.get('RAM', '')
            storage = product.get('Storage', '')
            brand = product.get('Brand', '')
            family = product.get('Family', '')
            
            # Create enriched document combining description + specs
            # This gives the embeddings more context about the laptop
            enriched_document = f"""
            {name}
            {description}

            Specifications:
            Brand: {brand}
            Family: {family}
            CPU: {cpu}
            GPU: {gpu}
            RAM: {ram}
            Storage: {storage}
            """.strip()
            
            if not enriched_document:
                continue
            
            ids.append(sku)
            documents.append(enriched_document)
            
            # Store all specs as metadata for filtering
            metadata = {
                'sku': sku,
                'name': product.get('Name', ''),
                'brand': product.get('Brand', '').lower(),
                'family': product.get('Family', ''),
                'cpu': product.get('CPU', '').lower(),
                'gpu': product.get('GPU', '').lower(),
                'ram': product.get('RAM', ''),
                'storage': product.get('Storage', ''),
                'price': product.get('Price', ''),
                'description': description
            }
            
            # Add parsed numeric values for range filtering
            metadata['ram_gb'] = self._parse_ram_gb(product.get('RAM', ''))
            metadata['storage_gb'] = self._parse_storage_gb(product.get('Storage', ''))
            metadata['price_usd'] = self._parse_price_usd(product.get('Price', ''))

            cpu_brand = self._extract_cpu_brand(product.get('CPU', ''))
            if cpu_brand:
                metadata['cpu_brand'] = cpu_brand

            gpu_brand = self._extract_gpu_brand(product.get('GPU', ''))
            if gpu_brand:
                metadata['gpu_brand'] = gpu_brand

            metadatas.append(metadata)
        
        # Add new products to collection
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"[CHROMADB] Added {len(ids)} new laptops to collection")
        else:
            print(f"[CHROMADB] All {len(self.products)} laptops already in collection (loaded from disk)")
    
    @staticmethod
    def _parse_ram_gb(ram_text: str) -> float:
        """Parse RAM string to GB value."""
        if not ram_text:
            return 0.0
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', ram_text.lower())
        return float(match.group(1)) if match else 0.0
    
    @staticmethod
    def _parse_storage_gb(storage_text: str) -> float:
        """Parse storage string to GB value."""
        if not storage_text:
            return 0.0
        import re
        match = re.search(r'(\d+(?:\.\d+)?)\s*(tb|gb)?', storage_text.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            return value * 1024 if unit == 'tb' else value
        return 0.0
    
    @staticmethod
    def _parse_price_usd(price_text: str) -> float:
        """Parse price string to USD value."""
        if not price_text:
            return float('inf')
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', price_text.replace(',', ''))
        return float(match.group(1)) if match else float('inf')

    @staticmethod
    def _extract_brand(text: str, brands: List[str]) -> Optional[str]:
        if not text:
            return None
        lowered = text.lower()
        for brand in brands:
            if brand in lowered:
                return brand
        return None

    @classmethod
    def _extract_cpu_brand(cls, cpu_text: str) -> Optional[str]:
        return cls._extract_brand(cpu_text, ['intel', 'amd', 'apple'])

    @classmethod
    def _extract_gpu_brand(cls, gpu_text: str) -> Optional[str]:
        return cls._extract_brand(gpu_text, ['nvidia', 'amd', 'intel', 'apple'])
    
    def find_semantic_matches(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True
        ) -> List[Tuple[str, float]]:
        """
        Find products semantically similar to the query using ChromaDB.
        
        Args:
            query: User preference query (e.g., "deep learning", "video editing")
            top_k: Maximum number of results to return
            use_rerank: Whether to use Cohere rerank for better accuracy (default True)
        
        Returns:
            List of tuples (SKU, distance_score) sorted by relevance descending
        """
        if not query.strip():
            return []
        
        try:
            # Get more candidates for reranking (3x top_k)
            n_results = top_k * 3 if use_rerank and self.reranker else top_k
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if not results or not results['ids'] or not results['ids'][0]:
                return []
            
            # Get SKUs and documents for reranking
            skus = results['ids'][0]
            
            # If reranking is enabled and we have a reranker, rerank the results
            print(f"[SEMANTIC_RAG] Reranking check: use_rerank={use_rerank}, has_reranker={self.reranker is not None}, skus_count={len(skus)}")
            if use_rerank and self.reranker and len(skus) > 1:
                print(f"[SEMANTIC_RAG] Executing reranking...")
                return self.rerank_results(query, skus, top_k)
            else:
                if not use_rerank:
                    print(f"[SEMANTIC_RAG] Reranking disabled by flag")
                elif not self.reranker:
                    print(f"[SEMANTIC_RAG] Reranker not initialized")
                elif len(skus) <= 1:
                    print(f"[SEMANTIC_RAG] Not enough results to rerank")
            
            # Otherwise, use ChromaDB's distance-based ranking
            # Convert distances to similarity scores (ChromaDB returns distances)
            # Distance 0 = perfect match, distance 2 = no match
            matches = []
            for sku, distance in zip(results['ids'][0], results['distances'][0]):
                # Convert distance to similarity (1 - normalized_distance)
                similarity = 1.0 - (distance / 2.0)
                matches.append((sku, max(0.0, similarity)))
            
            return matches[:top_k]
        except Exception as e:
            print(f"[CHROMADB] Error in semantic search: {e}")
            return []
    
    def rerank_results(
        self,
        query: str,
        skus: List[str],
        top_k: int = 5
        ) -> List[Tuple[str, float]]:
        """
        Rerank search results using Cohere's rerank API via langchain.
        
        Args:
            query: User's search query
            skus: List of SKUs to rerank
            top_k: Number of top results to return
        
        Returns:
            List of tuples (SKU, relevance_score) sorted by relevance descending
        """
        if not self.reranker or not skus:
            return [(sku, 0.0) for sku in skus[:top_k]]
        
        try:
            # Get documents for each SKU (use description + name for reranking)
            documents = []
            sku_list = []  # Maintain order mapping
            for sku in skus:
                product = self.sku_to_product.get(sku)
                if product:
                    # Create reranking document (prioritize description)
                    doc = f"{product.get('Name', '')} - {product.get('Description', '')}"
                    documents.append(doc)
                    sku_list.append(sku)
            
            if not documents:
                return [(sku, 0.0) for sku in skus[:top_k]]
            
            print(f"[RERANK] Reranking {len(documents)} results for query: '{query}'")
            
            # Log API call if logger is available
            if self.api_call_logger:
                self.api_call_logger("rerank", "COHERE_API_KEY", note="semantic_reranking")
            else:
                print(f"[API_CALL] type=rerank key=COHERE_API_KEY note=semantic_reranking", flush=True)
            
            # Call langchain CohereRerank API
            rerank_response = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents))
            )
            
            # Extract reranked results
            # rerank_response is a list of dicts with 'index' and 'relevance_score'
            reranked = []
            for result in rerank_response:
                doc_idx = result['index']
                relevance_score = result['relevance_score']
                if 0 <= doc_idx < len(sku_list):
                    sku = sku_list[doc_idx]
                    reranked.append((sku, relevance_score))
            
            print(f"[RERANK] Reranked top {len(reranked)} results")
            
            # Log top 5 results for debugging
            print(f"[RERANK] Top 5 results:")
            for i, (sku, score) in enumerate(reranked[:5], 1):
                product = self.sku_to_product.get(sku)
                name = product.get('Name', 'Unknown') if product else 'Unknown'
                print(f"[RERANK]   {i}. {name} (SKU: {sku}, score: {score:.4f})")
            
            return reranked
            
        except Exception as e:
            print(f"[RERANK] Error in reranking: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original order
            return [(sku, 0.0) for sku in skus[:top_k]]
    
    def filter_by_specs(
        self,
        where_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10
        ) -> List[str]:
        """
        Filter laptops by hard specs using ChromaDB metadata filtering.
        
        Args:
            where_filter: ChromaDB where clause for metadata filtering
            top_k: Maximum number of results
        
        Returns:
            List of matching SKUs
        """
        if not where_filter:
            # Return all SKUs if no filter
            results = self.collection.get()
            return results['ids'][:top_k] if results['ids'] else []
        
        try:
            results = self.collection.get(where=where_filter)
            return results['ids'][:top_k] if results['ids'] else []
        except Exception as e:
            print(f"[CHROMADB] Error in spec filtering: {e}")
            return []

