import asyncio
import os
import sys
import uuid
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryBufferMemory

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now use absolute imports
from utils.key_bank import get_keybank
from utils.semantic_rag import SemanticRAG
from utils.laptop_filter import LaptopFilter
from utils.llm_provider import LLMProviderFactory

class OferGPT:
    def __init__(self):
        self._api_counts = {}
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())
        self._keybank = get_keybank()
        
        # Track accumulated user preferences across the session
        # This persists preferences so we don't re-ask for specs already provided
        self.accumulated_preferences = {}
        
        # Track accumulated semantic query for reranking (excludes hard spec messages)
        # This ensures we use semantic intent like "deep learning" or "battery life" 
        # for reranking, not hard spec updates like "under $3600"
        self.accumulated_semantic_query = ""
        
        # Track recommendations given in this session
        self.recommendations_given = []
        
        # Track token counts from last LLM response
        self._last_token_counts = None
        
        # Track number of user prompts (product-related) to control recommendation timing
        self.product_prompt_count = 0
        
        # Minimum prompts required before showing recommendations
        self.min_prompts_for_recommendation = 2
        
        # Initialize LLM for memory summarization
        _mem_key = self._get_api_key_for_provider()
        _llm_provider = self._create_llm_instance(
            api_key=_mem_key,
            temperature=0.35,
            max_tokens=200
        )
        
        # Store both the provider and the underlying LangChain LLM
        self.llm_provider = _llm_provider
        self.langchain_llm = _llm_provider.get_langchain_llm()
        
        # Initialize memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.langchain_llm,
            max_token_limit=800,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize RAG system (will be set when products are available)
        self.semantic_rag = None
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant that helps customers find the perfect laptop based on their needs.
        Your role is to have a conversation to understand the user's requirements and recommend suitable laptops from our catalog.
        
        You can ask about TWO types of preferences:
        
        TECHNICAL SPECIFICATIONS (6 main specs):
        1. Brand (e.g., Dell, HP, Lenovo)
        2. CPU (e.g., Intel Core i7, AMD Ryzen 7)
        3. GPU (e.g., NVIDIA RTX 3060, Integrated)
        4. RAM (e.g., 8GB, 16GB, 32GB)
        5. Storage (e.g., 256GB SSD, 1TB HDD)
        6. Maximum price (e.g., $1000)
        
        SEMANTIC/ABSTRACT PREFERENCES (use-case based):
        - Deep learning / AI / ML work
        - Video editing / content creation
        - Photo editing
        - Gaming
        - Portability / travel
        - Professional workstation tasks (CAD, 3D rendering)
        - Battery life
        - Display quality
        - General productivity
        
        Guidelines:
        1. Start by understanding the user's use case (semantic preference)
        2. If they mention a use case, ask clarifying questions about it
           Example: User says "deep learning" -> Ask "Do you prioritize maximum GPU power, or do you need portability?"
        3. Then gather technical specifications as needed
        4. Remember what the user has already told you - don't re-ask for the same information
        5. Use accumulated preferences to refine recommendations
        6. Be friendly and professional in your explanations
        """

    ### Utils ###
    
    def _get_api_key_for_provider(self) -> str:
        """Get API key based on current LLM provider.
        
        Returns:
            str: API key for the current provider
        """
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        
        if provider == "cohere":
            return self._keybank.get_key("memory_summarize")
        elif provider == "google":
            # For Google, we don't use key bank rotation
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            return api_key
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _create_llm_instance(self, api_key: str, temperature: float = 0.7, 
                            max_tokens: int = 1000):
        """Create an LLM instance using the provider factory.
        
        Args:
            api_key: API key for the provider
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            BaseLLMProvider: Configured LLM provider instance
        """
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        
        if provider == "cohere":
            model = os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025")
        elif provider == "google":
            model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
        else:
            model = None
        
        return LLMProviderFactory.create_provider(
            provider_type=provider,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _reset_api_counts(self):
        """Reset per-prompt API call counters."""
        self._api_counts = {
            ("chat", "COHERE_API_KEY"): 0,
            ("embed", "COHERE_API_KEY_EMBED"): 0,
            ("embed", "COHERE_API_KEY"): 0,  # query-time embeddings use CHAT key
            ("rerank", "COHERE_API_KEY"): 0,
        }

    def _log_api_call(self, api_type: str, which_key: str, note: Optional[str] = None, key_index: Optional[int] = None):
        """Record and print a single API call occurrence.

        api_type: 'chat' | 'embed' | 'rerank'
        which_key: 'COHERE_API_KEY' | 'COHERE_API_KEY_EMBED'
        note: optional short context string
        key_index: optional index of key selected from the bank
        """
        key = (api_type, which_key)
        if key not in self._api_counts:
            self._api_counts[key] = 0
        self._api_counts[key] += 1
        try:
            extra = f" note={note}" if note else ""
            idxs = f" key_index={key_index}" if key_index is not None else ""
            print(f"[API_CALL] type={api_type} key={which_key}{extra}{idxs}", flush=True)
        except Exception:
            pass

    def _print_api_totals(self, where: str):
        """Print a per-prompt summary of API calls."""
        try:
            total = sum(self._api_counts.values()) if self._api_counts else 0
            breakdown = {f"{k[0]}:{k[1]}": v for k, v in self._api_counts.items() if v}
            print(f"[API_TOTAL][{where}] total={total} breakdown={json.dumps(breakdown)}", flush=True)
        except Exception:
            pass
    
    def _init_api_counts_for_turn(self):
        """Initialize API counters for a new conversation turn."""
        self._reset_api_counts()

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text to at most 'limit' characters, appending a notice if truncated."""
        try:
            limit = int(limit)
        except Exception:
            limit = 0
        if not text or limit <= 0:
            return "" if limit <= 0 else (text or "")
        if len(text) <= limit:
            return text
        # Leave room for suffix
        suffix = " … [TRUNCATED]"
        keep = max(0, limit - len(suffix))
        return text[:keep] + suffix

    ### Preference tracking ###
    
    def _merge_preferences(self, new_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new preferences with accumulated ones, keeping non-empty values.
        
        New preferences override accumulated ones if provided.
        """
        print(f"\n[PREF_MERGE] Starting merge")
        print(f"[PREF_MERGE] Current accumulated: {self.accumulated_preferences}")
        print(f"[PREF_MERGE] New preferences: {new_prefs}")
        
        merged = self.accumulated_preferences.copy()
        updates = []
        for key, value in new_prefs.items():
            if value is not None and value != "":
                merged[key] = value
                updates.append(f"{key}={value}")
        
        self.accumulated_preferences = merged
        
        print(f"[PREF_MERGE] Updates applied: {updates if updates else 'NONE'}")
        print(f"[PREF_MERGE] Final accumulated: {self.accumulated_preferences}")
        return merged
    
    def _convert_preferences_to_filter_keys(self, prefs: Dict[str, Any]) -> tuple[Dict[str, Any], set]:
        """Convert user-facing preference keys to LaptopFilter-compatible keys.
        
        Returns:
            tuple: (filter_prefs dict, set of processed preference keys)
            The second element contains the original preference keys that were converted.
            Any keys in prefs but not in the returned set are semantic (not hard specs).
        
        Maps:
        - brand -> brand (kept as-is for brand matching)
        - cpu -> cpu_brand (extract brand: intel, amd, apple)
        - gpu -> gpu_brand (extract brand: nvidia, amd, intel, apple)
        - ram -> min_ram_gb (parse GB value)
        - storage -> min_storage_gb (parse GB value)
        - max_price -> max_price (converted to USD when a currency symbol/code is present)
        """
        print(f"\n[PREF_CONVERT] Converting preferences: {prefs}")
        filter_prefs = {}
        processed_keys = set()  # Track which preference keys we actually processed
        
        # Brand (kept as-is)
        if 'brand' in prefs and prefs['brand']:
            filter_prefs['brand'] = prefs['brand']
            processed_keys.add('brand')
        
        # CPU brand extraction and advanced metadata
        if 'cpu' in prefs and prefs['cpu']:
            processed_keys.add('cpu')
            cpu_raw = str(prefs['cpu'])
            cpu_text = cpu_raw.lower()
            for brand in ['intel', 'amd', 'apple']:
                if brand in cpu_text:
                    filter_prefs['cpu_brand'] = brand
                    break

            cpu_series = LaptopFilter.extract_cpu_series(cpu_raw)
            if cpu_series:
                filter_prefs['cpu_series'] = cpu_series

            cpu_keywords = LaptopFilter.extract_cpu_keywords(cpu_raw)
            if cpu_keywords:
                # Focus on discriminative tokens (model numbers, series identifiers)
                prioritized_keywords = []
                seen = set()
                for keyword in cpu_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in seen:
                        continue
                    if any(ch.isdigit() for ch in keyword_lower) or keyword_lower.startswith('i') and len(keyword_lower) == 2 and keyword_lower[1].isdigit() or 'ultra' in keyword_lower or 'xeon' in keyword_lower:
                        prioritized_keywords.append(keyword_lower)
                        seen.add(keyword_lower)
                if prioritized_keywords:
                    filter_prefs['cpu_keywords'] = prioritized_keywords

            cpu_core_count = LaptopFilter.extract_core_count(cpu_raw)
            if cpu_core_count:
                filter_prefs['min_cpu_cores'] = cpu_core_count

            cpu_generation = LaptopFilter.extract_cpu_generation(cpu_raw)
            if cpu_generation:
                filter_prefs['min_cpu_generation'] = cpu_generation

        # GPU brand extraction
        if 'gpu' in prefs and prefs['gpu']:
            processed_keys.add('gpu')
            gpu_raw = str(prefs['gpu'])
            gpu_text = gpu_raw.lower()
            for brand in ['nvidia', 'amd', 'intel', 'apple']:
                if brand in gpu_text:
                    filter_prefs['gpu_brand'] = brand
                    break

            gpu_keywords = LaptopFilter.extract_gpu_keywords(gpu_raw)
            if gpu_keywords:
                prioritized_keywords = []
                seen = set()
                for keyword in gpu_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in seen:
                        continue
                    if any(ch.isdigit() for ch in keyword_lower) or any(prefix in keyword_lower for prefix in ['rtx', 'gtx', 'rx', 'radeon', 'arc', 'iris', 'mx', 'quadro']):
                        prioritized_keywords.append(keyword_lower)
                        seen.add(keyword_lower)
                if prioritized_keywords:
                    filter_prefs['gpu_keywords'] = prioritized_keywords

            gpu_core_count = LaptopFilter.extract_core_count(gpu_raw)
            if gpu_core_count:
                filter_prefs['min_gpu_cores'] = gpu_core_count
        
        # RAM parsing
        if 'ram' in prefs and prefs['ram']:
            processed_keys.add('ram')
            try:
                import re
                ram_text = str(prefs['ram']).lower()
                # Extract just the numeric part using regex
                match = re.search(r'(\d+(?:\.\d+)?)', ram_text)
                if match:
                    ram_gb = float(match.group(1))
                    filter_prefs['min_ram_gb'] = ram_gb
                    print(f"[PREF_CONVERT] Parsed RAM: {prefs['ram']} -> {ram_gb}GB")
                else:
                    print(f"[PREF_CONVERT] Could not extract numeric value from RAM '{prefs['ram']}'")
            except Exception as e:
                print(f"[PREF_CONVERT] Failed to parse RAM '{prefs['ram']}': {e}")
        
        # Storage parsing
        if 'storage' in prefs and prefs['storage']:
            processed_keys.add('storage')
            try:
                import re
                storage_text = str(prefs['storage']).lower()
                # Extract numeric value and unit (tb or gb)
                match = re.search(r'(\d+(?:\.\d+)?)\s*(tb|gb)?', storage_text)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    
                    if unit == 'tb':
                        storage_gb = value * 1024
                    else:  # gb or no unit
                        storage_gb = value
                    
                    filter_prefs['min_storage_gb'] = storage_gb
                    print(f"[PREF_CONVERT] Parsed Storage: {prefs['storage']} -> {storage_gb}GB")
                else:
                    print(f"[PREF_CONVERT] Could not extract numeric value from Storage '{prefs['storage']}'")
            except Exception as e:
                print(f"[PREF_CONVERT] Failed to parse Storage '{prefs['storage']}': {e}")
        
        # Max price (convert to USD if currency provided)
        if 'max_price' in prefs and prefs['max_price']:
            processed_keys.add('max_price')
            max_price_value = prefs['max_price']
            converted = LaptopFilter.parse_price_to_usd(max_price_value)
            if converted is not None:
                filter_prefs['max_price'] = converted
                print(f"[PREF_CONVERT] Parsed max price: {max_price_value} -> ${converted:.2f} USD")
            else:
                filter_prefs['max_price'] = max_price_value
                print(f"[PREF_CONVERT] Could not convert max price '{max_price_value}', keeping original")
        
        # Any keys in prefs but not in processed_keys are semantic preferences
        semantic_keys = set(prefs.keys()) - processed_keys
        if semantic_keys:
            print(f"[PREF_CONVERT] Semantic preferences (not converted): {semantic_keys}")
        print(f"[PREF_CONVERT] Converted to filter keys: {filter_prefs}")
        return filter_prefs, processed_keys

    ### Core chat ###

    def _extract_preferences(self, query: str) -> Dict[str, Any]:
        """Extract structured hard spec preferences from user query using LLM.
        
        Returns:
            Dict containing any of the following keys if specified:
            - brand: str (e.g., "Dell", "HP", "Lenovo")
            - cpu: str (e.g., "Intel Core i7", "AMD Ryzen 7")
            - gpu: str (e.g., "NVIDIA RTX 3060", "Integrated")
            - ram: str (e.g., "16GB", "32GB")
            - storage: str (e.g., "512GB SSD", "1TB HDD")
            - max_price: int (maximum price in dollars)
        """
        print(f"\n[PREF_EXTRACT] Starting hard spec extraction")
        print(f"[PREF_EXTRACT] Query: '{query}'")
        
        prompt = f"""You are an expert at understanding laptop specifications from natural language.
        Extract ONLY hard technical specifications from the user query and return a JSON object.
        
        Available fields:
        {{
            "brand": "",       // Laptop manufacturer (infer from common names: Mac/MacBook → Apple, ThinkPad → Lenovo, etc.)
            "cpu": "",         // Processor type and model
            "gpu": "",         // Graphics card or "Integrated"
            "ram": "",         // Memory amount (e.g., "16GB", "32GB")
            "storage": "",     // Storage capacity and type (e.g., "512GB SSD")
            "max_price": 0     // Maximum price in dollars (number only)
        }}

        Instructions:
        - ONLY extract technical specifications (brand, cpu, gpu, ram, storage, price)
        - Do NOT extract semantic preferences like use case, priorities, or requirements
        - Use your knowledge to infer technical specs from natural language
        - Normalize brand names to official manufacturer names (Mac → Apple, etc.)
        - Extract numeric values for RAM, storage, and price
        - Only include fields that are mentioned or clearly implied
        - Return ONLY valid JSON, no explanations
        
        User query: {query}
        
        JSON response:"""
        
        print(f"[PREF_EXTRACT] Sending to LLM for extraction")
        
        # Log API call for preference extraction
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        if provider == "cohere":
            _, pref_idx = self._keybank.get_key_with_index("preference_extraction")
            self._log_api_call("chat", "COHERE_API_KEY", note="preference_extraction", key_index=pref_idx)
        else:
            self._log_api_call("chat", "GOOGLE_API_KEY", note="preference_extraction")
        
        try:
            response_text = ""
            for chunk in self.langchain_llm.stream([HumanMessage(content=prompt)]):
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if not chunk_text:
                    continue

                if response_text and chunk_text.startswith(response_text):
                    response_text += chunk_text[len(response_text):]
                else:
                    response_text += chunk_text
            
            response_text = response_text.strip().strip('`').strip()
            print(f"[PREF_EXTRACT] LLM raw response: '{response_text}'")

            extracted_json = self._extract_json_block(response_text)
            if extracted_json is None:
                raise ValueError("LLM response did not contain valid JSON block")

            preferences = json.loads(extracted_json)
            print(f"[PREF_EXTRACT] Successfully parsed JSON: {preferences}")
            
            # Log which preferences were found
            found_prefs = [f"{k}={v}" for k, v in preferences.items()]
            print(f"[PREF_EXTRACT] Found preferences: {found_prefs if found_prefs else 'NONE'}")
            
        except Exception as e:
            print(f"[PREF_EXTRACT] ERROR extracting preferences: {e}")
            print(f"[PREF_EXTRACT] Response text was: '{response_text if 'response_text' in locals() else 'N/A'}'")
            preferences = {}

        # Remove any empty strings or None values
        cleaned = {k: v for k, v in preferences.items() if v is not None and v != ""}

        # Fallback parsing for budget / currency phrases when LLM misses them
        if 'max_price' not in cleaned:
            fallback_budget = self._extract_budget_fallback(query)
            if fallback_budget:
                cleaned['max_price'] = fallback_budget
                print(f"[PREF_EXTRACT] Fallback budget detected: {fallback_budget}")

        print(f"[PREF_EXTRACT] After cleanup: {cleaned if cleaned else 'EMPTY'}")
        return cleaned
    
    def _extract_semantic_context(self, query: str, conversation_context: str = "") -> str:
        """Extract semantic preferences as free text using LLM.
        
        This captures use cases, priorities, tradeoffs, and abstract requirements
        without forcing them into structured fields.
        
        Args:
            query: User's current message
            conversation_context: Previous conversation context for better understanding (can be str or list)
        
        Returns:
            Free text semantic context
        """
        print(f"\n[SEMANTIC_EXTRACT] Starting semantic context extraction")
        print(f"[SEMANTIC_EXTRACT] Query: '{query}'")
        
        # Ensure conversation_context is a string
        if conversation_context and not isinstance(conversation_context, str):
            conversation_context = str(conversation_context)
        
        prompt = f"""You are an expert at understanding user needs and preferences for laptops.
        Extract the semantic meaning, use case, priorities, and abstract requirements from the user's message.
        
        Focus on:
        - Use cases (e.g., "deep learning", "gaming", "video editing", "general productivity")
        - Priorities and tradeoffs (e.g., "prioritize GPU over portability", "battery life is important")
        - Abstract requirements (e.g., "needs to be portable", "good for travel", "high performance")
        - Quality preferences (e.g., "good display", "premium build quality", "lightweight")
        
        Instructions:
        - Extract ONLY semantic/abstract preferences, NOT technical specifications
        - Preserve the user's intent and priorities
        - Return a natural language description of their needs
        - If there are no semantic preferences in the message, return an empty string
        - Be concise but capture the full context
        
        {'Previous context: ' + conversation_context if conversation_context else ''}
        User query: {query}
        
        Semantic context:"""
        
        print(f"[SEMANTIC_EXTRACT] Sending to LLM for extraction")
        
        # Log API call for semantic extraction
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        if provider == "cohere":
            _, sem_idx = self._keybank.get_key_with_index("semantic_extraction")
            self._log_api_call("chat", "COHERE_API_KEY", note="semantic_extraction", key_index=sem_idx)
        else:
            self._log_api_call("chat", "GOOGLE_API_KEY", note="semantic_extraction")
        
        try:
            response_text = ""
            for chunk in self.langchain_llm.stream([HumanMessage(content=prompt)]):
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if not chunk_text:
                    continue

                if response_text and chunk_text.startswith(response_text):
                    response_text += chunk_text[len(response_text):]
                else:
                    response_text += chunk_text
            
            response_text = response_text.strip()
            print(f"[SEMANTIC_EXTRACT] LLM raw response: '{response_text}'")
            
            # Check if LLM indicated no semantic preferences
            if response_text.upper() == "NONE" or not response_text:
                print(f"[SEMANTIC_EXTRACT] No semantic context found")
                return ""
            
            print(f"[SEMANTIC_EXTRACT] Extracted semantic context: '{response_text}'")
            return response_text
            
        except Exception as e:
            print(f"[SEMANTIC_EXTRACT] ERROR extracting semantic context: {e}")
            return ""

    def _build_chromadb_where_filter(self, filter_prefs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where filter from preference dict.
        
        Args:
            filter_prefs: Dictionary of filter preferences
        
        Returns:
            ChromaDB where clause or None
        """
        if not filter_prefs:
            return None
        
        conditions = []
        
        # Brand filter
        if 'brand' in filter_prefs and filter_prefs['brand']:
            brand = filter_prefs['brand'].lower()
            conditions.append({"brand": brand})
        
        # RAM filter (minimum)
        if 'min_ram_gb' in filter_prefs:
            conditions.append({"ram_gb": {"$gte": filter_prefs['min_ram_gb']}})
        
        # Storage filter (minimum)
        if 'min_storage_gb' in filter_prefs:
            conditions.append({"storage_gb": {"$gte": filter_prefs['min_storage_gb']}})
        
        # Price filter (maximum)
        if 'max_price' in filter_prefs:
            conditions.append({"price_usd": {"$lte": filter_prefs['max_price']}})
        
        # CPU filter
        if 'cpu_brand' in filter_prefs:
            cpu_brand = filter_prefs['cpu_brand'].lower()
            conditions.append({"cpu_brand": cpu_brand})
        
        # GPU filter
        if 'gpu_brand' in filter_prefs:
            gpu_brand = filter_prefs['gpu_brand'].lower()
            conditions.append({"gpu_brand": gpu_brand})
        
        # Combine conditions with AND logic
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        # Multiple conditions - use $and
        return {"$and": conditions}
    
    @staticmethod
    def _extract_budget_fallback(query: str) -> Optional[str]:
        if not query:
            return ""

        import re

        budget_patterns = [
            re.compile(r"(?P<currency>nis|₪|ils|usd|\$|dollars?|eur|€|gbp|£|cad|c\$|aud|a\$|inr|₹|sgd|s\$|chf)\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>[km])?", re.IGNORECASE),
            re.compile(r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>[km])?\s*(?P<currency>nis|₪|ils|usd|\$|dollars?|eur|€|gbp|£|cad|c\$|aud|a\$|inr|₹|sgd|s\$|chf)", re.IGNORECASE),
        ]

        for pattern in budget_patterns:
            match = pattern.search(query)
            if not match:
                continue

            amount = match.group('amount')
            currency = match.group('currency')
            suffix = match.group('suffix') or ''
            if amount and currency:
                return f"{amount}{suffix} {currency}".strip()

        return None

    # unused
    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None

        import re

        fenced_match = re.search(r"```json\s*({[\s\S]*?})\s*```", text, flags=re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()

        brace_match = re.search(r"({[\s\S]*})", text)
        if brace_match:
            return brace_match.group(1).strip()

        if text.lower().startswith('json'):
            stripped = text[4:].strip()
            if stripped.startswith('{'):
                return stripped

        stripped = text.strip()
        return stripped if stripped.startswith('{') else None

    def _track_recommendation(self, products: List[Dict[str, Any]], filter_type: str = "semantic") -> None:
        """Track products recommended to the user.
        
        Args:
            products: List of recommended products
            filter_type: Type of filtering used ('semantic', 'hard_specs', or 'hybrid')
        """
        if products:
            recommendation = {
                "timestamp": datetime.utcnow().isoformat(),
                "filter_type": filter_type,
                "products": [
                    {
                        "sku": p.get('SKU'),
                        "name": p.get('Name'),
                        "brand": p.get('Brand'),
                        "price": p.get('Price'),
                        "cpu": p.get('CPU'),
                        "gpu": p.get('GPU'),
                        "ram": p.get('RAM'),
                        "storage": p.get('Storage'),
                        "description": p.get('Description')
                    }
                    for p in products[:3]  # Track top 3 recommendations
                ]
            }
            self.recommendations_given.append(recommendation)
            print(f"[RECOMMENDATION_TRACKED] {len(recommendation['products'])} products recommended (filter_type: {filter_type})")
    
    def _generate_no_matches_response(self, query: str, preferences: Dict[str, Any], 
                         semantic_context: str = "") -> Generator[str, None, None]:
        """Generate a response when no products match the user's preferences.
        
        This acknowledges the constraints and asks the user to refine their preferences.
        
        Args:
            query: Current user query
            preferences: Hard spec preferences that resulted in no matches
            semantic_context: Accumulated semantic preferences
        """
        # Build the prompt for the LLM to generate a natural no-matches response
        prompt = f"""You are a helpful AI assistant helping a customer find a laptop.

Unfortunately, no laptops in our catalog match the user's current preferences. Your task is to:
1. Acknowledge what they're looking for
2. Explain why no matches were found (based on their constraints)
3. Ask them to refine their preferences to find suitable options

User's query: {query}

{"User's priorities: " + semantic_context if semantic_context else ""}

Current preferences that resulted in no matches:
{json.dumps(preferences, indent=2) if preferences else "None specified"}

Your response should:
- Be empathetic and helpful
- Explain which constraints are too restrictive
- Suggest specific ways they could adjust their preferences
- Ask them to provide more flexible constraints
- End with a Yoda from Star Wars pun about finding the right laptop

Generate a natural, conversational response."""
        
        # Stream the response from the LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Get a new chat key for this response
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        if provider == "cohere":
            chat_key, chat_idx = self._keybank.get_key_with_index("no_matches_response")
            self._log_api_call("chat", "COHERE_API_KEY", note="no_matches_response", key_index=chat_idx)
        else:
            chat_key = self._get_api_key_for_provider()
            self._log_api_call("chat", "GOOGLE_API_KEY", note="no_matches_response")
        
        chat_llm = self._create_llm_instance(
            api_key=chat_key,
            temperature=0.7,
            max_tokens=800
        )
        
        # Stream the response
        full_response = ""
        chunk_count = 0
        print(f"[NO_MATCHES_STREAM] Starting to stream no-matches response from LLM", flush=True)
        for chunk in chat_llm.stream(messages):
            chunk_count += 1
            full_response += chunk
            yield chunk
        
        print(f"[NO_MATCHES_STREAM] Streaming complete. Total chunks: {chunk_count}, Final response length: {len(full_response)}", flush=True)
        
        # Add to memory
        self.memory.chat_memory.add_ai_message(full_response)
    
    def _generate_clarification_response(self, query: str, preferences: Dict[str, Any], 
                         semantic_context: str = "") -> Generator[str, None, None]:
        """Generate a conversational response asking for clarification before recommendations.
        
        This is used during the early stages of conversation (before min_prompts_for_recommendation)
        to gather user preferences without showing product recommendations yet.
        
        Args:
            query: Current user query
            preferences: Hard spec preferences gathered so far
            semantic_context: Accumulated semantic preferences (use cases, priorities)
        """
        # Determine what information we still need
        hard_spec_keys = ['brand', 'cpu', 'gpu', 'ram', 'storage', 'max_price']
        missing_specs = []
        
        if not preferences.get('brand'):
            missing_specs.append("preferred brand (e.g., Dell, HP, Lenovo)")
        if not preferences.get('cpu'):
            missing_specs.append("processor type (e.g., Intel Core i7, AMD Ryzen 7)")
        if not preferences.get('gpu'):
            missing_specs.append("graphics card (e.g., NVIDIA RTX 3060, Integrated)")
        if not preferences.get('ram'):
            missing_specs.append("amount of RAM (e.g., 8GB, 16GB, 32GB)")
        if not preferences.get('storage'):
            missing_specs.append("storage capacity (e.g., 256GB SSD, 1TB HDD)")
        if not preferences.get('max_price'):
            missing_specs.append("maximum budget (e.g., $1000)")
        
        # Show which specs we already have
        provided_specs = [k for k in hard_spec_keys if preferences.get(k)]
        
        # Build the prompt for the LLM to generate a natural clarification response
        prompt = f"""You are a helpful AI assistant gathering information about the user's laptop needs.

User's current query: {query}

{"User's priorities and use case: " + semantic_context if semantic_context else ""}

Information we already have:
{json.dumps(preferences, indent=2) if preferences else "None yet"}

Your task:
1. Acknowledge what the user has told you so far
2. Ask clarifying questions to understand their needs better
3. Gather information about BOTH technical specs AND use-case preferences:

   TECHNICAL SPECIFICATIONS:
   - Brand preference (e.g., Dell, HP, Lenovo)
   - Processor type (e.g., Intel Core i7, AMD Ryzen 7)
   - Graphics card (e.g., NVIDIA RTX 3060, Integrated)
   - RAM amount (e.g., 8GB, 16GB, 32GB)
   - Storage capacity (e.g., 256GB SSD, 1TB HDD)
   - Maximum budget (e.g., $1000)
   
   SEMANTIC/USE-CASE PREFERENCES (if not already mentioned):
   - Primary use case (e.g., deep learning, video editing, gaming, content creation, productivity)
   - Key priorities (e.g., performance, portability, battery life, display quality)
   - Specific workloads or applications they plan to use

Guidelines:
- Be conversational and friendly
- Don't ask for information you already have
- Ask 1-2 questions at a time, not all at once
- Mix technical and semantic questions naturally - don't ask all technical specs at once
- If they mention a use case, ask follow-up questions about it
- Help them clarify their priorities and tradeoffs
- Use markdown formatting for clarity
- End with a Yoda from Star Wars pun about gathering more information

Generate a natural response that continues the conversation naturally."""
        
        # Stream the response from the LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Get a new chat key for this response
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        if provider == "cohere":
            chat_key, chat_idx = self._keybank.get_key_with_index("clarification_response")
            self._log_api_call("chat", "COHERE_API_KEY", note="clarification_response", key_index=chat_idx)
        else:
            chat_key = self._get_api_key_for_provider()
            self._log_api_call("chat", "GOOGLE_API_KEY", note="clarification_response")
        
        chat_llm = self._create_llm_instance(
            api_key=chat_key,
            temperature=0.7,
            max_tokens=800
        )
        
        # Stream the response
        full_response = ""
        chunk_count = 0
        print(f"[CLARIFICATION_STREAM] Starting to stream clarification response from LLM", flush=True)
        for chunk in chat_llm.stream(messages):
            chunk_count += 1
            full_response += chunk
            yield chunk
        
        print(f"[CLARIFICATION_STREAM] Streaming complete. Total chunks: {chunk_count}, Final response length: {len(full_response)}", flush=True)
        
        # Add to memory
        self.memory.chat_memory.add_ai_message(full_response)
    
    def _generate_recommendation_response(self, query: str, products: List[Dict[str, Any]], preferences: Dict[str, Any], 
                         context: Optional[Dict] = None, semantic_context: str = "", filter_type: str = "semantic") -> Generator[str, None, None]:
        """Generate a conversational response based on filtered products.
        
        Args:
            query: Current user query
            products: List of matching products
            preferences: Hard spec preferences (brand, cpu, gpu, ram, storage, max_price)
            context: Additional context
            semantic_context: Accumulated semantic preferences (use cases, priorities, tradeoffs)
            filter_type: Type of filtering used ('semantic', 'hard_specs', or 'hybrid')
        
        The response should ONLY ask about the following specifications if needed:
        - Brand (e.g., Dell, HP, Lenovo)
        - CPU (e.g., Intel Core i7, AMD Ryzen 7)
        - GPU (e.g., NVIDIA RTX 3060, Integrated)
        - RAM (e.g., 8GB, 16GB, 32GB)
        - Storage (e.g., 256GB SSD, 1TB HDD)
        - Maximum price (in dollars)
        
        Do not ask about any other specifications or features.
        """
        if not products:
            # If no products match, ask for clarification on missing specs only
            missing_specs = []
            if not preferences.get('brand'):
                missing_specs.append("preferred brand (e.g., Dell, HP, Lenovo)")
            if not preferences.get('cpu'):
                missing_specs.append("processor type (e.g., Intel Core i7, AMD Ryzen 7)")
            if not preferences.get('gpu'):
                missing_specs.append("graphics card (e.g., NVIDIA RTX 3060, Integrated)")
            if not preferences.get('ram'):
                missing_specs.append("amount of RAM (e.g., 8GB, 16GB, 32GB)")
            if not preferences.get('storage'):
                missing_specs.append("storage capacity (e.g., 256GB SSD, 1TB HDD)")
            if not preferences.get('max_price'):
                missing_specs.append("maximum budget (e.g., $1000)")
            
            # Show which specs we already have
            spec_to_natural_language = {
                'brand': 'brand preference',
                'cpu': 'processor type',
                'gpu': 'graphics card',
                'ram': 'RAM amount',
                'storage': 'storage capacity',
                'max_price': 'budget'
            }
            provided_specs = [spec_to_natural_language[k] for k in ['brand', 'cpu', 'gpu', 'ram', 'storage', 'max_price'] if preferences.get(k)]
            if provided_specs:
                specs_text = ', '.join(provided_specs)
                yield f"Thanks for providing your {specs_text}. "
                
            if missing_specs:
                if len(missing_specs) > 1:
                    yield "To find the best match, could you also tell me: " + ", ".join(missing_specs[:-1]) + ", or " + missing_specs[-1] + "?"
                else:
                    yield f"To narrow down the options, could you provide: {missing_specs[0]}?"
            else:
                yield "I couldn't find any laptops matching all your criteria. Would you like to adjust any of your specifications?"
            return
        
        # Track the recommendation
        self._track_recommendation(products, filter_type=filter_type)

        # Determine which hard specs the user explicitly provided
        hard_spec_keys = ['brand', 'cpu', 'gpu', 'ram', 'storage', 'max_price']
        user_hard_specs = {k: preferences[k] for k in hard_spec_keys if preferences.get(k)}

        spec_to_product_field = {
            'brand': 'Brand',
            'cpu': 'CPU',
            'gpu': 'GPU',
            'ram': 'RAM',
            'storage': 'Storage',
            'max_price': 'Price'
        }

        def _format_price(value: Any) -> str:
            if isinstance(value, (int, float)):
                return f"${value:,.2f}"
            if value is None or value == "":
                return "N/A"
            return str(value)

        # Prepare product info for the prompt
        product_info = []
        for i, p in enumerate(products[:3], 1):  # Limit to top 3 matches
            price_value = p.get('Price')
            matched_specs = {}
            for pref_key, product_field in spec_to_product_field.items():
                if pref_key in user_hard_specs:
                    if pref_key == 'max_price':
                        matched_specs[pref_key] = _format_price(price_value)
                    else:
                        matched_specs[pref_key] = p.get(product_field)

            product_snapshot = {
                "rank": i,
                "name": p.get('Name', 'Unnamed'),
                "price": _format_price(price_value),
                "description": p.get('Description', ''),
                "matched_specs": matched_specs
            }

            product_info.append(json.dumps(product_snapshot, ensure_ascii=False))

        prompt = f"""You are a helpful AI assistant that helps customers find the perfect laptop.

        CRITICAL INSTRUCTIONS:
        - ONLY use information explicitly provided in the product specifications below
        - Do NOT use external knowledge or real-world assumptions about products
        - Do NOT add features or specifications not listed in the data
        - Only mention what is explicitly stated in each product's description and specs
        - Format your response with proper markdown for clarity and organization
        - Use markdown formatting:
           - Use **bold** for product names and important specs
           - Use bullet points (- or *) for lists of features
           - Use line breaks between sections for clarity
           - Use numbered lists when showing multiple options
                
        User's current query: {query}
        
        {"User's full context and priorities: " + semantic_context if semantic_context else ""}
        
        User's technical specifications:
        {json.dumps(preferences, indent=2) if preferences else "None specified"}
        
        Matching laptops (up to 3 best matches):
        {chr(10).join(product_info)}
        
        Your response MUST:
        1. Start with a brief intro that EXACTLY reflects the user's priorities and context (if provided above). Preserve the user's exact tradeoffs and priorities
        2. Provide natural language descriptions for each laptop, using the product description text as the foundation
        3. Include all the details mentioned in the original product description
        4. Always mention the price of each laptop
        5. Only include hard specs that the user explicitly requested; use the "matched_specs" field to reference them
        6. Explain briefly why each laptop fits the user's stated needs, priorities, or constraints. Reference the user's exact priorities 
        7. End by asking if they'd like more details about any specific model
        8. If there are no matches, explain why
        9. Always end with some Yoda from star wars pun about that you can get more preferences from the user and refine the reccomendations
        
        """
        
        # Stream the response from the LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Get a new chat key for this response
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        if provider == "cohere":
            chat_key, chat_idx = self._keybank.get_key_with_index("recommendation_response")
            self._log_api_call("chat", "COHERE_API_KEY", note="recommendation_response", key_index=chat_idx)
        else:
            chat_key = self._get_api_key_for_provider()
            self._log_api_call("chat", "GOOGLE_API_KEY", note="recommendation_response")
        
        chat_llm = self._create_llm_instance(
            api_key=chat_key,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Stream the response
        full_response = ""
        chunk_count = 0
        print(f"[CHAT_STREAM] Starting to stream response from LLM", flush=True)
        for chunk in chat_llm.stream(messages):
            # The provider's stream method yields strings directly
            chunk_count += 1
            full_response += chunk
            # print(f"[CHAT_STREAM] Received chunk {chunk_count}: {len(chunk)} chars, total: {len(full_response)} chars", flush=True)
            yield chunk
        
        print(f"[CHAT_STREAM] Streaming complete. Total chunks: {chunk_count}, Final response length: {len(full_response)}", flush=True)
        print(f"[CHAT_STREAM] Raw LLM response:\n{full_response}", flush=True)
        
        # Capture token counts from the LLM provider after streaming
        print(f"[CHAT_STREAM] Attempting to retrieve token counts from LLM provider", flush=True)
        print(f"[CHAT_STREAM] Provider type: {type(chat_llm).__name__}", flush=True)
        token_counts = chat_llm.get_token_counts()
        if token_counts:
            input_tokens = token_counts.get('input_tokens', 0)
            output_tokens = token_counts.get('output_tokens', 0)
            print(f"[CHAT_STREAM] ✓ Token counts captured: input={input_tokens}, output={output_tokens}", flush=True)
            # Store token counts for later retrieval
            self._last_token_counts = token_counts
            print(f"[CHAT_STREAM] ✓ Token counts stored in self._last_token_counts", flush=True)
        else:
            print(f"[CHAT_STREAM] ✗ No token counts available from LLM provider", flush=True)
            self._last_token_counts = None
        
        # Add to memory
        self.memory.chat_memory.add_ai_message(full_response)
        
    def chat_stream(self, user_input: str, context: Optional[Dict] = None, 
                   message_history: Optional[List[Dict]] = None) -> Generator[str, None, None]:
        """Main chat method following the new architecture."""
        # Initialize API counters for this turn
        self._init_api_counts_for_turn()
        
        try:
            # Add user message to memory
            self.memory.chat_memory.add_user_message(user_input)
            
            # Get memory context for intent detection
            memory_context = self.memory.load_memory_variables({}).get('chat_history', '')
            
            # Combine memory context with full message history if available
            full_context = memory_context
            
            # Get message history from context if not provided directly
            if message_history is None and context and 'message_history' in context:
                message_history = context['message_history']
            
            # If we have message_history, use it to provide concise context (no timestamps)
            if message_history:
                formatted_history = []
                for msg in message_history:
                    role = 'User' if msg.get('role') == 'user' else 'Assistant'
                    content = msg.get('content', '').strip()
                    if content:
                        formatted_history.append(f"{role}: {content}")

                if formatted_history:
                    full_context = (
                        f"{memory_context}\n\n"
                        f"Conversation history (most recent first):\n"
                        f"{chr(10).join(reversed(formatted_history))}"
                    )
                    print(f"\n=== FULL CONTEXT ===\n{full_context}\n===================\n")
            
            # Detect user intent (only using current query, not full history to save tokens)
            intent = self._detect_intent_llm(user_input)
            self._last_detected_intent = intent  # Store for metrics tracking
            print(f"Detected intent: {intent}")
            
            # Intent categories
            greeting_intents = ['greeting', 'chitchat', 'nonsense']
            preference_intents = ['preferences_given', 'recommendation_request', 'product_inquiry']
            
            if intent in greeting_intents:
                # For greetings and small talk, just respond without product recommendations
                print(f"Processing greeting/small talk: {intent}")
                yield from self.stream_response_with_memory(
                    query=user_input,
                    memory_context=full_context,
                    context=context
                )
                return
                
            if intent in preference_intents:
                print(f"\n[CHAT_FLOW] Processing product intent: {intent}")
                
                # Increment product prompt counter
                self.product_prompt_count += 1
                print(f"[CHAT_FLOW] Product prompt count: {self.product_prompt_count}/{self.min_prompts_for_recommendation}")
                
                # Step 1a: Extract technical (hard spec) preferences using LLM
                print(f"[CHAT_FLOW] Step 1a: Extracting hard spec preferences from user input")
                new_preferences = self._extract_preferences(user_input)
                print(f"[CHAT_FLOW] Step 1a result: {new_preferences if new_preferences else 'NO HARD SPECS FOUND'}")
                
                # Step 1b: Extract semantic context as free text
                print(f"[CHAT_FLOW] Step 1b: Extracting semantic context from user input")
                semantic_context = self._extract_semantic_context(user_input, full_context)
                # Ensure semantic_context is a string
                if semantic_context and not isinstance(semantic_context, str):
                    semantic_context = str(semantic_context)
                
                if semantic_context:
                    print(f"[CHAT_FLOW] Step 1b result: '{semantic_context}'")
                    # Accumulate semantic context
                    if self.accumulated_semantic_query:
                        # Add with single space after period
                        self.accumulated_semantic_query = f"{self.accumulated_semantic_query} {semantic_context}"
                    else:
                        self.accumulated_semantic_query = semantic_context
                    print(f"[CHAT_FLOW] Accumulated semantic query: '{self.accumulated_semantic_query}'")
                else:
                    print(f"[CHAT_FLOW] Step 1b result: NO SEMANTIC CONTEXT FOUND")
                
                # Step 2: Merge with accumulated preferences (remember what user already told us)
                print(f"[CHAT_FLOW] Step 2: Merging with accumulated preferences")
                preferences = self._merge_preferences(new_preferences)
                print(f"[CHAT_FLOW] Step 2 result: {preferences if preferences else 'EMPTY AFTER MERGE'}")
                
                # Step 3: Convert to filter-compatible keys and filter products
                print(f"[CHAT_FLOW] Step 3: Converting to filter keys")
                
                # Convert to filter keys - this will only convert hard specs
                # Returns: (filter_prefs dict, set of processed keys)
                filter_prefs, processed_keys = self._convert_preferences_to_filter_keys(preferences)
                
                has_hard_specs = bool(processed_keys)
                
                if has_hard_specs:
                    print(f"[CHAT_FLOW] Detected hard spec preferences: {processed_keys}")
                
                # Step 3b: Only do semantic search if we have enough prompts to show recommendations
                # Skip expensive API calls during clarification phase
                semantic_matching_skus = []
                filter_type = "semantic"  # Track which filtering method was used
                
                if self.product_prompt_count < self.min_prompts_for_recommendation:
                    print(f"[CHAT_FLOW] Step 3b: Skipping semantic search (not enough prompts yet: {self.product_prompt_count}/{self.min_prompts_for_recommendation})")
                else:
                    print(f"[CHAT_FLOW] Step 3b: Querying embeddings for semantic similarity + hard spec filtering")
                    
                    if self.semantic_rag:
                        where_filter = self._build_chromadb_where_filter(filter_prefs)
                        
                        # Determine query for semantic search and reranking
                        # Use accumulated semantic query if available, otherwise use current input
                        if self.accumulated_semantic_query:
                            query_for_rerank = self.accumulated_semantic_query
                            print(f"[CHAT_FLOW] Using accumulated semantic query: '{query_for_rerank}'")
                        else:
                            query_for_rerank = user_input
                            print(f"[CHAT_FLOW] No accumulated semantic context, using current query: '{user_input}'")
                        
                        # Decide whether to use reranking
                        # Use reranking if we have semantic context accumulated
                        use_rerank = bool(self.accumulated_semantic_query)
                        
                        # Query the user input directly against enriched embeddings
                        # This works even without explicit semantic preferences
                        # Returns list of tuples: [(sku, similarity), ...]
                        semantic_matches = self.semantic_rag.find_semantic_matches(
                            query=query_for_rerank,
                            top_k=5,
                            use_rerank=use_rerank
                        )

                        # Extract just the SKUs from the tuples
                        semantic_matching_skus = [sku for sku, _ in semantic_matches]

                        # Filter results by hard specs
                        spec_matching_skus: List[str] = []
                        if where_filter:
                            print(f"[CHAT_FLOW] Step 3b: Applying hard spec filters to semantic results")
                            spec_matching_skus = self.semantic_rag.filter_by_specs(where_filter, top_k=5)

                            if semantic_matching_skus:
                                spec_sku_set = set(spec_matching_skus)
                                # Keep only semantic results that also match hard specs (intersection)
                                semantic_matching_skus = [sku for sku in semantic_matching_skus if sku in spec_sku_set]
                                if semantic_matching_skus:
                                    filter_type = "hybrid"  # Both semantic and hard specs matched

                            if not semantic_matching_skus and spec_matching_skus:
                                print("[CHAT_FLOW] Step 3b: No semantic overlap, falling back to hard spec matches")
                                semantic_matching_skus = spec_matching_skus
                                filter_type = "hard_specs"  # Only hard specs matched
                        
                        print(f"[CHAT_FLOW] Step 3b result: Found {len(semantic_matching_skus)} matches (semantic + hard specs)")
                        print(f"[CHAT_FLOW] Filter type used: {filter_type}")
                    else:
                        print(f"[CHAT_FLOW] Step 3b: SemanticRAG not initialized, skipping semantic search")
                
                matching_products = []
                print(f"[CHAT_FLOW] Step 4: Checking context - context={bool(context)}, has products={bool(context and 'products' in context)}")
                if context and 'products' in context:
                    num_products = len(context['products']) if context['products'] else 0
                    print(f"[CHAT_FLOW] Step 4: Filtering {num_products} products")
                    if num_products > 0:
                        # Get full product objects for matched SKUs
                        if semantic_matching_skus:
                            print(f"[CHAT_FLOW] Step 4: Using ChromaDB semantic + hard spec results")
                            # Create a SKU->product lookup to preserve reranked order
                            sku_to_product = {p.get('SKU'): p for p in context['products']}
                            # Build matching_products in the RERANKED order
                            matching_products = [
                                sku_to_product[sku] for sku in semantic_matching_skus
                                if sku in sku_to_product
                            ]
                        else:
                            print(f"[CHAT_FLOW] Step 4: No matches found")
                        print(f"[CHAT_FLOW] Step 4 result: Found {len(matching_products)} matching products")
                        # Log top 3 products to verify order
                        if matching_products:
                            print(f"[CHAT_FLOW] Top 3 products (in reranked order):")
                            for i, p in enumerate(matching_products[:3], 1):
                                print(f"[CHAT_FLOW]   {i}. {p.get('Name')} (SKU: {p.get('SKU')})")
                    else:
                        print(f"[CHAT_FLOW] Step 4: Products list is empty!")
                else:
                    print(f"[CHAT_FLOW] Step 4: No products in context!")
                
                # Step 5: Generate and stream response
                # Check if we have enough prompts to show recommendations
                if self.product_prompt_count < self.min_prompts_for_recommendation:
                    print(f"[CHAT_FLOW] Not enough prompts yet ({self.product_prompt_count}/{self.min_prompts_for_recommendation}), using clarification response")
                    # If no matches found, inform user and ask to refine preferences
                    if not matching_products:
                        print(f"[CHAT_FLOW] No matching products found during clarification phase - informing user")
                        yield from self._generate_no_matches_response(
                            query=user_input,
                            preferences=preferences,
                            semantic_context=self.accumulated_semantic_query
                        )
                    else:
                        # Use clarification response to gather more information
                        yield from self._generate_clarification_response(
                            query=user_input,
                            preferences=preferences,
                            semantic_context=self.accumulated_semantic_query
                        )
                else:
                    print(f"[CHAT_FLOW] Sufficient prompts ({self.product_prompt_count}/{self.min_prompts_for_recommendation}), showing recommendations")
                    # If no matches found, inform user and ask to refine preferences
                    if not matching_products:
                        print(f"[CHAT_FLOW] No matching products found - informing user to refine preferences")
                        yield from self._generate_no_matches_response(
                            query=user_input,
                            preferences=preferences,
                            semantic_context=self.accumulated_semantic_query
                        )
                    else:
                        # Show recommendations
                        yield from self._generate_recommendation_response(
                            query=user_input,
                            products=matching_products,
                            preferences=preferences,
                            context=context,
                            semantic_context=self.accumulated_semantic_query,
                            filter_type=filter_type
                        )
                
            else:
                # For non-product related intents, use the memory-based response
                print(f"Processing non-product intent: {intent}")
                # Use the same full_context we constructed (includes message_history) to avoid losing context
                yield from self.stream_response_with_memory(
                    query=user_input,
                    memory_context=full_context,
                    context=context
                )
                
        except Exception as e:
            error_msg = f"Error in chat_stream: {str(e)}"
            print(error_msg)
            yield "I'm sorry, I encountered an error processing your request. Please try again."
            yield error_msg
        finally:
            # Print API call totals for this turn
            self._print_api_totals("chat_stream")

    ### Intent detection ###

    def _check_sufficient_info(self, query: str, memory_context) -> bool:
        """Check if we have enough information to recommend products.
        
        Args:
            query: The user's query
            memory_context: Can be a string or list of messages from the conversation
            
        Returns:
            bool: True if we have enough details about the user's requirements
        """
        # Check for key information in the current query
        required_phrases = [
            'recommend', 'suggest', 'find', 'looking for', 'need',
            'what would you recommend', 'what do you suggest',
            'which one', 'what are my options', 'help me choose'
        ]
        
        # Convert memory_context to string if it's a list
        if isinstance(memory_context, list):
            # Extract content from message objects if they're in LangChain format
            try:
                memory_text = " ".join(
                    msg.content if hasattr(msg, 'content') else 
                    str(msg) for msg in memory_context
                )
            except Exception:
                memory_text = str(memory_context)
        else:
            memory_text = str(memory_context)
        
        # Check if the query contains any of the required phrases
        query_lower = query.lower()
        has_trigger_phrase = any(phrase in query_lower for phrase in required_phrases)
        
        # Check if we have enough context in the conversation
        context_keywords = ['budget', 'price', 'need', 'looking for', 'purpose', 'use case', 'feature']
        memory_lower = memory_text.lower()
        has_context = any(keyword in memory_lower for keyword in context_keywords)
        
        return has_trigger_phrase and has_context

    def _detect_intent_llm(self, query: str, memory_context=None) -> str:
        """Use the LLM to classify high-level intent for product recommendations.

        Args:
            query: The user's query (only this is used for intent detection)
            memory_context: Unused - kept for backward compatibility
            
        Returns:
            str: One of: greeting, chitchat, nonsense, product_inquiry, recommendation_request, 
                 feature_question, preferences_given. Falls back to 'greeting' on any failure.
                 
        Note: preferences_given is returned when the user mentions ANY of these specifications:
              - Brand (e.g., Dell, HP, Lenovo)
              - CPU (e.g., Intel Core i7, AMD Ryzen 7)
              - GPU (e.g., NVIDIA RTX 3060, Integrated)
              - RAM (e.g., 8GB, 16GB, 32GB)
              - Storage (e.g., 256GB SSD, 1TB HDD)
              - Maximum price (e.g., $1000, budget 2000)
        """
        print(f"\n[INTENT_LLM] Detecting intent for query: '{query}'")
        
        # Handle empty queries
        if not query or not query.strip():
            print(f"[INTENT_LLM] Empty query detected, returning 'nonsense'")
            return 'nonsense'
        
        try:
            instruction = (
                "Classify the user's intent as exactly one of: "
                "greeting, chitchat, nonsense, product_inquiry, recommendation_request, preferences_given.\n\n"
                "Definitions:\n"
                "- greeting: Simple hello/hi/hey or other greetings\n"
                "- chitchat: Casual small talk or off-topic conversation\n"
                "- nonsense: Empty, gibberish, or meaningless input\n"
                "- product_inquiry: General questions about products (e.g., 'what laptops do you have')\n"
                "- recommendation_request: Asking for product recommendations based on needs/features\n"
                "  * Includes queries about specific features (battery life, portability, gaming, etc.)\n"
                "  * Examples: 'I need a laptop for gaming', 'laptop with long battery life', 'portable laptop'\n"
                "- preferences_given: User mentions ANY of these laptop specifications:\n"
                "  * Brand/Brand (e.g., Dell, HP, Lenovo, ASUS, Acer)\n"
                "  * CPU/Processor (e.g., Intel Core i7, AMD Ryzen 7, Intel i5)\n"
                "  * GPU/Graphics (e.g., NVIDIA RTX 3060, Integrated, AMD Radeon)\n"
                "  * RAM/Memory (e.g., 8GB, 16GB, 32GB)\n"
                "  * Storage (e.g., 256GB SSD, 1TB HDD, 512GB)\n"
                "  * Price/Budget (e.g., $1000, 2000$, budget 200, under 1500)\n\n"
                "Examples:\n"
                "- '2000$' -> preferences_given (mentions price)\n"
                "- 'budget 200' -> preferences_given (mentions budget)\n"
                "- 'Dell' -> preferences_given (mentions brand)\n"
                "- 'Intel i7' -> preferences_given (mentions CPU)\n"
                "- 'RTX 3060' -> preferences_given (mentions GPU)\n"
                "- '16GB RAM' -> preferences_given (mentions RAM)\n"
                "- '512GB SSD' -> preferences_given (mentions storage)\n"
                "- 'hi' -> greeting\n"
                "- 'hello' -> greeting\n"
                "- 'laptop with long battery life' -> recommendation_request\n"
                "- 'I need a laptop for gaming' -> recommendation_request\n"
                "- 'recommend a laptop' -> recommendation_request\n"
                "- 'what laptops do you have' -> product_inquiry\n\n"
                f"User query: {query}\n\n"
                "Respond ONLY with the intent word (one word, lowercase, no punctuation)."
            )
            
            print(f"[INTENT_LLM] Sending to LLM for classification")
            
            # Get a key for the LLM call and log the API call
            provider = os.getenv("LLM_PROVIDER", "cohere").lower()
            if provider == "cohere":
                _intent_key, _intent_idx = self._keybank.get_key_with_index("intent_detection")
                self._log_api_call("chat", "COHERE_API_KEY", note="intent_detection", key_index=_intent_idx)
            else:
                _intent_key = self._get_api_key_for_provider()
                self._log_api_call("chat", "GOOGLE_API_KEY", note="intent_detection")
            
            _intent_llm = self._create_llm_instance(
                api_key=_intent_key,
                temperature=0.1,
                max_tokens=10
            )
            
            # Call the LLM using stream to avoid token_count issues
            print(f"[INTENT_LLM] Calling LLM with stream method")
            response_text = ""
            try:
                for chunk in _intent_llm.stream([HumanMessage(content=instruction)]):
                    # The provider's stream method yields strings directly
                    response_text += chunk
                
                response_text = response_text.strip().lower()
                print(f"[INTENT_LLM] LLM raw response: '{response_text}'")
            except Exception as response_error:
                print(f"[INTENT_LLM] Error calling LLM stream: {response_error}")
                raise
            
            # Extract and clean the intent from the response
            intent = response_text.split('\n')[0].strip()
            intent = ''.join(c for c in intent if c.isalpha() or c == '_')
            print(f"[INTENT_LLM] Cleaned intent: '{intent}'")
            
            # Validate the intent
            valid_intents = {
                'greeting', 'chitchat', 'nonsense', 
                'product_inquiry', 'recommendation_request', 
                'feature_question', 'preferences_given'
            }
            
            if intent not in valid_intents:
                print(f"[INTENT_LLM] Invalid intent '{intent}', defaulting to 'greeting'")
                return 'greeting'
                
            print(f"[INTENT_LLM] Final intent: {intent}")
            return intent
            
        except Exception as e:
            print(f"[INTENT_LLM] Error in intent detection: {str(e)}")
            return 'greeting'  # Default to greeting on error

    ### Generation (LLM output) ###

    def stream_response_with_memory(self, query: str, memory_context: str = "", context: Optional[Dict] = None):
        """Generate streaming response using LangChain with conversation memory.
        
        Args:
            query: The user's message
            memory_context: Summary of previous conversation
            context: Additional context (e.g., product info)
        """
        import time
        try:
            print("🔄 Starting response generation...", flush=True)
            # Truncate memory to stay within token limits
            try:
                mem_budget = int(os.getenv("OFERGPT_MEMORY_CHAR_BUDGET", "700"))
            except Exception:
                mem_budget = 700
                
            mem_text = memory_context if memory_context else "This is the start of our conversation."
            mem_text = self._truncate(mem_text, mem_budget)
            
            # Check if we have enough information to recommend products
            has_sufficient_info = self._check_sufficient_info(query, mem_text)
            
            # Prepare the base prompt
            base_prompt = self.system_prompt
            
            # Only include product details if we have sufficient information
            if has_sufficient_info and context and 'products' in context:
                base_prompt += f"\n\nAvailable products for recommendation:\n{context['products']}"
            
            prompt = f"{base_prompt}\n\nPrevious conversation summary:\n{mem_text}\n\nUser question: {query}\n\nPlease provide a helpful response based on our previous conversation:"
            try:
                print("🚀 Starting LangChain streaming...", flush=True)
                # Real-time streaming only; global timeout is enforced by the caller (frontend)
                full_response = ""
                token_count = 0
                
                # Determine if we should use key rotation (Cohere only)
                provider = os.getenv("LLM_PROVIDER", "cohere").lower()
                max_tries = self._keybank.key_count() if provider == "cohere" else 1
                last_err = None
                
                for _ in range(max_tries):
                    # For Google, get API key once; for Cohere, rotate through keys
                    if provider == "cohere":
                        _stream_key, _stream_idx = self._keybank.get_key_with_index("chat_stream")
                        self._log_api_call("chat", "COHERE_API_KEY", note="chat_stream", key_index=_stream_idx)
                    else:
                        _stream_key = self._get_api_key_for_provider()
                        _stream_idx = None
                        self._log_api_call("chat", "GOOGLE_API_KEY", note="chat_stream")
                    
                    _stream_llm = self._create_llm_instance(
                        api_key=_stream_key,
                        temperature=0.35,
                        max_tokens=300
                    )
                    try:
                        for chunk in _stream_llm.stream([HumanMessage(content=prompt)]):
                            # The provider's stream method yields strings directly
                            if chunk:
                                full_response += chunk
                                token_count += 1
                                yield chunk
                        print(f"✅ Streaming completed: {token_count} chunks", flush=True)
                        if token_count == 0:
                            raise RuntimeError("Streaming returned 0 chunks; forcing fallback to non-streaming generation.")
                        # Save assistant response to memory so future turns have context
                        try:
                            if full_response:
                                self.memory.chat_memory.add_ai_message(full_response)
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        last_err = e
                        try:
                            error_msg = f"[CHAT][ERROR] note=chat_stream"
                            if _stream_idx is not None:
                                error_msg += f" key_index={_stream_idx}"
                            error_msg += f" error={e}"
                            print(error_msg, flush=True)
                        except Exception:
                            pass
                        # Only penalize Cohere keys
                        if provider == "cohere" and _stream_idx is not None:
                            try:
                                self._keybank.penalize_key(_stream_idx, seconds=1.5)
                            except Exception:
                                pass
                        continue
                else:
                    raise RuntimeError(f"stream_failed_all_keys: {last_err}")
            except Exception as stream_error:
                print(f"❌ Streaming failed: {stream_error}", flush=True)
                # Remove simulated streaming fallback; propagate to outer handler
                raise
        except Exception as e:
            print(f"Error generating streaming response: {e}", flush=True)
            msg = (
                "I'm sorry, I'm having trouble generating a response right now. "
                "I am using a trial key, which is limited to 10 API calls/minute. "
                "Please try again in a few seconds."
            )
            print("🎭 Simulating streaming of error message", flush=True)
            for i, word in enumerate(msg.split(" ")):
                yield (("" if i == 0 else " ") + word)
                time.sleep(0.03)
            print("✅ Error message streaming completed", flush=True)

    # Removed async chat_with_memory method - using chat_stream instead

    ### Memory management ###

    def clear_conversation_memory(self):
        """Clear only the conversation memory, keep knowledge base."""
        self.memory.clear()
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())