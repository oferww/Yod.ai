import re
from typing import Dict, List, Any, Optional, Set, Union

from forex_python.converter import CurrencyRates, CurrencyCodes, RatesNotAvailableError


class LaptopFilter:
    _currency_client: Optional[CurrencyRates] = None
    _currency_client_failed: bool = False
    _currency_codes: Optional[CurrencyCodes] = None
    _currency_codes_failed: bool = False
    CURRENCY_ALIASES: Dict[str, str] = {
        '$': 'USD',
        'usd': 'USD',
        'dollar': 'USD',
        'dollars': 'USD',
        'nis': 'ILS',
        '₪': 'ILS',
        'ils': 'ILS',
        'shekel': 'ILS',
        'sheqel': 'ILS',
        '€': 'EUR',
        'eur': 'EUR',
        'euro': 'EUR',
        '£': 'GBP',
        'gbp': 'GBP',
        'pound': 'GBP',
        '£': 'GBP',
        'aud': 'AUD',
        'a$': 'AUD',
        'cad': 'CAD',
        'c$': 'CAD',
        'yen': 'JPY',
        '¥': 'JPY',
        'jpy': 'JPY',
    }

    @staticmethod
    def _parse_size_to_gb(size_str: str) -> float:
        """Convert size string to GB."""
        if not isinstance(size_str, str):
            return 0.0
            
        size_str = size_str.lower().replace(' ', '')
        
        # Remove all non-numeric suffixes (lpddr5, ddr5, ddr4, ssd, hdd, nvme, x, etc.)
        # Keep only the numeric part and unit (tb, gb)
        import re
        
        # Extract numeric value and unit
        match = re.search(r'(\d+(?:\.\d+)?)\s*(tb|gb)?', size_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            if unit == 'tb':
                return value * 1024
            else:  # gb or no unit specified
                return value
        
        return 0.0

    @staticmethod
    def _extract_brand(text: str, brands: List[str]) -> Optional[str]:
        """Extract brand from text."""
        text = text.lower()
        for brand in brands:
            if brand in text:
                return brand
        return None

    @classmethod
    def _get_currency_client(cls) -> Optional[CurrencyRates]:
        if cls._currency_client is None and not cls._currency_client_failed:
            try:
                cls._currency_client = CurrencyRates()
            except Exception as exc:
                print(f"[CURRENCY] Failed to initialize CurrencyRates: {exc}")
                cls._currency_client_failed = True
                cls._currency_client = None
        return cls._currency_client

    @classmethod
    def _get_currency_codes(cls) -> Optional[CurrencyCodes]:
        if cls._currency_codes is None and not cls._currency_codes_failed:
            try:
                cls._currency_codes = CurrencyCodes()
            except Exception as exc:
                print(f"[CURRENCY] Failed to initialize CurrencyCodes: {exc}")
                cls._currency_codes_failed = True
                cls._currency_codes = None
        return cls._currency_codes

    @classmethod
    def parse_price_to_usd(cls, value: Union[str, int, float, None]) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        text_lower = text.lower()
        detected_currency: Optional[str] = None

        # 1. Check common aliases first (covers ambiguous symbols like '$')
        for token, code in cls.CURRENCY_ALIASES.items():
            token_lower = token.lower()
            if token_lower.isalpha():
                if re.search(rf"\b{re.escape(token_lower)}\b", text_lower):
                    detected_currency = code
                    break
            else:
                if token in text or token_lower in text_lower:
                    detected_currency = code
                    break

        codes = cls._get_currency_codes()

        # 2. Try to detect currency symbols using forex-python lookup
        if not detected_currency and codes:
            seen_symbols = set()
            for char in text:
                if char.isdigit() or char.isalpha() or char.isspace() or char in {'.', ',', '-', '_'}:
                    continue
                if char in seen_symbols:
                    continue
                seen_symbols.add(char)
                try:
                    code = codes.get_currency_code_from_symbol(char)
                except Exception:
                    code = None
                if code:
                    detected_currency = code.upper()
                    break

        # 3. Try to detect ISO currency codes (three-letter sequences)
        if not detected_currency and codes:
            candidates = re.findall(r"\b([a-zA-Z]{3})\b", text)
            for candidate in candidates:
                iso_code = candidate.upper()
                try:
                    name = codes.get_currency_name(iso_code)
                except Exception:
                    name = None
                if name:
                    detected_currency = iso_code
                    break

        if not detected_currency:
            detected_currency = 'USD'

        # Clean the text from any detected currency indicators
        cleaned = text_lower
        for token in cls.CURRENCY_ALIASES.keys():
            token_lower = token.lower()
            if token_lower.isalpha():
                cleaned = re.sub(rf"\b{re.escape(token_lower)}\b", ' ', cleaned)
            else:
                cleaned = cleaned.replace(token_lower, ' ')

        if codes:
            try:
                symbol = codes.get_symbol(detected_currency)
            except Exception:
                symbol = None
            if symbol:
                cleaned = cleaned.replace(symbol.lower(), ' ')
        cleaned = re.sub(rf"\b{detected_currency.lower()}\b", ' ', cleaned)

        cleaned = cleaned.replace(',', ' ')
        cleaned = re.sub(r"[^0-9\.\skm]", ' ', cleaned)
        cleaned = re.sub(r"\s+", ' ', cleaned).strip()

        multiplier = 1.0
        match = re.search(r"(\d+(?:\.\d+)?)(?:\s*([km]))?", cleaned)
        if not match:
            return None

        try:
            amount = float(match.group(1))
        except ValueError:
            return None

        suffix = match.group(2)
        if suffix == 'k':
            multiplier = 1_000
        elif suffix == 'm':
            multiplier = 1_000_000

        amount *= multiplier

        if detected_currency == 'USD':
            return amount

        client = cls._get_currency_client()
        if not client:
            return None

        try:
            usd_value = client.convert(detected_currency, 'USD', amount)
        except RatesNotAvailableError:
            cls._currency_client_failed = True
            cls._currency_client = None
            return None
        except Exception as exc:
            print(f"[CURRENCY] Conversion error {detected_currency} -> USD: {exc}")
            cls._currency_client_failed = True
            cls._currency_client = None
            return None

        try:
            return float(usd_value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_token(token: str) -> str:
        return token.strip("(),:+")

    @staticmethod
    def _extract_spec_keywords(text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
        if not isinstance(text, str) or not text:
            return []

        stopwords = stopwords or set()
        keywords = set()
        text_lower = text.lower()
        raw_tokens = re.split(r"\s+", text_lower)

        previous_token = None
        for raw in raw_tokens:
            token = LaptopFilter._clean_token(raw)
            if not token or token in stopwords:
                previous_token = raw
                continue

            keywords.add(token)

            # Split by hyphen or slash to capture sub-parts (e.g., i9-13900h -> i9, 13900h)
            for part in re.split(r"[-/]", token):
                sub_token = part.strip()
                if sub_token and sub_token not in stopwords:
                    keywords.add(sub_token)

            # Add numeric portions as separate keywords for generation/model matching
            for digits in re.findall(r"\d+", token):
                if digits:
                    keywords.add(digits)

            # Combine with previous token to capture patterns like "rtx 4090"
            if previous_token:
                prev_clean = LaptopFilter._clean_token(previous_token)
                if prev_clean and prev_clean not in stopwords:
                    combo = f"{prev_clean} {token}".strip()
                    keywords.add(combo)

            previous_token = raw

        return sorted(keywords)

    @staticmethod
    def normalize_cpu_series(series: str) -> Optional[str]:
        if not isinstance(series, str):
            return None

        text = series.lower()
        text = text.replace('-', ' ')
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None

        # Intel Core Ultra series
        match = re.search(r"(core\s+)?ultra\s+([0-9]+)", text)
        if match:
            return f"core ultra {match.group(2)}"

        # Intel Core i-series (allow any single digit to cover i3/i5/i7/i9 etc.)
        match = re.search(r"(core\s+)?i\s*([0-9])", text)
        if match:
            return f"core i{match.group(2)}"

        # Intel Xeon W-series (w3, w5, w7, w9)
        match = re.search(r"xeon\s+w\s*([0-9])", text)
        if match:
            return f"xeon w{match.group(1)}"

        return text if 'core' in text or 'xeon' in text else None

    @classmethod
    def extract_cpu_series(cls, text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []

        lowered = text.lower()
        normalized = lowered.replace('-', ' ')
        normalized = normalized.replace('/', ' ')
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []

        tokens = normalized.split()
        candidates = set()

        for idx, token in enumerate(tokens):
            # Detect patterns like "core ultra 7"
            if token == 'core' and idx + 1 < len(tokens):
                next_token = tokens[idx + 1]
                if next_token == 'ultra' and idx + 2 < len(tokens):
                    digit = tokens[idx + 2]
                    if digit.isdigit():
                        normalized_series = cls.normalize_cpu_series(f"core ultra {digit}")
                        if normalized_series:
                            candidates.add(normalized_series)
                elif next_token.startswith('i'):
                    normalized_series = cls.normalize_cpu_series(f"core {next_token}")
                    if normalized_series:
                        candidates.add(normalized_series)

            # Detect standalone "ultra 7" sequences
            if token == 'ultra' and idx + 1 < len(tokens):
                digit = tokens[idx + 1]
                if digit.isdigit():
                    normalized_series = cls.normalize_cpu_series(f"ultra {digit}")
                    if normalized_series:
                        candidates.add(normalized_series)

            # Detect tokens like "i7"
            if token.startswith('i') and len(token) == 2 and token[1].isdigit():
                normalized_series = cls.normalize_cpu_series(token)
                if normalized_series:
                    candidates.add(normalized_series)

            # Detect Xeon W tokens like "w9"
            if token == 'xeon' and idx + 1 < len(tokens):
                next_token = tokens[idx + 1]
                if next_token.startswith('w') and len(next_token) >= 2 and next_token[1].isdigit():
                    normalized_series = cls.normalize_cpu_series(f"xeon {next_token[:2]}")
                    if normalized_series:
                        candidates.add(normalized_series)

        return sorted(candidates)

    @staticmethod
    def extract_cpu_keywords(text: str) -> List[str]:
        stopwords = {"intel", "amd", "apple", "processor", "cpu", "series", "chip"}
        return LaptopFilter._extract_spec_keywords(text, stopwords)

    @staticmethod
    def extract_gpu_keywords(text: str) -> List[str]:
        stopwords = {"graphics", "gpu"}
        return LaptopFilter._extract_spec_keywords(text, stopwords)

    @staticmethod
    def extract_storage_types(text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []

        lowered = text.lower()
        types: Set[str] = set()

        if 'nvme' in lowered:
            types.add('nvme')
            types.add('ssd')
        if 'ssd' in lowered:
            types.add('ssd')
        if 'hdd' in lowered:
            types.add('hdd')
        if 'hybrid' in lowered:
            types.add('hybrid')
        if 'pcie' in lowered:
            types.add('pcie')
        if 'sata' in lowered:
            types.add('sata')

        return sorted(types)

    @staticmethod
    def extract_ram_types(text: str) -> List[str]:
        if not isinstance(text, str) or not text:
            return []

        lowered = text.lower()
        types: Set[str] = set()

        if 'lpddr5x' in lowered:
            types.add('lpddr5x')
            types.add('lpddr5')
        if 'lpddr5' in lowered and 'lpddr5x' not in lowered:
            types.add('lpddr5')
        if 'lpddr4x' in lowered:
            types.add('lpddr4x')
        if 'ddr5 ecc' in lowered:
            types.add('ddr5 ecc')
            types.add('ddr5')
            types.add('ecc')
        elif 'ddr5' in lowered:
            types.add('ddr5')
        if 'ddr4' in lowered:
            types.add('ddr4')
        if 'unified' in lowered:
            types.add('unified')
        if 'ecc' in lowered and 'ddr5 ecc' not in lowered:
            types.add('ecc')

        return sorted(types)

    @staticmethod
    def extract_core_count(text: str) -> Optional[int]:
        if not isinstance(text, str):
            return None
        match = re.search(r"(\d+)\s*-\s*core", text.lower())
        if not match:
            match = re.search(r"(\d+)\s*core", text.lower())
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def extract_cpu_generation(text: str) -> Optional[int]:
        if not isinstance(text, str):
            return None

        lowered = text.lower()

        explicit = re.search(r"(\d+)\s*(?:th|st|nd|rd)\s+gen", lowered)
        if explicit:
            try:
                return int(explicit.group(1))
            except ValueError:
                pass

        apple = re.search(r"m\s*(\d)", lowered)
        if apple:
            try:
                return int(apple.group(1))
            except ValueError:
                pass

        # Look for 4-5 digit identifiers (e.g., 13900 -> generation 13)
        number = re.search(r"\b(\d{5})\b", lowered)
        if number:
            num = number.group(1)
            gen = int(num[:2])
            return gen if gen >= 10 else int(num[0])

        number = re.search(r"\b(\d{4})\b", lowered)
        if number:
            num = number.group(1)
            try:
                return int(num[0])
            except ValueError:
                return None

        number = re.search(r"\b(\d{3})\b", lowered)
        if number:
            num = number.group(1)
            try:
                return int(num[0])
            except ValueError:
                return None

        return None

    # deprecated
    # def __init__(self, products: List[Dict[str, Any]]):
    #     self.products = products
    #     self._prepare_products()

    # def _prepare_products(self):
    #     """Preprocess products for easier filtering."""
    #     for product in self.products:
    #         ram_text = product.get('RAM', '') or ''
    #         storage_text = product.get('Storage', '') or ''

    #         # Normalize RAM to GB
    #         if ram_text:
    #             product['_ram_gb'] = self._parse_size_to_gb(ram_text)
    #         product['_ram_text'] = ram_text.lower()
    #         product['_ram_types'] = self.extract_ram_types(ram_text)
            
    #         # Normalize Storage to GB
    #         if storage_text:
    #             product['_storage_gb'] = self._parse_size_to_gb(storage_text)
    #         product['_storage_text'] = storage_text.lower()
    #         product['_storage_types'] = self.extract_storage_types(storage_text)
            
    #         # Normalize Price to float (USD)
    #         if 'Price' in product:
    #             parsed_price = self.parse_price_to_usd(product['Price'])
    #             product['_price'] = parsed_price if parsed_price is not None else float('inf')
    #         else:
    #             product['_price'] = float('inf')
            
    #         cpu_text = product.get('CPU', '') or ''
    #         gpu_text = product.get('GPU', '') or ''

    #         # Extract CPU metadata
    #         product['_cpu_brand'] = self._extract_brand(cpu_text, ['intel', 'amd', 'apple'])
    #         product['_cpu_text'] = cpu_text.lower()
    #         cpu_series = self.extract_cpu_series(cpu_text)
    #         cpu_keywords = self.extract_cpu_keywords(cpu_text)
    #         if cpu_series:
    #             keyword_set = set(cpu_keywords)
    #             keyword_set.update(cpu_series)
    #             cpu_keywords = sorted(keyword_set)
    #         product['_cpu_keywords'] = cpu_keywords
    #         product['_cpu_series'] = cpu_series
    #         product['_cpu_core_count'] = self.extract_core_count(cpu_text)
    #         product['_cpu_generation'] = self.extract_cpu_generation(cpu_text)
            
    #         # Extract GPU metadata
    #         product['_gpu_brand'] = self._extract_brand(gpu_text, ['nvidia', 'amd', 'intel', 'apple'])
    #         product['_gpu_text'] = gpu_text.lower()
    #         product['_gpu_core_count'] = self.extract_core_count(gpu_text)
    #         product['_gpu_keywords'] = self.extract_gpu_keywords(gpu_text)

    # def filter_products(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        # """Filter products based on the given criteria."""
        # if not filters:
        #     return self.products[:5]  # Return first 5 if no filters
        
        # print(f"\n[FILTER] Starting filter with criteria: {filters}")
        # filtered = []
        # for product in self.products:
        #     matches = True
        #     product_name = product.get('Name', 'Unknown')
            
        #     # Brand filter
        #     if 'brand' in filters and filters['brand']:
        #         brand_filter = filters['brand'].lower()
        #         product_brand = product.get('Brand', '').lower()
        #         if brand_filter not in product_brand:
        #             matches = False
        #             print(f"[FILTER] {product_name}: Brand mismatch - filter={brand_filter}, product={product_brand}")
            
        #     # RAM filter
        #     if matches and 'min_ram_gb' in filters:
        #         product_ram = product.get('_ram_gb', 0)
        #         filter_ram = filters['min_ram_gb']
        #         if product_ram < filter_ram:
        #             matches = False
        #             print(f"[FILTER] {product_name}: RAM too low - filter={filter_ram}GB, product={product_ram}GB")

        #     # RAM type filter
        #     if matches:
        #         ram_type_filters = filters.get('ram_types') or filters.get('ram_type')
        #         if ram_type_filters:
        #             if isinstance(ram_type_filters, str):
        #                 ram_type_filters = [ram_type_filters]
        #             normalized_filters: Set[str] = set()
        #             for ram_type in ram_type_filters:
        #                 extracted = LaptopFilter.extract_ram_types(str(ram_type))
        #                 if extracted:
        #                     normalized_filters.update(extracted)
        #                 else:
        #                     normalized_filters.add(str(ram_type).lower().strip())
        #             product_ram_types = set(product.get('_ram_types', []))
        #             if not normalized_filters.issubset(product_ram_types):
        #                 matches = False
        #                 print(f"[FILTER] {product_name}: RAM type mismatch - filter={normalized_filters}, product_types={product_ram_types}")
                
        #     # Storage filter
        #     if matches and 'min_storage_gb' in filters:
        #         product_storage = product.get('_storage_gb', 0)
        #         filter_storage = filters['min_storage_gb']
        #         if product_storage < filter_storage:
        #             matches = False
        #             print(f"[FILTER] {product_name}: Storage too low - filter={filter_storage}GB, product={product_storage}GB")

        #     # Storage type filter
        #     if matches:
        #         storage_type_filters = filters.get('storage_types') or filters.get('storage_type')
        #         if storage_type_filters:
        #             if isinstance(storage_type_filters, str):
        #                 storage_type_filters = [storage_type_filters]
        #             normalized_filters: Set[str] = set()
        #             for storage_type in storage_type_filters:
        #                 extracted = LaptopFilter.extract_storage_types(str(storage_type))
        #                 if extracted:
        #                     normalized_filters.update(extracted)
        #                 else:
        #                     normalized_filters.add(str(storage_type).lower().strip())
        #             product_storage_types = set(product.get('_storage_types', []))
        #             if not normalized_filters.issubset(product_storage_types):
        #                 matches = False
        #                 print(f"[FILTER] {product_name}: Storage type mismatch - filter={normalized_filters}, product_types={product_storage_types}")
                
        #     # Price filter
        #     if matches and 'max_price' in filters:
        #         product_price = product.get('_price', float('inf'))
        #         filter_price = filters['max_price']
        #         if product_price > filter_price:
        #             matches = False
        #             print(f"[FILTER] {product_name}: Price too high - filter=${filter_price}, product=${product_price}")
                
        #     # CPU brand filter
        #     if matches and 'cpu_brand' in filters:
        #         cpu_brand = filters['cpu_brand'].lower()
        #         product_cpu_brand = product.get('_cpu_brand')
        #         if not (product_cpu_brand and cpu_brand in product_cpu_brand):
        #             matches = False
        #             print(f"[FILTER] {product_name}: CPU brand mismatch - filter={cpu_brand}, product={product_cpu_brand}")

        #     # CPU keywords filter (supports model/generation identifiers)
        #     if matches and 'cpu_keywords' in filters:
        #         cpu_keywords = filters['cpu_keywords']
        #         if isinstance(cpu_keywords, str):
        #             cpu_keywords = [cpu_keywords]
        #         cpu_text = product.get('_cpu_text', '')
        #         product_keywords = set(product.get('_cpu_keywords', []))
        #         for keyword in cpu_keywords:
        #             if not keyword:
        #                 continue
        #             keyword_lower = keyword.lower()
        #             if keyword_lower not in cpu_text and keyword_lower not in product_keywords:
        #                 matches = False
        #                 print(f"[FILTER] {product_name}: CPU keyword missing - keyword={keyword_lower}, product_cpu={cpu_text}")
        #                 break

        #     # CPU series filter (e.g., Core i7, Core Ultra 9)
        #     if matches and 'cpu_series' in filters:
        #         cpu_series_filters = filters['cpu_series']
        #         if isinstance(cpu_series_filters, str):
        #             cpu_series_filters = [cpu_series_filters]
        #         product_series = set(product.get('_cpu_series', []))
        #         for series in cpu_series_filters or []:
        #             normalized_series = self.normalize_cpu_series(series) if isinstance(series, str) else None
        #             if normalized_series:
        #                 check_value = normalized_series
        #             else:
        #                 check_value = str(series).lower().strip()
        #             if check_value not in product_series:
        #                 matches = False
        #                 print(f"[FILTER] {product_name}: CPU series mismatch - filter={check_value}, product_series={product_series}")
        #                 break

        #     # CPU minimum core count filter
        #     if matches and 'min_cpu_cores' in filters:
        #         required_cores = filters['min_cpu_cores']
        #         product_cores = product.get('_cpu_core_count') or 0
        #         if product_cores < required_cores:
        #             matches = False
        #             print(f"[FILTER] {product_name}: CPU cores too low - filter={required_cores}, product={product_cores}")

        #     # CPU minimum generation filter
        #     if matches and 'min_cpu_generation' in filters:
        #         required_generation = filters['min_cpu_generation']
        #         product_generation = product.get('_cpu_generation') or 0
        #         if product_generation < required_generation:
        #             matches = False
        #             print(f"[FILTER] {product_name}: CPU generation too low - filter={required_generation}, product={product_generation}")
                    
        #     # GPU brand filter
        #     if matches and 'gpu_brand' in filters:
        #         gpu_brand = filters['gpu_brand'].lower()
        #         product_gpu_brand = product.get('_gpu_brand')
        #         if not (product_gpu_brand and gpu_brand in product_gpu_brand):
        #             matches = False
        #             print(f"[FILTER] {product_name}: GPU brand mismatch - filter={gpu_brand}, product={product_gpu_brand}")

        #     # GPU keywords filter (model / tier matching)
        #     if matches and 'gpu_keywords' in filters:
        #         gpu_keywords = filters['gpu_keywords']
        #         if isinstance(gpu_keywords, str):
        #             gpu_keywords = [gpu_keywords]
        #         gpu_text = product.get('_gpu_text', '')
        #         product_keywords = set(product.get('_gpu_keywords', []))
        #         for keyword in gpu_keywords:
        #             if not keyword:
        #                 continue
        #             keyword_lower = keyword.lower()
        #             if keyword_lower not in gpu_text and keyword_lower not in product_keywords:
        #                 matches = False
        #                 print(f"[FILTER] {product_name}: GPU keyword missing - keyword={keyword_lower}, product_gpu={gpu_text}")
        #                 break

        #     # GPU minimum core count filter
        #     if matches and 'min_gpu_cores' in filters:
        #         required_cores = filters['min_gpu_cores']
        #         product_cores = product.get('_gpu_core_count') or 0
        #         if product_cores < required_cores:
        #             matches = False
        #             print(f"[FILTER] {product_name}: GPU cores too low - filter={required_cores}, product={product_cores}")
                    
        #     if matches:
        #         print(f"[FILTER] {product_name}: MATCH!")
        #         filtered.append(product)
        #         if len(filtered) >= 5:  # Limit to top 5 matches
        #             break
        
        # print(f"[FILTER] Found {len(filtered)} matching products")
        # return filtered
    