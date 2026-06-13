import json
import os
from urllib import error, request

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import torch
from tqdm import tqdm
from ..utils import Response

try:
    from src.config.services.ml_config import settings as ml_settings
except Exception:
    ml_settings = None


class Translator:
    def __init__(self, model_name="facebook/wmt19-en-ru", device=None):
        print('Initializing Translator...')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = FSMTTokenizer.from_pretrained(model_name)
        self.model = FSMTForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()  # отключаем режим обучения

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total model parameters: {total_params}")


    def translate(self, text: str) -> str:
        with torch.no_grad():  # отключаем вычисление градиентов
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_translate(self, texts: list, max_length: int = 512, batch_size: int = 16) -> list:
        """
        Батчевая генерация с tqdm и разбивкой на подбатчи для экономии памяти.
        """
        translations = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                encoded = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(self.device)
                outputs = self.model.generate(**encoded)
                translations.extend([self.tokenizer.decode(t, skip_special_tokens=True) for t in outputs])
        return translations


import torch
from tqdm import tqdm
from transformers import (
    FSMTForConditionalGeneration, FSMTTokenizer,
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)


import torch
from tqdm import tqdm
from transformers import (
    FSMTForConditionalGeneration, FSMTTokenizer,
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    AutoTokenizer, AutoModelForCausalLM
)


class OpenAICompatibleTranslatorProvider:
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        timeout: int = 120,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.base_url = (base_url or "http://127.0.0.1:8080").rstrip("/")
        self.api_key = api_key or "sk-no-key-required"
        self.system_prompt = system_prompt or (
            "Translate the user text from English to Russian. "
            "Return only the translated text."
        )
        self.timeout = timeout
        self.temperature = temperature

    def translate(self, text: str, max_new_tokens: int = 256) -> Response:
        if not text.strip():
            return Response(True, None, text)
        try:
            translation = self._translate_with_chat_completions(text, max_new_tokens)
            return Response(True, None, translation)
        except Exception as chat_error:
            try:
                translation = self._translate_with_completions(text, max_new_tokens)
                return Response(True, None, translation)
            except Exception as completion_error:
                return Response(
                    False,
                    f"chat_completions_error={chat_error}; completions_error={completion_error}",
                    None,
                )

    def batch_translate(
        self,
        texts: list,
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ) -> Response:
        del batch_size
        if not texts:
            return Response(True, None, [])
        results = []
        for text in tqdm(texts, desc="Translating batches"):
            response = self.translate(text, max_new_tokens=max_new_tokens)
            if response.status is False:
                return response
            results.append(response.result)
        return Response(True, None, results)

    def _translate_with_chat_completions(self, text: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": max_new_tokens,
        }
        response = self._post_json("/v1/chat/completions", payload)
        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        translation = str(content).strip()
        if not translation:
            raise ValueError("Empty translation returned from chat completions")
        return translation

    def _translate_with_completions(self, text: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model_name,
            "prompt": (
                f"{self.system_prompt}\n\n"
                f"Text:\n{text}\n\n"
                "Translation:"
            ),
            "temperature": self.temperature,
            "max_tokens": max_new_tokens,
        }
        response = self._post_json("/v1/completions", payload)
        translation = str(response["choices"][0].get("text", "")).strip()
        if not translation:
            raise ValueError("Empty translation returned from completions")
        return translation

    def _post_json(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"******"
        http_request = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {exc.code} returned by {self.base_url}{path}: {details}"
            ) from exc


class UniversalTranslator:
    """
    Универсальный переводчик, поддерживающий:

    - facebook/wmt19-en-ru (FSMT)
    - rinkorn/marian-finetuned-opus100-en-to-ru (MarianMT)
    - utrobinmv/t5_translate_en_ru_zh_small_1024 (T5)
    - NiuTrans/LMT-60-8B (Causal LM chat model)
    """

    def __init__(self, model_name: str, device: str = None, model_type: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.provider = None
        resolved_model_type = model_type or self._detect_model_type(model_name)
        self.model_type = resolved_model_type

        # -----------------------------------------
        # Auto-detection of model type
        # -----------------------------------------
        if resolved_model_type == 'fsmt':
            self.tokenizer = FSMTTokenizer.from_pretrained(model_name)
            self.model = FSMTForConditionalGeneration.from_pretrained(model_name)

        elif resolved_model_type == 'marian':
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)

        elif resolved_model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        elif resolved_model_type == 'chatlm':
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        elif resolved_model_type in {'openai', 'llama_cpp_openai'}:
            base_url = getattr(
                ml_settings,
                "TRANSLATOR_API_BASE",
                os.getenv("TRANSLATOR_API_BASE", os.getenv("OPENAI_BASE_URL")),
            )
            api_key = getattr(
                ml_settings,
                "TRANSLATOR_API_KEY",
                os.getenv("TRANSLATOR_API_KEY", os.getenv("OPENAI_API_KEY")),
            )
            system_prompt = getattr(
                ml_settings,
                "TRANSLATOR_SYSTEM_PROMPT",
                os.getenv("TRANSLATOR_SYSTEM_PROMPT"),
            )
            timeout = getattr(
                ml_settings,
                "TRANSLATOR_TIMEOUT",
                int(os.getenv("TRANSLATOR_TIMEOUT", "120")),
            )
            temperature = getattr(
                ml_settings,
                "TRANSLATOR_TEMPERATURE",
                float(os.getenv("TRANSLATOR_TEMPERATURE", "0")),
            )
            self.provider = OpenAICompatibleTranslatorProvider(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if self.provider is None:
            self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def _detect_model_type(model_name: str) -> str:
        model_name_lower = model_name.lower()
        if "wmt19" in model_name:
            return "fsmt"
        if "marian" in model_name_lower or "opus100" in model_name_lower:
            return "marian"
        if "t5" in model_name_lower:
            return "t5"
        if "LMT-60" in model_name or "NiuTrans" in model_name:
            return "chatlm"
        if any(
            marker in model_name_lower
            for marker in ("llama", "mistral", "qwen", "gemma", "openchat", "neural-chat")
        ):
            return "openai"
        if (
            os.getenv("TRANSLATOR_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or getattr(ml_settings, "TRANSLATOR_API_BASE", None)
        ):
            return "openai"
        return "unknown"

    def translate(self, text: str, max_new_tokens: int = 256) -> str:
        with torch.no_grad():
            if self.provider is not None:
                return self.provider.translate(text, max_new_tokens=max_new_tokens)

            if self.model_type == "t5":
                try:
                    encoded = self.tokenizer(
                        text, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)

                    output = self.model.generate(
                        encoded.input_ids,
                        max_new_tokens=max_new_tokens
                    )

                    text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    return Response(True, None, text)
                except Exception as e:
                    return Response(False, e, None)

            # ------------------------------
            # FSMT / Marian
            # ------------------------------
            try:
                encoded = self.tokenizer(
                    text, return_tensors="pt"
                ).to(self.device)

                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens
                )

                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                return Response(True, None, text)
            except Exception as e:
                return Response(False, e, None)

    # =======================================================================
    #               BATCH TRANSLATION
    # =======================================================================

    def batch_translate(self, texts: list, batch_size: int = 8, max_new_tokens: int = 256):
        results = []

        if self.provider is not None:
            return self.provider.batch_translate(
                texts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )

        # -----------------------------------------------------
        # LMT-60-8B: batch mode via chat template
        # -----------------------------------------------------
        if self.model_type == "chatlm":
            try:
                for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
                    batch = texts[i:i + batch_size]

                    messages_batch = []
                    for txt in batch:
                        prompt = (
                            "Translate the following text from English into Chinese.\n"
                            f"English: {txt}\n"
                            "Chinese:"
                        )
                        messages_batch.append(
                            self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": prompt}],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        )

                    encoded = self.tokenizer(
                        messages_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)

                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        num_beams=5,
                        do_sample=False
                    )

                    # remove prompts
                    for j, output_ids in enumerate(generated):
                        prompt_len = len(encoded.input_ids[j])
                        ans_ids = output_ids[prompt_len:]
                        results.append(
                            self.tokenizer.decode(ans_ids, skip_special_tokens=True)
                        )


                return Response(True, None, results)
            except Exception as e:
                return Response(False, e, None)

        # -----------------------------------------------------
        # Other models: FSMT / Marian / T5
        # -----------------------------------------------------
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Translating batches"):
                batch = texts[i:i + batch_size]

                with torch.no_grad():
                    encoded = self.tokenizer(
                        batch, return_tensors="pt", padding=True, truncation=True
                    ).to(self.device)

                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens
                    )

                    decoded = [
                        self.tokenizer.decode(t, skip_special_tokens=True)
                        for t in outputs
                    ]

                results.extend(decoded)

            return Response(True, None, results)
        except Exception as e:
            return Response(False, e, None)


from time import perf_counter
import json
def test_universal_translator(transcript_output_dir, translator: UniversalTranslator):
    start_time = perf_counter()
    with open(transcript_output_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)

        all_texts = []
        for page in data:
            for item in page:
                source_text = item['text']
                # translation = self.translator.translate(source_text)
                # item['translation'] = translation
                all_texts.append(source_text)
        translations = translator.batch_translate(all_texts, batch_size=32)
        # idx = 0
        # for page in data:
        #     for item in page:
        #         item['translation'] = translations[idx]
        #         idx += 1
        end_time = perf_counter()
        # print(f"⌛Функция Translator.batch_translate завершена | Время выполнения: {end_time - start_time:.4f} сек")

        return end_time - start_time

if __name__ == "__main__":
    # Пример использования
    tr_fsm = UniversalTranslator("facebook/wmt19-en-ru", device='mps')
    print(tr_fsm.translate("Hello world"))
    tr_marian = UniversalTranslator("glazzova/translation_en_ru", device='mps', model_type='marian')
    print(tr_marian.translate("Hello world"))
    # tr_t5 = UniversalTranslator("utrobinmv/t5_translate_en_ru_zh_small_1024", device='mps')
    # print(tr_t5.translate("Hello world"))
    tr_chatlm = UniversalTranslator("NiuTrans/LMT-60-0.6B-Base", device='mps')
    print(tr_chatlm.translate("Hello world"))

    transcript_output_dir = "var/tmp/test.vi.deo/ocr_transcript.json"
    # tr_t5_time = test_universal_translator(transcript_output_dir, tr_t5) 
    # tr_fsm_time = test_universal_translator(transcript_output_dir, tr_fsm)
    
    tr_marian_time = test_universal_translator(transcript_output_dir, tr_marian)
    
    # 
    # print(f"FSMT time: {tr_fsm_time:.4f} sec", f"Total parameters: {tr_fsm.total_params:,}")
    
    print(f"MarianMT glazzova/translation_en_ru time: {tr_marian_time:.4f} sec", f"Total parameters: {tr_marian.total_params:,}")
    # print(f"T5 time: {tr_t5_time:.4f} sec", f"Total parameters: {tr_t5.total_params:,}")


    # FSMT time: 143.4698 sec Total parameters: 293,195,776
    # MarianMT glazzova/translation_en_ru time: 63.0334 sec Total parameters: 76,672,000